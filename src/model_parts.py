"""
Split ``models/random_forest.pkl`` so it fits inside GitHub's 100 MB file limit.

The trained random forest is ~544 MB and GitHub rejects any single file over
100 MB, so it cannot be pushed as-is. It is the only artifact in this project
with that problem — everything else in ``models/`` is under 50 MB and commits
directly — so this module deals with that one file and nothing else.

``split`` rewrites it with joblib compression (544 MB -> 140 MB) and slices the
compressed bytes into numbered parts small enough to commit, recording a
checksum in a manifest. The original file is left untouched.

``join`` concatenates the parts back into a loadable .pkl and refuses to keep
the result if the checksum does not match. Compression is lossless, so the
rebuilt file unpickles to exactly the same estimator.

    models/random_forest.pkl              # local only, gitignored
    models/parts/random_forest.pkl.001    # committed, 45 MB each
    models/parts/random_forest.pkl.002    # committed
    models/parts/random_forest.pkl.003    # committed
    models/parts/random_forest.pkl.004    # committed
    models/parts/random_forest.pkl.manifest.json

Command line
------------
Run from the repo root (``apartment_rental_price_prediction/``)::

    python -m src.model_parts join      # rebuild the .pkl after cloning
    python -m src.model_parts split     # re-split it after retraining

``split`` also takes ``--chunk-mb`` (default 45, below GitHub's 50 MB warning)
and ``--compress`` (joblib level 0-9, default 3; 0 splits the file as it is).

From Python
-----------
``ensure_available()`` rebuilds the forest from its parts if the .pkl is
missing, which is how ``app/streamlit_app.py`` survives a fresh clone::

    from src.model_parts import ensure_available
    random_forest = joblib.load(ensure_available())
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "random_forest.pkl"
PARTS_DIR = MODELS_DIR / "parts"
MANIFEST_PATH = PARTS_DIR / f"{MODEL_PATH.name}.manifest.json"

# GitHub hard-rejects files above 100 MB and warns above 50 MB, so the default
# part size stays under both. Parts are not compressed individually — the whole
# file is compressed once before slicing, which compresses far better.
DEFAULT_CHUNK_MB = 45
DEFAULT_COMPRESS = 3

MB = 1024 * 1024
READ_BLOCK = 8 * MB


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _megabytes(path: Path) -> float:
    return path.stat().st_size / MB


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_model(
    chunk_mb: int = DEFAULT_CHUNK_MB,
    compress: int = DEFAULT_COMPRESS,
    verbose: bool = True,
) -> Path:
    """Compress the random forest and slice it into committable parts.

    ``compress=0`` skips the compression step and splits the file as it stands.
    Returns the manifest path.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No such model file: {MODEL_PATH}. Train it with "
            "notebooks/03_modelling.ipynb, or run `join` if you have the parts."
        )

    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Rewriting through joblib is what shrinks the file; the staged copy is also
    # what gets checksummed, so the manifest always describes the rebuilt bytes.
    staged = PARTS_DIR / f"{MODEL_PATH.name}.staged"
    if compress:
        if verbose:
            print(f"Compressing {MODEL_PATH.name} ({_megabytes(MODEL_PATH):.1f} MB)…")
        joblib.dump(joblib.load(MODEL_PATH), staged, compress=compress)
        # Read it straight back: a part set that cannot be unpickled is worse
        # than no part set at all, and this is the only cheap moment to notice.
        joblib.load(staged)
        if verbose:
            print(f"  compressed to {_megabytes(staged):.1f} MB")
    else:
        staged.write_bytes(MODEL_PATH.read_bytes())

    checksum = _sha256(staged)
    total_size = staged.stat().st_size

    for stale in sorted(PARTS_DIR.glob(f"{MODEL_PATH.name}.[0-9][0-9][0-9]")):
        stale.unlink()

    chunk_bytes = chunk_mb * MB
    parts: list[dict] = []
    with open(staged, "rb") as source:
        while True:
            block = source.read(chunk_bytes)
            if not block:
                break
            part = PARTS_DIR / f"{MODEL_PATH.name}.{len(parts) + 1:03d}"
            part.write_bytes(block)
            parts.append({"name": part.name, "bytes": len(block)})
            if verbose:
                print(f"  wrote {part.name} ({len(block) / MB:.1f} MB)")

    staged.unlink()

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "file": MODEL_PATH.name,
                "bytes": total_size,
                "sha256": checksum,
                "compress": compress,
                "chunk_mb": chunk_mb,
                "parts": parts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if verbose:
        print(f"Split {MODEL_PATH.name} into {len(parts)} parts -> {PARTS_DIR}")
    return MANIFEST_PATH


# ---------------------------------------------------------------------------
# Joining
# ---------------------------------------------------------------------------
def join_model(verbose: bool = True) -> Path:
    """Rebuild the random forest from its parts and verify the checksum."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"No manifest at {MANIFEST_PATH} — nothing to rebuild from."
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    absent = [
        part["name"] for part in manifest["parts"] if not (PARTS_DIR / part["name"]).exists()
    ]
    if absent:
        raise FileNotFoundError(
            f"Missing parts: {', '.join(absent)}. Pull the full repository or "
            "re-run the split."
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Assembled under a temporary name and renamed at the end, so an interrupted
    # join never leaves a half-written .pkl that looks loadable.
    staging = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".joining")
    with open(staging, "wb") as target:
        for part in manifest["parts"]:
            with open(PARTS_DIR / part["name"], "rb") as source:
                for block in iter(lambda: source.read(READ_BLOCK), b""):
                    target.write(block)

    if _sha256(staging) != manifest["sha256"]:
        staging.unlink()
        raise ValueError(
            "Checksum mismatch: the parts are corrupt or out of date. Re-run the "
            "split on the machine that has the original file."
        )

    os.replace(staging, MODEL_PATH)
    if verbose:
        print(f"Rebuilt {MODEL_PATH.name} ({_megabytes(MODEL_PATH):.1f} MB) from parts")
    return MODEL_PATH


def parts_available() -> bool:
    """True if the forest has been split and can be rebuilt from its parts."""
    return MANIFEST_PATH.exists()


def ensure_available(verbose: bool = False) -> Path:
    """Return the model path, rebuilding it from parts first if it is missing."""
    if not MODEL_PATH.exists():
        join_model(verbose=verbose)
    return MODEL_PATH


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser(
        "split", help="compress and split models/random_forest.pkl"
    )
    split_parser.add_argument("--chunk-mb", type=int, default=DEFAULT_CHUNK_MB)
    split_parser.add_argument(
        "--compress", type=int, default=DEFAULT_COMPRESS, help="joblib level 0-9 (0 disables)"
    )

    subparsers.add_parser("join", help="rebuild models/random_forest.pkl from its parts")

    args = parser.parse_args()

    if args.command == "split":
        split_model(chunk_mb=args.chunk_mb, compress=args.compress)
    else:
        join_model()


if __name__ == "__main__":
    main()
