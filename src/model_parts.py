"""
Split oversized model artifacts so they fit inside GitHub's 100 MB file limit.

``models/random_forest.pkl`` is ~570 MB — GitHub rejects any single file over
100 MB, so it cannot be pushed as-is. This module rewrites such a file with
joblib compression (570 MB -> ~140 MB for the forest, same object on load) and
then slices the compressed bytes into numbered parts that are small enough to
commit. A manifest records the checksum so the rebuilt file can be verified.

    models/random_forest.pkl              # local only, gitignored
    models/parts/random_forest.pkl.001    # committed
    models/parts/random_forest.pkl.002    # committed
    models/parts/random_forest.pkl.manifest.json

Command line
------------
Run from the repo root (``apartment_rental_price_prediction/``)::

    python -m src.model_parts status              # what is too big / already split
    python -m src.model_parts split --all         # split every oversized model
    python -m src.model_parts split models/random_forest.pkl
    python -m src.model_parts join --all          # rebuild after cloning

From Python
-----------
``ensure_available(path)`` rebuilds a model from its parts if the .pkl is
missing, which is how ``app/streamlit_app.py`` survives a fresh clone::

    from src.model_parts import ensure_available
    model = joblib.load(ensure_available(MODELS_DIR / "random_forest.pkl"))
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PARTS_DIR_NAME = "parts"

# GitHub hard-rejects files above 100 MB and warns above 50 MB, so the default
# part size stays under both. Parts are not compressed individually — the whole
# file is compressed once before slicing, which compresses far better.
GITHUB_LIMIT_MB = 100
DEFAULT_CHUNK_MB = 45
DEFAULT_COMPRESS = 3

MB = 1024 * 1024
READ_BLOCK = 8 * MB


# ---------------------------------------------------------------------------
# Paths and checksums
# ---------------------------------------------------------------------------
def parts_dir(model_path: Path) -> Path:
    """Directory holding the parts for ``model_path``."""
    return Path(model_path).parent / PARTS_DIR_NAME


def manifest_path(model_path: Path) -> Path:
    """Manifest describing how ``model_path`` was split."""
    model_path = Path(model_path)
    return parts_dir(model_path) / f"{model_path.name}.manifest.json"


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
    model_path: Path | str,
    chunk_mb: int = DEFAULT_CHUNK_MB,
    compress: int = DEFAULT_COMPRESS,
    verbose: bool = True,
) -> Path:
    """Compress ``model_path`` and slice it into committable parts.

    ``compress=0`` skips the compression step and splits the file as it stands,
    which is the right choice for artifacts that are already compact.

    The original file is left untouched: the parts rebuild to a compressed copy
    that unpickles to exactly the same object. Returns the manifest path.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No such model file: {model_path}")

    destination = parts_dir(model_path)
    destination.mkdir(parents=True, exist_ok=True)

    # Rewriting through joblib is what shrinks the file; the staged copy is also
    # what gets checksummed, so the manifest always describes the rebuilt bytes.
    staged = destination / f"{model_path.name}.staged"
    if compress:
        if verbose:
            print(f"Compressing {model_path.name} ({_megabytes(model_path):.1f} MB)…")
        joblib.dump(joblib.load(model_path), staged, compress=compress)
        # Read it straight back: a part set that cannot be unpickled is worse
        # than no part set at all, and this is the only cheap moment to notice.
        joblib.load(staged)
        if verbose:
            print(f"  compressed to {_megabytes(staged):.1f} MB")
    else:
        staged.write_bytes(model_path.read_bytes())

    checksum = _sha256(staged)
    total_size = staged.stat().st_size

    for stale in sorted(destination.glob(f"{model_path.name}.[0-9][0-9][0-9]")):
        stale.unlink()

    chunk_bytes = chunk_mb * MB
    parts: list[dict] = []
    with open(staged, "rb") as source:
        while True:
            block = source.read(chunk_bytes)
            if not block:
                break
            part = destination / f"{model_path.name}.{len(parts) + 1:03d}"
            part.write_bytes(block)
            parts.append({"name": part.name, "bytes": len(block)})
            if verbose:
                print(f"  wrote {part.name} ({len(block) / MB:.1f} MB)")

    staged.unlink()

    manifest = {
        "file": model_path.name,
        "bytes": total_size,
        "sha256": checksum,
        "compress": compress,
        "chunk_mb": chunk_mb,
        "parts": parts,
    }
    target = manifest_path(model_path)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if verbose:
        print(f"Split {model_path.name} into {len(parts)} parts -> {destination}")
    return target


# ---------------------------------------------------------------------------
# Joining
# ---------------------------------------------------------------------------
def join_model(model_path: Path | str, verbose: bool = True) -> Path:
    """Rebuild ``model_path`` from its parts and verify the checksum."""
    model_path = Path(model_path)
    manifest_file = manifest_path(model_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"No manifest for {model_path.name} at {manifest_file}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    source_dir = parts_dir(model_path)

    absent = [
        part["name"] for part in manifest["parts"] if not (source_dir / part["name"]).exists()
    ]
    if absent:
        raise FileNotFoundError(
            f"Missing parts for {model_path.name}: {', '.join(absent)}. "
            "Pull the full repository or re-run the split."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Assembled next to the target and renamed at the end, so an interrupted
    # join never leaves a half-written .pkl that looks loadable.
    staging = model_path.with_suffix(model_path.suffix + ".joining")
    with open(staging, "wb") as target:
        for part in manifest["parts"]:
            with open(source_dir / part["name"], "rb") as source:
                for block in iter(lambda: source.read(READ_BLOCK), b""):
                    target.write(block)

    rebuilt = _sha256(staging)
    if rebuilt != manifest["sha256"]:
        staging.unlink()
        raise ValueError(
            f"Checksum mismatch rebuilding {model_path.name}: the parts are "
            "corrupt or out of date. Re-run the split on the machine that has "
            "the original file."
        )

    os.replace(staging, model_path)
    if verbose:
        print(f"Rebuilt {model_path.name} ({_megabytes(model_path):.1f} MB) from parts")
    return model_path


def is_available(model_path: Path | str) -> bool:
    """True if the model exists on disk or can be rebuilt from its parts."""
    model_path = Path(model_path)
    return model_path.exists() or manifest_path(model_path).exists()


def ensure_available(model_path: Path | str, verbose: bool = False) -> Path:
    """Return ``model_path``, rebuilding it from parts first if it is missing."""
    model_path = Path(model_path)
    if not model_path.exists():
        join_model(model_path, verbose=verbose)
    return model_path


def oversized_models(directory: Path = MODELS_DIR, limit_mb: int = GITHUB_LIMIT_MB) -> list[Path]:
    """Model files that GitHub would reject, largest first."""
    files = [path for path in Path(directory).glob("*.pkl") if _megabytes(path) > limit_mb]
    return sorted(files, key=lambda path: path.stat().st_size, reverse=True)


def split_manifests(directory: Path = MODELS_DIR) -> list[Path]:
    """Every manifest under ``directory/parts``."""
    return sorted((Path(directory) / PARTS_DIR_NAME).glob("*.manifest.json"))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _report_status(directory: Path) -> None:
    models = sorted(Path(directory).glob("*.pkl"))
    if not models and not split_manifests(directory):
        print(f"No model files in {directory}")
        return

    print(f"{'File':<28}{'Size (MB)':>11}  Status")
    for model in models:
        size = _megabytes(model)
        if manifest_path(model).exists():
            count = len(json.loads(manifest_path(model).read_text(encoding="utf-8"))["parts"])
            status = f"split into {count} parts"
        elif size > GITHUB_LIMIT_MB:
            status = f"TOO LARGE for GitHub (>{GITHUB_LIMIT_MB} MB) — run split"
        else:
            status = "commits directly"
        print(f"{model.name:<28}{size:>11.1f}  {status}")

    for manifest_file in split_manifests(directory):
        rebuilt = Path(directory) / manifest_file.name.replace(".manifest.json", "")
        if not rebuilt.exists():
            print(f"{rebuilt.name:<28}{'—':>11}  parts present, run join to rebuild")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="compress and split model files")
    split_parser.add_argument("paths", nargs="*", type=Path, help="model files to split")
    split_parser.add_argument(
        "--all", action="store_true", help=f"split every model over {GITHUB_LIMIT_MB} MB"
    )
    split_parser.add_argument("--chunk-mb", type=int, default=DEFAULT_CHUNK_MB)
    split_parser.add_argument(
        "--compress", type=int, default=DEFAULT_COMPRESS, help="joblib level 0-9 (0 disables)"
    )

    join_parser = subparsers.add_parser("join", help="rebuild model files from parts")
    join_parser.add_argument("paths", nargs="*", type=Path, help="model files to rebuild")
    join_parser.add_argument("--all", action="store_true", help="rebuild every manifest")

    subparsers.add_parser("status", help="show which models fit on GitHub")

    args = parser.parse_args()

    if args.command == "status":
        _report_status(MODELS_DIR)
        return

    if args.command == "split":
        targets = list(args.paths)
        if args.all:
            targets = oversized_models()
            if not targets:
                print(f"Nothing over {GITHUB_LIMIT_MB} MB in {MODELS_DIR}")
                return
        if not targets:
            split_parser.error("pass one or more paths, or --all")
        for target in targets:
            split_model(target, chunk_mb=args.chunk_mb, compress=args.compress)
        return

    targets = list(args.paths)
    if args.all:
        targets = [
            MODELS_DIR / manifest.name.replace(".manifest.json", "")
            for manifest in split_manifests()
        ]
        if not targets:
            print(f"No manifests in {MODELS_DIR / PARTS_DIR_NAME}")
            return
    if not targets:
        join_parser.error("pass one or more paths, or --all")
    for target in targets:
        join_model(target)


if __name__ == "__main__":
    main()
