"""Interactive property preview for the predictor page.

The scene is generated entirely in code from the values the user typed into the
predictor — floor count, lit windows, unit footprint and pets all derive from
`square_feet` / `bedrooms` / `bathrooms` / `pets_allowed`. There are no model
files to ship or license; every texture is drawn onto a canvas at runtime.

Runtime: three.js r0.185.1, vendored as ES modules under `app/static/three/`
and served by Streamlit's static file server (`enableStaticServing` in
`.streamlit/config.toml`). Loading it by URL rather than inlining it keeps the
bundle out of the page on every rerun and lets the browser cache it.

This is a Custom Component v2: it renders into a shadow root in the app
document, not an iframe. Streamlit turns the `js` string below into a blob-URL
module, so every import inside it must be an absolute path.
"""

from __future__ import annotations

import streamlit as st


_HTML = """
<div class="preview-shell">
  <canvas class="property-canvas" aria-label="Interactive property preview"></canvas>
  <div class="location"></div>
  <div class="rent"><span>Predicted rent</span><strong></strong></div>
  <div class="view-toggle">
    <button type="button" data-view="building" class="active">Building</button>
    <button type="button" data-view="rooms">Room view</button>
  </div>
  <div class="summary"></div>
  <div class="room-labels"></div>
  <div class="hint">Drag to orbit · Scroll to zoom</div>
</div>
"""

_CSS = """
:host { display: block; }
.preview-shell { position: relative; height: 520px; overflow: hidden; border-radius: 14px; background: linear-gradient(#4a90d9, #bfe0f5 72%, #e7f4fb); }
.property-canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
.property-canvas:active { cursor: grabbing; }
.location, .rent, .summary, .hint, .view-toggle, .room-labels { position: absolute; z-index: 2; border: 1px solid rgba(255,255,255,.20); border-radius: 10px; color: #f4f7ff; background: rgba(24,31,45,.58); box-shadow: 0 8px 24px rgba(0,0,0,.16); backdrop-filter: blur(12px); font: 600 11px/1.25 system-ui, sans-serif; }
.location { left: 16px; top: 16px; padding: 8px 10px; }
.rent { right: 16px; top: 16px; padding: 8px 10px; text-align: right; }
.rent span { display: block; color: #c3cad9; font-size: 10px; letter-spacing: .06em; text-transform: uppercase; }
.rent strong { display: block; margin-top: 3px; font-size: 17px; }
.view-toggle { top: 70px; left: 16px; display: flex; padding: 3px; }
.view-toggle button { border: 0; border-radius: 7px; padding: 6px 8px; color: #d6dceb; background: transparent; font: 650 11px/1 system-ui, sans-serif; cursor: pointer; }
.view-toggle button.active { color: white; background: rgba(255,255,255,.18); }
.summary { left: 16px; bottom: 16px; padding: 7px 9px; }
.hint { right: 16px; bottom: 16px; padding: 7px 9px; color: #d1d8e8; }
.room-labels { display: none; top: 112px; left: 16px; max-width: 230px; padding: 8px 10px; line-height: 1.65; }
.room-labels strong { display: block; font-size: 12px; }
.room-labels span { display: inline-block; margin: 2px 4px 2px 0; padding: 2px 5px; border-radius: 5px; }
.room-labels .bedroom { color: #ffd4ad; background: rgba(208, 123, 67, .22); }
.room-labels .bathroom { color: #c7efff; background: rgba(112, 195, 232, .20); }
.switching { animation: switch .24s ease-out both; }
@keyframes switch { from { opacity: .22; transform: scale(.97); } to { opacity: 1; transform: scale(1); } }
@media (prefers-reduced-motion: reduce) { .hint { display: none; } .switching { animation: none; } }
"""

_JS = r"""
// Streamlit compiles this file into a blob: URL module, and relative — including
// root-relative — specifiers cannot be resolved against a blob base. Build fully
// absolute URLs off the document instead. Resolving against `document.baseURI`
// rather than the origin also keeps this correct under `server.baseUrlPath`.
const ASSET_BASE = new URL('app/static/three/', document.baseURI).href;
const THREE_URL = ASSET_BASE + 'three.module.min.js';
const ORBIT_URL = ASSET_BASE + 'addons/OrbitControls.js';

// One viewer per mounted component. Streamlit re-runs this default export on
// every data change but only calls the returned cleanup on real unmount, so
// the WebGL context must be built once and then updated in place — rebuilding
// it per rerun would leak contexts until the browser starts dropping them.
const VIEWERS = new WeakMap();

export default function (component) {
  const { data, parentElement } = component;
  const shell = parentElement.querySelector('.preview-shell');
  if (!shell) return undefined;
  shell.style.height = (data && data.height ? data.height : 520) + 'px';

  let viewer = VIEWERS.get(parentElement);
  if (!viewer) {
    viewer = createViewer(shell);
    VIEWERS.set(parentElement, viewer);
  }
  viewer.update(data || {});

  return () => {
    VIEWERS.delete(parentElement);
    viewer.dispose();
  };
}

/* ---------------------------------------------------------------- helpers */

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * Empty a group, freeing its GPU buffers.
 *
 * `disposeMaterials` must stay false for groups drawn with the viewer's shared
 * materials: those outlive any single rebuild, and disposing them on a rerun
 * frees the shader programs still in use and every later frame renders black.
 * Only pass true for groups that own their materials (the pets).
 */
function clearGroup(root, disposeMaterials) {
  root.traverse((node) => {
    if (node.geometry) node.geometry.dispose();
    if (!disposeMaterials || !node.material) return;
    (Array.isArray(node.material) ? node.material : [node.material]).forEach((entry) => entry.dispose());
  });
  root.clear();
}

/** Sobel a greyscale height canvas into a tangent-space normal map. */
function normalFromHeight(heightCanvas, strength) {
  const size = heightCanvas.width;
  const source = heightCanvas.getContext('2d').getImageData(0, 0, size, size).data;
  const target = document.createElement('canvas');
  target.width = target.height = size;
  const out = target.getContext('2d').createImageData(size, size);
  const at = (x, y) => source[((y & (size - 1)) * size + (x & (size - 1))) * 4] / 255;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const dx = (at(x + 1, y) - at(x - 1, y)) * strength;
      const dy = (at(x, y + 1) - at(x, y - 1)) * strength;
      const length = Math.hypot(dx, dy, 1);
      const index = (y * size + x) * 4;
      out.data[index] = ((-dx / length) * 0.5 + 0.5) * 255;
      out.data[index + 1] = ((-dy / length) * 0.5 + 0.5) * 255;
      out.data[index + 2] = ((1 / length) * 0.5 + 0.5) * 255;
      out.data[index + 3] = 255;
    }
  }
  target.getContext('2d').putImageData(out, 0, 0);
  return target;
}

/** Brick facade: a colour map plus a normal map derived from the same layout. */
function makeBrickTextures(THREE) {
  const size = 512;
  const color = document.createElement('canvas');
  const height = document.createElement('canvas');
  color.width = color.height = height.width = height.height = size;
  const cc = color.getContext('2d');
  const hc = height.getContext('2d');
  cc.fillStyle = '#9d8c74';
  hc.fillStyle = '#3c3c3c';
  cc.fillRect(0, 0, size, size);
  hc.fillRect(0, 0, size, size);
  const rows = 16;
  const brickH = size / rows;
  const brickW = size / 8;
  for (let row = 0; row < rows; row += 1) {
    const offset = row % 2 ? brickW / 2 : 0;
    for (let brick = -1; brick < 9; brick += 1) {
      const x = brick * brickW + offset + 2;
      const y = row * brickH + 2;
      const w = brickW - 4;
      const h = brickH - 4;
      const tint = 0.82 + Math.random() * 0.36;
      cc.fillStyle = 'rgb(' + Math.round(196 * tint) + ',' + Math.round(158 * tint) + ',' + Math.round(114 * tint) + ')';
      cc.fillRect(x, y, w, h);
      hc.fillStyle = 'rgb(' + Math.round(190 + Math.random() * 50) + ',0,0)';
      hc.fillRect(x, y, w, h);
      // Speckle so each brick has grain rather than reading as flat paint.
      for (let fleck = 0; fleck < 26; fleck += 1) {
        cc.fillStyle = Math.random() < 0.5 ? 'rgba(255,255,255,.06)' : 'rgba(0,0,0,.07)';
        cc.fillRect(x + Math.random() * w, y + Math.random() * h, 2, 2);
      }
    }
  }
  const map = new THREE.CanvasTexture(color);
  const normalMap = new THREE.CanvasTexture(normalFromHeight(height, 2.6));
  [map, normalMap].forEach((texture) => {
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  });
  map.colorSpace = THREE.SRGBColorSpace;
  return { map, normalMap };
}

/** Fine noise used as a normal map so trim and fur are not mirror-flat. */
function makeNoiseNormal(THREE, size, strength, repeat) {
  const height = document.createElement('canvas');
  height.width = height.height = size;
  const ctx = height.getContext('2d');
  const image = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i += 1) {
    const value = 110 + Math.random() * 145;
    image.data[i * 4] = image.data[i * 4 + 1] = image.data[i * 4 + 2] = value;
    image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  const normalMap = new THREE.CanvasTexture(normalFromHeight(height, strength));
  normalMap.wrapS = normalMap.wrapT = THREE.RepeatWrapping;
  normalMap.repeat.set(repeat, repeat);
  return normalMap;
}

/** Turf colour under the blades so gaps read as ground, not void. */
function makeGroundTexture(THREE) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 256;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#41702f';
  ctx.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 2600; i += 1) {
    const x = Math.random() * 256;
    const y = Math.random() * 256;
    ctx.strokeStyle = Math.random() < 0.5 ? 'rgba(46,94,34,.55)' : 'rgba(104,158,74,.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + (Math.random() - 0.5) * 4, y - 3 - Math.random() * 4);
    ctx.stroke();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(9, 9);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/**
 * Sweep a circle along a curve with a varying radius. This is what stops the
 * animals reading as a stack of intersecting balls: the body, neck and tail
 * become one continuous surface that is fat at the ribcage and narrow at the
 * hips and throat.
 */
function tubeAlongCurve(THREE, curve, radiusAt, tubularSegments, radialSegments) {
  const frames = curve.computeFrenetFrames(tubularSegments, false);
  const positions = [];
  const uvs = [];
  const indices = [];
  const point = new THREE.Vector3();
  for (let i = 0; i <= tubularSegments; i += 1) {
    const u = i / tubularSegments;
    curve.getPointAt(u, point);
    const normal = frames.normals[i];
    const binormal = frames.binormals[i];
    const radius = radiusAt(u);
    for (let j = 0; j <= radialSegments; j += 1) {
      const theta = (j / radialSegments) * Math.PI * 2;
      const sin = Math.sin(theta);
      const cos = -Math.cos(theta);
      positions.push(
        point.x + radius * (normal.x * cos + binormal.x * sin),
        point.y + radius * (normal.y * cos + binormal.y * sin),
        point.z + radius * (normal.z * cos + binormal.z * sin),
      );
      uvs.push(u, j / radialSegments);
    }
  }
  for (let i = 1; i <= tubularSegments; i += 1) {
    for (let j = 1; j <= radialSegments; j += 1) {
      const a = (radialSegments + 1) * (i - 1) + (j - 1);
      const b = (radialSegments + 1) * i + (j - 1);
      const c = (radialSegments + 1) * i + j;
      const d = (radialSegments + 1) * (i - 1) + j;
      indices.push(a, b, d, b, c, d);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setIndex(indices);
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.computeVertexNormals();
  return geometry;
}

/* -------------------------------------------------------------- the viewer */

function createViewer(shell) {
  const canvas = shell.querySelector('.property-canvas');
  const ui = {
    location: shell.querySelector('.location'),
    rent: shell.querySelector('.rent strong'),
    summary: shell.querySelector('.summary'),
    labels: shell.querySelector('.room-labels'),
    hint: shell.querySelector('.hint'),
    buttons: Array.from(shell.querySelectorAll('.view-toggle button')),
  };

  const state = {
    data: null,
    view: 'building',
    // zoom / targetRotation / dragging / lastX drive the 2D fallback only; the
    // 3D path orbits a real camera via OrbitControls instead.
    zoom: 1,
    targetRotation: 0.3,
    dragging: false,
    lastX: 0,
    visible: true,
    disposed: false,
  };

  const setView = (next) => {
    state.view = next;
    const rooms = next === 'rooms';
    ui.labels.style.display = rooms ? 'block' : 'none';
    ui.hint.textContent = rooms ? 'Room layout'
      : (state.flat ? 'Drag to explore · Scroll to zoom' : 'Drag to orbit · Scroll to zoom');
    ui.buttons.forEach((button) => button.classList.toggle('active', button.dataset.view === next));
    canvas.classList.remove('switching');
    void canvas.offsetWidth;
    canvas.classList.add('switching');
    if (state.onViewChange) state.onViewChange(next);
  };

  ui.buttons.forEach((button) => button.addEventListener('click', () => setView(button.dataset.view)));
  // Pointer handling belongs to whichever renderer wins: the 3D path hands the
  // canvas to OrbitControls, the 2D fallback installs its own drag/zoom.

  // Pause the render loop while the component is off screen or the tab is
  // hidden — with instanced grass and a shadow pass this is no longer cheap.
  const observer = new IntersectionObserver((entries) => {
    state.visible = entries.some((entry) => entry.isIntersecting);
    if (state.onVisibility) state.onVisibility();
  }, { threshold: 0.01 });
  observer.observe(canvas);
  const onVisibilityChange = () => { if (state.onVisibility) state.onVisibility(); };
  document.addEventListener('visibilitychange', onVisibilityChange);

  const applyOverlay = (data) => {
    ui.location.textContent = data.location || '';
    ui.rent.textContent = data.rent || '';
    const baths = Number(data.bathrooms) || 0;
    const bathText = baths % 1 ? baths.toFixed(1) : String(baths);
    ui.summary.textContent = (data.bedrooms || 0) + ' bed   ·   ' + bathText
      + ' bath   ·   ' + (Number(data.squareFeet) || 0).toLocaleString() + ' sq ft';
    const bathCount = Math.ceil(baths);
    const rooms = [];
    for (let i = 0; i < (data.bedrooms || 0); i += 1) {
      rooms.push('<span class="bedroom">Bedroom ' + (i + 1) + '</span>');
    }
    for (let i = 0; i < bathCount; i += 1) {
      const half = baths % 1 && i === bathCount - 1;
      rooms.push('<span class="bathroom">' + (half ? 'Half bath' : 'Bathroom ' + (i + 1)) + '</span>');
    }
    ui.labels.innerHTML = '<strong>Room layout</strong>' + rooms.join('');
  };

  let scene3d = null;
  let fallback2d = null;

  /**
   * Fill in anything missing before it reaches the geometry.
   *
   * Streamlit can invoke the component before its data has arrived. Letting an
   * undefined `squareFeet` through turns every dimension into NaN, which
   * silently empties the scene and leaves nothing but sky.
   */
  const normalise = (data) => {
    const source = data && typeof data === 'object' ? data : {};
    const number = (value, fallback) => (Number.isFinite(Number(value)) ? Number(value) : fallback);
    return {
      location: source.location || '',
      rent: source.rent || '',
      bedrooms: Math.max(0, Math.round(number(source.bedrooms, 1))),
      bathrooms: Math.max(0.5, number(source.bathrooms, 1)),
      squareFeet: Math.max(120, number(source.squareFeet, 850)),
      hasCats: !!source.hasCats,
      hasDogs: !!source.hasDogs,
      height: number(source.height, 520),
    };
  };

  const update = (raw) => {
    const data = normalise(raw);
    state.data = data;
    applyOverlay(data);
    if (scene3d) scene3d.apply(data);
    if (fallback2d) fallback2d.apply(data);
  };

  const dispose = () => {
    state.disposed = true;
    observer.disconnect();
    document.removeEventListener('visibilitychange', onVisibilityChange);
    if (scene3d) scene3d.dispose();
    if (fallback2d) fallback2d.dispose();
  };

  Promise.all([import(THREE_URL), import(ORBIT_URL)])
    .then(([THREE, orbitModule]) => {
      if (state.disposed) return;
      scene3d = startThree(THREE, orbitModule.OrbitControls, canvas, state, setView);
      if (state.data) scene3d.apply(state.data);
    })
    .catch((error) => {
      console.warn('three.js preview unavailable; using the 2D canvas fallback\n', (error && error.stack) || String(error));
      if (state.disposed) return;
      fallback2d = startCanvasFallback(canvas, state, setView);
      if (state.data) fallback2d.apply(state.data);
    });

  return { update, dispose };
}

/* ------------------------------------------------------------ 3D renderer */

function startThree(THREE, OrbitControls, canvas, state, setView) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  // Exposure is set for the sky (Preetham is bright); the sun and hemisphere
  // below are scaled up to compensate so the building is not left dim.
  renderer.toneMappingExposure = 0.72;
  renderer.shadowMap.enabled = true;
  // PCFSoftShadowMap is deprecated in r185; PCF plus a blur radius gives the
  // same soft contact shadow without the warning.
  renderer.shadowMap.type = THREE.PCFShadowMap;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 300);

  // A gradient dome rather than the Preetham Sky addon. Preetham's radiance
  // near the horizon is so high that any exposure which keeps the building
  // properly lit clips the sky to white — and the horizon band is exactly what
  // this camera looks at. `toneMapped: false` takes the dome out of the
  // exposure calculation entirely, so sky and subject can be tuned separately.
  const skyCanvas = document.createElement('canvas');
  skyCanvas.width = 2;
  skyCanvas.height = 512;
  const skyCtx = skyCanvas.getContext('2d');
  const skyGradient = skyCtx.createLinearGradient(0, 0, 0, 512);
  skyGradient.addColorStop(0, '#3a76c4');
  skyGradient.addColorStop(0.3, '#79aede');
  skyGradient.addColorStop(0.47, '#c7dcee');
  skyGradient.addColorStop(0.5, '#dfe8ee');
  skyGradient.addColorStop(1, '#c3ccc2');
  skyCtx.fillStyle = skyGradient;
  skyCtx.fillRect(0, 0, 2, 512);
  const skyTexture = new THREE.CanvasTexture(skyCanvas);
  skyTexture.colorSpace = THREE.SRGBColorSpace;
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(60, 32, 24),
    new THREE.MeshBasicMaterial({ map: skyTexture, side: THREE.BackSide, fog: false, depthWrite: false, toneMapped: false }),
  );
  const sunDirection = new THREE.Vector3();
  sunDirection.setFromSphericalCoords(1, THREE.MathUtils.degToRad(52), THREE.MathUtils.degToRad(140));
  scene.add(sky);
  // Fades the lawn disc's edge into the horizon so the scene does not read
  // as an island floating in the sky.
  scene.fog = new THREE.Fog(0xdfe8ee, 15, 34);

  // Reflections come from a small equirectangular gradient — sky above, grass
  // below. It is deliberately a bounded LDR image: pre-filtering an unbounded
  // HDR sky produces a cube map that zeroes out every lit surface, and the
  // whole scene then renders black however far the sun is turned up.
  const envCanvas = document.createElement('canvas');
  envCanvas.width = 64;
  envCanvas.height = 32;
  const envCtx = envCanvas.getContext('2d');
  const envGradient = envCtx.createLinearGradient(0, 0, 0, 32);
  envGradient.addColorStop(0, '#7fb2e5');
  envGradient.addColorStop(0.46, '#cfe4f5');
  envGradient.addColorStop(0.54, '#9fb489');
  envGradient.addColorStop(1, '#4e6b3c');
  envCtx.fillStyle = envGradient;
  envCtx.fillRect(0, 0, 64, 32);
  const envTexture = new THREE.CanvasTexture(envCanvas);
  envTexture.mapping = THREE.EquirectangularReflectionMapping;
  envTexture.colorSpace = THREE.SRGBColorSpace;
  const pmrem = new THREE.PMREMGenerator(renderer);
  const environment = pmrem.fromEquirectangular(envTexture);
  scene.environment = environment.texture;
  envTexture.dispose();
  pmrem.dispose();

  const sun = new THREE.DirectionalLight(0xfff2dd, 4.4);
  sun.position.copy(sunDirection).multiplyScalar(24);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -9;
  sun.shadow.camera.right = 9;
  sun.shadow.camera.top = 11;
  sun.shadow.camera.bottom = -6;
  sun.shadow.camera.near = 2;
  sun.shadow.camera.far = 60;
  sun.shadow.normalBias = 0.02;
  sun.shadow.bias = -0.0004;
  sun.shadow.radius = 2.5;
  scene.add(sun);
  // The environment map now carries most of the fill, so the hemisphere light
  // only needs to lift the shadowed sides rather than flatten everything.
  scene.add(new THREE.HemisphereLight(0xbfe3ff, 0x557a45, 0.85));

  const brick = makeBrickTextures(THREE);
  const trimNormal = makeNoiseNormal(THREE, 128, 0.7, 3);
  const furNormal = makeNoiseNormal(THREE, 128, 1.5, 6);

  const materials = {
    facade: new THREE.MeshStandardMaterial({ map: brick.map, normalMap: brick.normalMap, normalScale: new THREE.Vector2(0.85, 0.85), roughness: 0.92, metalness: 0.02 }),
    trim: new THREE.MeshStandardMaterial({ color: 0xe8e2d6, normalMap: trimNormal, normalScale: new THREE.Vector2(0.25, 0.25), roughness: 0.6, metalness: 0.03 }),
    // Strong envMapIntensity is what stops the panes reading as black holes:
    // recessed glass sees very little sky, so it needs help catching it.
    glass: new THREE.MeshPhysicalMaterial({ color: 0x3c4f5c, roughness: 0.05, metalness: 0.55, envMapIntensity: 2.6, clearcoat: 1, clearcoatRoughness: 0.04 }),
    litGlass: new THREE.MeshPhysicalMaterial({ color: 0xf6e6c6, roughness: 0.28, metalness: 0.1, envMapIntensity: 1.2, emissive: 0xffc879, emissiveIntensity: 0.6, clearcoat: 1 }),
    metal: new THREE.MeshStandardMaterial({ color: 0x8d9299, roughness: 0.38, metalness: 0.85 }),
    roof: new THREE.MeshStandardMaterial({ color: 0x4a4844, roughness: 0.88, metalness: 0.05 }),
  };

  // Brick scale comes from per-geometry UVs (see brickBox), so the shared
  // texture stays at 1:1 here.
  materials.facade.map.repeat.set(1, 1);
  materials.facade.normalMap.repeat.set(1, 1);

  // The building stands still and the camera orbits it, so the model reads as a
  // fixed object you walk around rather than a turntable prop.
  const exterior = new THREE.Group();
  scene.add(exterior);
  const interior = new THREE.Group();
  interior.visible = false;
  scene.add(interior);
  const petsGroup = new THREE.Group();
  scene.add(petsGroup);

  /* ground + grass ------------------------------------------------------- */

  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(6.2, 64),
    new THREE.MeshStandardMaterial({ map: makeGroundTexture(THREE), roughness: 1, metalness: 0 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.23;
  ground.receiveShadow = true;
  scene.add(ground);

  const BLADE_H = 0.13;
  const bladeGeometry = new THREE.BufferGeometry();
  const halfWidth = 0.017;
  bladeGeometry.setAttribute('position', new THREE.Float32BufferAttribute([
    -halfWidth, 0, 0,
    halfWidth, 0, 0,
    -halfWidth * 0.6, BLADE_H * 0.55, 0,
    halfWidth * 0.6, BLADE_H * 0.55, 0,
    0, BLADE_H, 0,
  ], 3));
  bladeGeometry.setAttribute('normal', new THREE.Float32BufferAttribute([
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
  ], 3));
  bladeGeometry.setIndex([0, 1, 2, 1, 3, 2, 2, 3, 4]);

  const grassMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.85, metalness: 0, side: THREE.DoubleSide });
  grassMaterial.customProgramCacheKey = () => 'apartment-grass';
  grassMaterial.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = { value: 0 };
    shader.uniforms.uWind = { value: 0.05 };
    shader.vertexShader = 'uniform float uTime;\nuniform float uWind;\n' + shader.vertexShader.replace(
      '#include <begin_vertex>',
      [
        '#include <begin_vertex>',
        // Bend from the root: the tip travels, the base stays planted.
        'float bladeT = clamp(transformed.y / ' + BLADE_H.toFixed(3) + ', 0.0, 1.0);',
        'float bend = bladeT * bladeT * uWind;',
        'vec3 bladeRoot = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);',
        'float phase = uTime * 1.7 + bladeRoot.x * 1.4 + bladeRoot.z * 1.1;',
        'transformed.x += sin(phase) * bend;',
        'transformed.z += cos(phase * 0.7) * bend * 0.55;',
      ].join('\n'),
    );
    grassMaterial.userData.shader = shader;
  };

  const lowPower = (navigator.hardwareConcurrency || 8) <= 4;
  const bladeCount = lowPower ? 11000 : 26000;
  const grass = new THREE.InstancedMesh(bladeGeometry, grassMaterial, bladeCount);
  grass.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  grass.castShadow = false;
  grass.receiveShadow = true;
  grass.frustumCulled = false;
  grass.position.y = -0.23;
  scene.add(grass);

  const dummy = new THREE.Object3D();
  const bladeColor = new THREE.Color();
  let grassScale = null;
  const placeGrass = (footprintScale) => {
    if (grassScale === footprintScale) return;
    grassScale = footprintScale;
    // Keep blades out of the building's footprint so none poke through the
    // plinth. The building is axis-aligned (the camera orbits, not the model),
    // so a plain rectangle test is enough.
    const halfW = 3.55 * footprintScale * 0.5 + 0.05;
    const halfD = 2.8 * footprintScale * 0.5 + 0.05;
    for (let i = 0; i < bladeCount; i += 1) {
      let x = 0;
      let z = 0;
      for (let attempt = 0; attempt < 12; attempt += 1) {
        const radius = 0.9 + Math.sqrt(Math.random()) * 5.3;
        const angle = Math.random() * Math.PI * 2;
        x = Math.cos(angle) * radius;
        z = Math.sin(angle) * radius;
        if (Math.abs(x) > halfW || Math.abs(z) > halfD) break;
      }
      dummy.position.set(x, 0, z);
      dummy.rotation.set((Math.random() - 0.5) * 0.35, Math.random() * Math.PI * 2, (Math.random() - 0.5) * 0.35);
      const height = 0.65 + Math.random() * 0.75;
      dummy.scale.set(0.8 + Math.random() * 0.5, height, 1);
      dummy.updateMatrix();
      grass.setMatrixAt(i, dummy.matrix);
      const shade = 0.62 + Math.random() * 0.5;
      bladeColor.setRGB(0.30 * shade, 0.62 * shade, 0.24 * shade);
      grass.setColorAt(i, bladeColor);
    }
    grass.instanceMatrix.needsUpdate = true;
    if (grass.instanceColor) grass.instanceColor.needsUpdate = true;
  };

  /* shrubs and trees ----------------------------------------------------- */

  const foliageMaterial = new THREE.MeshStandardMaterial({ color: 0x497a35, roughness: 0.95, metalness: 0, flatShading: true });
  const barkMaterial = new THREE.MeshStandardMaterial({ color: 0x5b4632, roughness: 0.95, metalness: 0 });
  const plants = new THREE.Group();
  plants.position.y = -0.23;
  scene.add(plants);
  const roughen = (geometry, amount) => {
    const position = geometry.attributes.position;
    for (let i = 0; i < position.count; i += 1) {
      position.setXYZ(
        i,
        position.getX(i) * (1 + (Math.random() - 0.5) * amount),
        position.getY(i) * (1 + (Math.random() - 0.5) * amount),
        position.getZ(i) * (1 + (Math.random() - 0.5) * amount),
      );
    }
    geometry.computeVertexNormals();
    return geometry;
  };
  for (let i = 0; i < 9; i += 1) {
    const angle = (i / 9) * Math.PI * 2 + 0.4;
    const radius = 4.4 + Math.random() * 1.2;
    const shrub = new THREE.Mesh(roughen(new THREE.IcosahedronGeometry(0.26 + Math.random() * 0.16, 1), 0.42), foliageMaterial);
    shrub.position.set(Math.cos(angle) * radius, 0.2, Math.sin(angle) * radius);
    shrub.castShadow = true;
    shrub.receiveShadow = true;
    plants.add(shrub);
  }
  for (let i = 0; i < 3; i += 1) {
    const angle = (i / 3) * Math.PI * 2 + 1.1;
    const radius = 4.6;
    const tree = new THREE.Group();
    tree.position.set(Math.cos(angle) * radius, 0, Math.sin(angle) * radius);
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.075, 0.95, 7), barkMaterial);
    trunk.position.y = 0.47;
    trunk.castShadow = true;
    tree.add(trunk);
    for (let blob = 0; blob < 3; blob += 1) {
      const crown = new THREE.Mesh(roughen(new THREE.IcosahedronGeometry(0.42 - blob * 0.06, 1), 0.36), foliageMaterial);
      crown.position.set((Math.random() - 0.5) * 0.24, 1.0 + blob * 0.25, (Math.random() - 0.5) * 0.24);
      crown.castShadow = true;
      crown.receiveShadow = true;
      tree.add(crown);
    }
    plants.add(tree);
  }

  /* building ------------------------------------------------------------- */

  const buildExterior = (data) => {
    clearGroup(exterior, false);
    const scale = Math.min(1.38, Math.max(0.68, data.squareFeet / 850));
    const floors = Math.min(9, Math.max(4, Math.round(data.squareFeet / 185)));
    const width = 3.1 * scale;
    const depth = 2.35 * scale;
    const floorH = 0.74;
    const front = depth / 2;
    // Everything is measured off two heights: the lawn, and the top of the
    // plinth the tower stands on.
    const GROUND = -0.23;
    const plinthH = 0.42;
    const baseTop = GROUND + plinthH;
    const plinthDepth = 2.8 * scale;

    const add = (mesh, cast = true, receive = true) => {
      mesh.castShadow = cast;
      mesh.receiveShadow = receive;
      exterior.add(mesh);
      return mesh;
    };

    // Brick courses must be the same physical size on a wide spandrel and a
    // narrow pier, so UVs are scaled by each box's world dimensions instead of
    // relying on a single texture repeat.
    const brickBox = (w, h, d) => {
      const geometry = new THREE.BoxGeometry(w, h, d);
      const uv = geometry.attributes.uv;
      const spans = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]];
      const density = 2.1;
      spans.forEach(([su, sv], face) => {
        for (let corner = 0; corner < 4; corner += 1) {
          const index = face * 4 + corner;
          uv.setXY(index, uv.getX(index) * su * density, uv.getY(index) * sv * density);
        }
      });
      uv.needsUpdate = true;
      return geometry;
    };

    const plinth = add(new THREE.Mesh(new THREE.BoxGeometry(3.55 * scale, plinthH, plinthDepth), materials.trim));
    plinth.position.y = GROUND + plinthH / 2;

    const wallT = 0.09;
    const winW = 0.56 * scale;
    const winH = 0.38;
    const winDy = -0.04;

    // All four elevations are perforated, so the model holds up from any orbit
    // angle rather than showing a blank slab from behind. Local +z is outward
    // for each; a solid core behind them hides the interior.
    const elevations = [
      { rotY: 0, span: width, offset: depth / 2, count: 3 },
      { rotY: Math.PI, span: width, offset: depth / 2, count: 3 },
      { rotY: Math.PI / 2, span: depth, offset: width / 2, count: 2 },
      { rotY: -Math.PI / 2, span: depth, offset: width / 2, count: 2 },
    ];
    // Window furniture is identical everywhere, so it is collected here and
    // drawn as a handful of InstancedMeshes instead of ~40 meshes per floor.
    const litPanes = [];
    const darkPanes = [];
    const casings = [];
    const mullions = [];

    const toWorld = (lx, lz, rotY) => ({
      x: lx * Math.cos(rotY) + lz * Math.sin(rotY),
      z: -lx * Math.sin(rotY) + lz * Math.cos(rotY),
    });

    for (let floor = 0; floor < floors; floor += 1) {
      const centreY = baseTop + floor * floorH + floorH / 2;

      const core = add(new THREE.Mesh(brickBox(width - 2 * wallT, floorH, depth - 2 * wallT), materials.facade));
      core.position.y = centreY;

      const winTop = centreY + winDy + winH / 2;
      const winBottom = centreY + winDy - winH / 2;

      elevations.forEach((elevation) => {
        const { rotY, span, offset, count } = elevation;
        const wallLz = offset - wallT / 2;
        const addPiece = (w, h, lx, cy) => {
          if (w <= 0.002 || h <= 0.002) return;
          const piece = add(new THREE.Mesh(brickBox(w, h, wallT), materials.facade));
          const world = toWorld(lx, wallLz, rotY);
          piece.position.set(world.x, cy, world.z);
          piece.rotation.y = rotY;
        };
        const headH = centreY + floorH / 2 - winTop;
        addPiece(span, headH, 0, winTop + headH / 2);
        const sillH = winBottom - (centreY - floorH / 2);
        addPiece(span, sillH, 0, centreY - floorH / 2 + sillH / 2);

        const step = span / count;
        const xs = Array.from({ length: count }, (_, i) => -span / 2 + step * (i + 0.5));
        let pierFrom = -span / 2;
        xs.forEach((lx) => {
          addPiece(lx - winW / 2 - pierFrom, winH, (pierFrom + lx - winW / 2) / 2, centreY + winDy);
          pierFrom = lx + winW / 2;
        });
        addPiece(span / 2 - pierFrom, winH, (pierFrom + span / 2) / 2, centreY + winDy);

        xs.forEach((lx, column) => {
          const wy = centreY + winDy;
          // Glass sits at the back of the opening, so the wall thickness itself
          // reads as the reveal.
          // The glass must sit clear of the core's outer face at
          // `offset - wallT`. Landing flush on it makes the two surfaces
          // coplanar, and the resulting depth fight is what flickers across the
          // windows as the camera orbits. 16mm forward puts the pane inside the
          // 90mm reveal with room to spare.
          const paneAt = toWorld(lx, offset - wallT + 0.016, rotY);
          (column < data.bedrooms ? litPanes : darkPanes).push({ x: paneAt.x, y: wy, z: paneAt.z, rotY });
          const casingAt = toWorld(lx, offset - 0.012, rotY);
          casings.push({ x: casingAt.x, y: wy, z: casingAt.z, rotY });
          const mullionAt = toWorld(lx, offset - wallT + 0.05, rotY);
          mullions.push({ x: mullionAt.x, y: wy, z: mullionAt.z, rotY });
        });
      });

      // Band course between floors breaks the flat slab silhouette.
      const band = add(new THREE.Mesh(new THREE.BoxGeometry(width + 0.05, 0.07, depth + 0.05), materials.trim));
      band.position.y = centreY + floorH / 2;

      // Balcony with a real railing rather than a solid parapet slab.
      const balconyY = centreY + floorH / 2 - 0.06;
      const balcony = add(new THREE.Mesh(new THREE.BoxGeometry(3.42 * scale, 0.06, 0.34), materials.trim));
      balcony.position.set(0, balconyY, front + 0.17);
      const postCount = 18;
      const posts = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.008, 0.008, 0.2, 5), materials.metal, postCount);
      posts.castShadow = true;
      for (let post = 0; post < postCount; post += 1) {
        dummy.position.set(-1.68 * scale + (post / (postCount - 1)) * 3.36 * scale, balconyY + 0.13, front + 0.32);
        dummy.rotation.set(0, 0, 0);
        dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        posts.setMatrixAt(post, dummy.matrix);
      }
      posts.instanceMatrix.needsUpdate = true;
      exterior.add(posts);
      const rail = add(new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 3.36 * scale, 6), materials.metal));
      rail.rotation.z = Math.PI / 2;
      rail.position.set(0, balconyY + 0.23, front + 0.32);
    }

    // Emit the window furniture collected above. Every pane, casing and
    // mullion is the same size, so one InstancedMesh per kind replaces roughly
    // forty individual meshes per floor.
    const emitInstances = (records, geometry, material, cast, receive) => {
      if (!records.length) {
        geometry.dispose();
        return;
      }
      const mesh = new THREE.InstancedMesh(geometry, material, records.length);
      records.forEach((record, index) => {
        dummy.position.set(record.x, record.y, record.z);
        dummy.rotation.set(0, record.rotY, 0);
        dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        mesh.setMatrixAt(index, dummy.matrix);
      });
      mesh.instanceMatrix.needsUpdate = true;
      mesh.castShadow = cast;
      mesh.receiveShadow = receive;
      exterior.add(mesh);
    };

    // Casing is one extruded picture frame rather than four separate bars.
    const casingShape = new THREE.Shape();
    const outerW = winW / 2 + 0.05;
    const outerH = winH / 2 + 0.055;
    casingShape.moveTo(-outerW, -outerH);
    casingShape.lineTo(outerW, -outerH);
    casingShape.lineTo(outerW, outerH);
    casingShape.lineTo(-outerW, outerH);
    casingShape.closePath();
    const casingHole = new THREE.Path();
    casingHole.moveTo(-winW / 2, -winH / 2);
    casingHole.lineTo(-winW / 2, winH / 2);
    casingHole.lineTo(winW / 2, winH / 2);
    casingHole.lineTo(winW / 2, -winH / 2);
    casingHole.closePath();
    casingShape.holes.push(casingHole);
    const casingGeometry = new THREE.ExtrudeGeometry(casingShape, { depth: 0.05, bevelEnabled: false });
    casingGeometry.translate(0, 0, -0.025);

    emitInstances(litPanes, new THREE.BoxGeometry(winW, winH, 0.02), materials.litGlass, false, false);
    emitInstances(darkPanes, new THREE.BoxGeometry(winW, winH, 0.02), materials.glass, false, false);
    emitInstances(casings, casingGeometry, materials.trim, true, true);
    emitInstances(mullions, new THREE.BoxGeometry(0.022, winH, 0.025), materials.trim, false, false);

    // Corner pilasters run the full height and catch the sun edge-on.
    const towerH = floors * floorH;
    [-1, 1].forEach((side) => {
      [-1, 1].forEach((face) => {
        const pilaster = add(new THREE.Mesh(new THREE.BoxGeometry(0.1, towerH, 0.1), materials.trim));
        pilaster.position.set(side * (width / 2 - 0.02), baseTop + towerH / 2, face * (depth / 2 - 0.02));
      });
    });

    // Entrance sits on the plinth at street level, clear of the window grid.
    const doorH = 0.3;
    const door = add(new THREE.Mesh(new THREE.BoxGeometry(0.42 * scale, doorH, 0.05), materials.glass), false, false);
    door.position.set(0, GROUND + doorH / 2 + 0.02, plinthDepth / 2 + 0.005);
    const canopy = add(new THREE.Mesh(new THREE.BoxGeometry(0.92 * scale, 0.05, 0.4), materials.trim));
    canopy.position.set(0, GROUND + 0.38, plinthDepth / 2 + 0.18);
    [-1, 1].forEach((side) => {
      const strut = add(new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.36, 6), materials.metal));
      strut.position.set(side * 0.4 * scale, GROUND + 0.19, plinthDepth / 2 + 0.34);
    });

    // Roof: deck, parapet on four sides, and a couple of plant units.
    const roofY = baseTop + towerH + 0.05;
    const roof = add(new THREE.Mesh(new THREE.BoxGeometry(width + 0.08, 0.1, depth + 0.08), materials.roof));
    roof.position.y = roofY;
    const parapet = (w, d, ox, oz) => {
      const wall = add(new THREE.Mesh(new THREE.BoxGeometry(w, 0.17, d), materials.trim));
      wall.position.set(ox, roofY + 0.13, oz);
    };
    parapet(width + 0.12, 0.07, 0, depth / 2 + 0.02);
    parapet(width + 0.12, 0.07, 0, -depth / 2 - 0.02);
    parapet(0.07, depth + 0.12, width / 2 + 0.02, 0);
    parapet(0.07, depth + 0.12, -width / 2 - 0.02, 0);
    for (let unit = 0; unit < 2; unit += 1) {
      const hvac = add(new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.18, 0.26), materials.metal));
      hvac.position.set(-0.4 + unit * 0.8, roofY + 0.14, -0.3);
    }

    placeGrass(scale);
    return { scale, floors, topY: roofY };
  };

  /* interior ------------------------------------------------------------- */

  const interiorMaterials = {
    floor: new THREE.MeshStandardMaterial({ color: 0x8a6b4f, roughness: 0.8 }),
    bedroom: new THREE.MeshStandardMaterial({ color: 0xe2a06b, roughness: 0.42, emissive: 0x6d3e20, emissiveIntensity: 0.16 }),
    bathroom: new THREE.MeshStandardMaterial({ color: 0xb9d9ea, roughness: 0.32, emissive: 0x325267, emissiveIntensity: 0.25 }),
    living: new THREE.MeshStandardMaterial({ color: 0x62738b, roughness: 0.58, emissive: 0x182333, emissiveIntensity: 0.18 }),
  };

  const buildInterior = (data) => {
    clearGroup(interior, false);
    const unitW = Math.min(5.8, Math.max(3.8, data.squareFeet / 220));
    const unitD = Math.min(4.8, Math.max(3.1, data.squareFeet / 285));
    interior.add(new THREE.Mesh(new THREE.BoxGeometry(unitW, 0.12, unitD), interiorMaterials.floor));
    const rows = Math.max(1, Math.ceil(Math.max(1, data.bedrooms) / 2));
    const bedroomWidth = (unitW * 0.62) / 2 - 0.08;
    const bedroomDepth = (unitD * 0.58) / rows - 0.08;
    for (let room = 0; room < data.bedrooms; room += 1) {
      const row = Math.floor(room / 2);
      const col = room % 2;
      const block = new THREE.Mesh(new THREE.BoxGeometry(bedroomWidth, 0.18, bedroomDepth), interiorMaterials.bedroom);
      block.position.set(
        -unitW / 2 + bedroomWidth / 2 + 0.08 + col * (bedroomWidth + 0.08),
        0.13,
        unitD / 2 - bedroomDepth / 2 - 0.08 - row * (bedroomDepth + 0.08),
      );
      interior.add(block);
    }
    const bathCount = Math.ceil(data.bathrooms);
    for (let bath = 0; bath < bathCount; bath += 1) {
      const block = new THREE.Mesh(
        new THREE.BoxGeometry(unitW * 0.29, 0.2, (unitD * 0.48) / bathCount - 0.07),
        interiorMaterials.bathroom,
      );
      block.position.set(unitW * 0.34, 0.14, unitD / 2 - (unitD * 0.24) / bathCount - 0.1 - bath * ((unitD * 0.48) / bathCount));
      interior.add(block);
    }
    const living = new THREE.Mesh(new THREE.BoxGeometry(unitW * 0.62, 0.15, unitD * 0.31), interiorMaterials.living);
    living.position.set(-unitW * 0.18, 0.11, -unitD * 0.31);
    interior.add(living);
    const entry = new THREE.Mesh(new THREE.BoxGeometry(unitW * 0.29, 0.15, unitD * 0.36), interiorMaterials.living);
    entry.position.set(unitW * 0.34, 0.11, -unitD * 0.27);
    interior.add(entry);
  };

  /* pets ----------------------------------------------------------------- */

  // Anatomy, not colour swaps: a cat is a short-faced, fine-boned animal with a
  // long carried tail, a dog is deeper in the chest with a real snout, drop
  // ears and a stubbier tail. Everything downstream reads from these numbers.
  const PET_SPECS = {
    cat: {
      bodyLen: 0.36, bodyR: 0.070, legR: 0.016, legH: 0.135, headR: 0.080,
      // Sampled rump to shoulder: haunch, tucked waist, then the ribcage.
      girth: [0.30, 0.88, 0.99, 0.90, 1.0, 0.86, 0.56],
      backArch: 0.020, neckLift: 0.085, snout: 0.44, snoutDrop: 0.30,
      earKind: 'prick', tailR: 0.017, whiskers: true, collar: false,
      skull: [0.95, 0.92, 0.96],
      coat: 0xcf9f70, belly: 0xf4e6d2, nose: 0xd97a8f, eye: 0x4e7a35,
    },
    dog: {
      bodyLen: 0.50, bodyR: 0.094, legR: 0.021, legH: 0.155, headR: 0.088,
      // Same waist, but the chest end is markedly deeper than the hips.
      girth: [0.34, 0.86, 0.96, 0.90, 1.08, 0.96, 0.62],
      backArch: 0.008, neckLift: 0.070, snout: 0.95, snoutDrop: 0.34,
      earKind: 'drop', tailR: 0.026, whiskers: false, collar: true,
      skull: [0.88, 0.90, 1.05],
      coat: 0xa97040, belly: 0xe6d2b0, nose: 0x2e231d, eye: 0x6b4423,
    },
  };

  /** Piecewise-linear radius lookup, so body shapes are tuned as a curve of numbers. */
  const profileFrom = (points, radius) => (u) => {
    const x = THREE.MathUtils.clamp(u, 0, 0.9999) * (points.length - 1);
    const i = Math.floor(x);
    const f = x - i;
    return radius * (points[i] * (1 - f) + points[Math.min(points.length - 1, i + 1)] * f);
  };

  const buildPet = (kind) => {
    const spec = PET_SPECS[kind];
    const cat = kind === 'cat';
    const { bodyLen, bodyR, headR, legH, legR } = spec;

    const coat = new THREE.MeshStandardMaterial({
      color: spec.coat, roughness: 0.93, metalness: 0,
      normalMap: furNormal, normalScale: new THREE.Vector2(0.55, 0.55),
    });
    const belly = new THREE.MeshStandardMaterial({
      color: spec.belly, roughness: 0.95, metalness: 0, normalMap: furNormal,
    });
    const scleraMat = new THREE.MeshStandardMaterial({ color: 0xf7f2ea, roughness: 0.25 });
    const irisMat = new THREE.MeshStandardMaterial({ color: spec.eye, roughness: 0.18, metalness: 0.05 });
    // Darker ears give the dog's head some contrast against its own coat.
    const earMat = new THREE.MeshStandardMaterial({
      color: cat ? spec.coat : 0x8a5730, roughness: 0.93, normalMap: furNormal,
    });
    const pupilMat = new THREE.MeshStandardMaterial({ color: 0x120d0b, roughness: 0.15 });
    const glintMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.1, emissive: 0x666666 });
    const noseMat = new THREE.MeshStandardMaterial({ color: spec.nose, roughness: 0.35 });
    const materials = [coat, belly, scleraMat, irisMat, pupilMat, glintMat, noseMat, earMat];

    const pet = new THREE.Group();
    const backY = legH + bodyR * 0.95;
    const solid = (mesh) => { mesh.castShadow = true; return mesh; };

    /* torso ------------------------------------------------------------- */

    const spine = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, backY - spec.backArch * 0.6, -bodyLen * 0.5),
      new THREE.Vector3(0, backY + spec.backArch, -bodyLen * 0.15),
      new THREE.Vector3(0, backY + spec.backArch * 0.8, bodyLen * 0.18),
      new THREE.Vector3(0, backY + spec.backArch * 0.2, bodyLen * 0.44),
    ]);
    const girth = profileFrom(spec.girth, bodyR);
    // The curve runs rump-to-shoulder, so the profile is sampled reversed.
    const torso = solid(new THREE.Mesh(tubeAlongCurve(THREE, spine, (u) => girth(u), 30, 16), coat));
    pet.add(torso);
    // Rounded rump and a lighter belly panel slung under the ribs.
    const rump = solid(new THREE.Mesh(new THREE.SphereGeometry(girth(0) * 1.06, 16, 12), coat));
    rump.position.set(0, backY - spec.backArch * 0.6, -bodyLen * 0.5);
    pet.add(rump);
    const underside = new THREE.Mesh(tubeAlongCurve(THREE, spine, (u) => girth(u) * 0.66, 24, 12), belly);
    underside.position.y = -bodyR * 0.44;
    pet.add(underside);

    /* neck and head ------------------------------------------------------ */

    const headY = backY + spec.neckLift;
    const headZ = bodyLen * 0.5 + headR * 0.5;
    const neckCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, backY + spec.backArch * 0.2, bodyLen * 0.38),
      new THREE.Vector3(0, backY + spec.neckLift * 0.55, bodyLen * 0.47),
      new THREE.Vector3(0, headY - headR * 0.3, headZ - headR * 0.55),
    ]);
    const neckR = bodyR * (cat ? 0.62 : 0.70);
    pet.add(solid(new THREE.Mesh(
      tubeAlongCurve(THREE, neckCurve, (u) => neckR * (1 - 0.16 * u), 14, 12), coat,
    )));

    if (spec.collar) {
      const collarMat = new THREE.MeshStandardMaterial({ color: 0x9c3f3f, roughness: 0.6 });
      materials.push(collarMat);
      const collar = new THREE.Mesh(new THREE.TorusGeometry(neckR * 1.02, 0.011, 8, 20), collarMat);
      collar.position.set(0, headY - headR * 0.55, headZ - headR * 0.85);
      collar.rotation.x = Math.PI / 2 - 0.5;
      pet.add(collar);
    }

    // Head parts live in their own group so the whole head can nod as one.
    const head = new THREE.Group();
    head.position.set(0, headY, headZ);
    pet.add(head);

    const skullScale = spec.skull;
    const skull = solid(new THREE.Mesh(new THREE.SphereGeometry(headR, 20, 16), coat));
    skull.scale.set(skullScale[0], skullScale[1], skullScale[2]);
    head.add(skull);

    /**
     * Point on the skull's surface in a given direction.
     *
     * The skull is a *scaled* sphere, so a feature placed at some fraction of
     * headR is not on its surface — it ends up buried inside the ellipsoid.
     * Solving for the surface along the direction is what keeps eyes and ears
     * on the outside of the head where they can be seen.
     */
    const onSkull = (dirX, dirY, dirZ, push = 0) => {
      const d = new THREE.Vector3(dirX, dirY, dirZ).normalize();
      const inv = Math.hypot(
        d.x / (headR * skullScale[0]),
        d.y / (headR * skullScale[1]),
        d.z / (headR * skullScale[2]),
      );
      return d.multiplyScalar(1 / inv + push);
    };
    if (cat) {
      // Cheek tufts give the cat its round, short face.
      [-1, 1].forEach((side) => {
        const cheek = new THREE.Mesh(new THREE.SphereGeometry(headR * 0.42, 12, 10), coat);
        cheek.scale.set(0.8, 0.75, 0.7);
        cheek.position.set(side * headR * 0.56, -headR * 0.20, headR * 0.38);
        head.add(cheek);
      });
    }

    // The snout is the clearest species tell, so it is swept rather than a
    // stuck-on sphere: long and tapered on the dog, short and blunt on the cat.
    const snoutLen = headR * spec.snout;
    const snoutCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, -headR * 0.10, headR * 0.35),
      new THREE.Vector3(0, -headR * spec.snoutDrop * 0.6, headR * 0.35 + snoutLen * 0.5),
      new THREE.Vector3(0, -headR * spec.snoutDrop, headR * 0.35 + snoutLen),
    ]);
    const snoutR = headR * (cat ? 0.44 : 0.36);
    head.add(solid(new THREE.Mesh(
      tubeAlongCurve(THREE, snoutCurve, (u) => snoutR * (1 - 0.30 * u), 12, 12), cat ? belly : coat,
    )));
    const noseTipZ = headR * 0.35 + snoutLen;
    const noseTipY = -headR * spec.snoutDrop;
    const nose = new THREE.Mesh(new THREE.SphereGeometry(headR * 0.15, 10, 8), noseMat);
    nose.scale.set(1.15, 0.85, 0.8);
    nose.position.set(0, noseTipY + headR * 0.04, noseTipZ + headR * 0.04);
    head.add(nose);

    [-1, 1].forEach((side) => {
      // Eye = white, iris, pupil and a catchlight, all seated on the skull
      // surface and set just far enough in that the eyeball bulges naturally.
      const eyeR = headR * 0.17;
      const eyeDir = new THREE.Vector3(side * (cat ? 0.62 : 0.60), cat ? 0.20 : 0.26, cat ? 0.78 : 0.76);
      const eyeAt = onSkull(eyeDir.x, eyeDir.y, eyeDir.z, -eyeR * 0.45);
      const outward = eyeAt.clone().normalize();
      const white = new THREE.Mesh(new THREE.SphereGeometry(eyeR, 12, 10), scleraMat);
      white.position.copy(eyeAt);
      head.add(white);
      const iris = new THREE.Mesh(new THREE.SphereGeometry(eyeR * 0.72, 12, 10), irisMat);
      iris.position.copy(eyeAt).addScaledVector(outward, eyeR * 0.45);
      head.add(iris);
      const pupil = new THREE.Mesh(new THREE.SphereGeometry(eyeR * 0.42, 8, 8), pupilMat);
      // A cat's pupil is a vertical slit, a dog's is round.
      pupil.scale.set(cat ? 0.4 : 1, 1, 1);
      pupil.position.copy(eyeAt).addScaledVector(outward, eyeR * 0.72);
      head.add(pupil);
      const glint = new THREE.Mesh(new THREE.SphereGeometry(eyeR * 0.26, 6, 6), glintMat);
      glint.position.copy(eyeAt)
        .addScaledVector(outward, eyeR * 0.80)
        .add(new THREE.Vector3(side * eyeR * 0.30, eyeR * 0.34, 0));
      head.add(glint);

      if (spec.earKind === 'prick') {
        // Cat: tall triangle with a pink inner ear set just inside it.
        const ear = solid(new THREE.Mesh(new THREE.ConeGeometry(headR * 0.40, headR * 0.90, 6), coat));
        ear.position.set(side * headR * 0.56, headR * 0.86, -headR * 0.06);
        ear.rotation.set(-0.12, 0, side * 0.28);
        head.add(ear);
        const inner = new THREE.Mesh(new THREE.ConeGeometry(headR * 0.22, headR * 0.52, 6), noseMat);
        inner.position.set(side * headR * 0.55, headR * 0.80, headR * 0.04);
        inner.rotation.set(-0.12, 0, side * 0.28);
        head.add(inner);
      } else {
        // Dog: a long flap hung off the side of the skull. Anchoring it on the
        // surface is the whole trick — placed by eye it sat inside the head and
        // the dog appeared to have no ears at all.
        // Anchored on the skull, then pushed clear of it before being dropped,
        // so the flap hangs beside the head instead of sinking into it.
        const anchor = onSkull(side, 0.35, -0.12).multiplyScalar(1.12);
        const ear = solid(new THREE.Mesh(new THREE.SphereGeometry(headR * 0.52, 12, 10), earMat));
        ear.scale.set(0.28, 1.15, 0.66);
        ear.position.set(anchor.x, anchor.y - headR * 0.30, anchor.z);
        ear.rotation.set(0.2, 0, side * 0.18);
        head.add(ear);
      }
    });

    if (spec.whiskers) {
      const whiskerMat = new THREE.MeshStandardMaterial({ color: 0xf2ece2, roughness: 0.5 });
      materials.push(whiskerMat);
      const whiskerGeometry = new THREE.CylinderGeometry(0.0012, 0.0008, headR * 1.5, 4);
      [-1, 1].forEach((side) => {
        [-0.16, 0, 0.16].forEach((tilt, row) => {
          const whisker = new THREE.Mesh(whiskerGeometry, whiskerMat);
          whisker.position.set(side * headR * 0.34, -headR * 0.18 + row * headR * 0.09, noseTipZ - headR * 0.12);
          whisker.rotation.set(0, 0, side * (Math.PI / 2 - 0.25) + tilt * side);
          head.add(whisker);
        });
      });
    }
    pet.userData.head = head;

    /* tail --------------------------------------------------------------- */

    const tailBase = new THREE.Vector3(0, backY - spec.backArch * 0.4, -bodyLen * 0.5);
    const tailCurve = new THREE.CatmullRomCurve3(cat ? [
      tailBase,
      new THREE.Vector3(0, backY + 0.09, -bodyLen * 0.70),
      new THREE.Vector3(0, backY + 0.24, -bodyLen * 0.66),
      new THREE.Vector3(0, backY + 0.34, -bodyLen * 0.48),
      new THREE.Vector3(0, backY + 0.36, -bodyLen * 0.30),
    ] : [
      tailBase,
      new THREE.Vector3(0, backY + 0.07, -bodyLen * 0.64),
      new THREE.Vector3(0, backY + 0.17, -bodyLen * 0.62),
      new THREE.Vector3(0, backY + 0.24, -bodyLen * 0.50),
    ]);
    const tail = solid(new THREE.Mesh(
      tubeAlongCurve(THREE, tailCurve, (u) => spec.tailR * (1 - 0.45 * u), 20, 10), coat,
    ));
    pet.add(tail);
    pet.userData.tail = tail;

    /* legs --------------------------------------------------------------- */

    // Each leg is a hip pivot with a knee below it, so the gait genuinely
    // articulates instead of sliding the whole animal along.
    const legs = [];
    [[-1, 1], [1, 1], [-1, -1], [1, -1]].forEach(([sx, sz]) => {
      const front = sz > 0;
      const hip = new THREE.Group();
      hip.position.set(
        sx * bodyR * (front ? 0.62 : 0.72),
        backY - bodyR * 0.30,
        sz * bodyLen * (front ? 0.30 : 0.34),
      );
      const upper = solid(new THREE.Mesh(new THREE.CapsuleGeometry(legR, legH * 0.40, 4, 10), coat));
      upper.position.y = -legH * 0.26;
      hip.add(upper);
      const knee = new THREE.Group();
      knee.position.y = -legH * 0.48;
      const lower = solid(new THREE.Mesh(new THREE.CapsuleGeometry(legR * 0.8, legH * 0.38, 4, 10), coat));
      lower.position.y = -legH * 0.24;
      knee.add(lower);
      const paw = new THREE.Mesh(new THREE.SphereGeometry(legR * 1.45, 10, 8), belly);
      paw.scale.set(1, 0.7, 1.3);
      paw.position.set(0, -legH * 0.48, legR * 0.35);
      knee.add(paw);
      hip.add(knee);
      pet.add(hip);
      // Diagonal pairs move together, which is what a real walk looks like.
      legs.push({ hip, knee, phase: (sx * sz > 0 ? 0 : Math.PI), rest: front ? 0 : -0.06 });
    });
    pet.userData.legs = legs;
    pet.userData.materials = materials;
    return pet;
  };

  const buildPets = (data) => {
    clearGroup(petsGroup, true);
    const scale = Math.min(1.38, Math.max(0.68, data.squareFeet / 850));
    const kinds = [];
    if (data.hasCats) kinds.push('cat');
    if (data.hasDogs) kinds.push('dog');
    kinds.forEach((kind, index) => {
      const pet = buildPet(kind);
      pet.scale.setScalar(0.95 * scale);
      pet.position.y = -0.23;
      pet.userData.radius = (2.9 + (index % 2) * 0.9) * scale;
      pet.userData.speed = 0.18 + index * 0.05;
      pet.userData.direction = index % 2 === 0 ? 1 : -1;
      pet.userData.angleOffset = (index / kinds.length) * Math.PI * 2 + (kind === 'dog' ? Math.PI / 3 : 0);
      petsGroup.add(pet);
    });
  };

  /* frame loop ----------------------------------------------------------- */

  let geometryInfo = { scale: 1, floors: 5, topY: 4 };
  const timer = new THREE.Timer();
  const cameraTarget = new THREE.Vector3();

  // Orbit the building like a product model: drag to circle it, drag up/down to
  // change the viewing height, scroll to move in and out. Constrained so the
  // camera can never drop below the lawn or end up inside the walls.
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.rotateSpeed = 0.75;
  controls.zoomSpeed = 0.8;
  controls.minPolarAngle = 0.22;
  controls.maxPolarAngle = 1.42;
  camera.position.set(7.4, 5.4, 9.2);
  controls.target.set(0, 2.6, 0);
  controls.update();
  // Remembered so switching to Room view and back returns you to the same
  // vantage point instead of snapping to a top-down orbit.
  const savedOrbit = { position: camera.position.clone(), target: controls.target.clone() };

  const frameBuilding = (floors) => {
    const height = 0.19 + floors * 0.74;
    controls.target.set(0, height * 0.45, 0);
    controls.minDistance = height * 1.15 + 3.2;
    controls.maxDistance = height * 1.9 + 9;
    const distance = THREE.MathUtils.clamp(
      camera.position.distanceTo(controls.target),
      controls.minDistance,
      controls.maxDistance,
    );
    camera.position.sub(controls.target).setLength(distance).add(controls.target);
    controls.update();
    savedOrbit.position.copy(camera.position);
    savedOrbit.target.copy(controls.target);
  };

  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    renderer.setSize(rect.width, rect.height, false);
    camera.aspect = rect.width / rect.height;
    camera.updateProjectionMatrix();
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(canvas);
  resize();

  const frame = () => {
    const reduced = reducedMotion();
    timer.update();
    const elapsed = timer.getElapsed();
    const rooms = state.view === 'rooms';

    const shader = grassMaterial.userData.shader;
    if (shader) shader.uniforms.uTime.value = reduced ? 0 : elapsed;

    petsGroup.children.forEach((pet) => {
      const info = pet.userData;
      const t = reduced ? info.angleOffset : elapsed * info.speed * info.direction + info.angleOffset;
      const x = Math.cos(t) * info.radius;
      const z = Math.sin(t) * info.radius;
      const nextT = t + 0.02 * info.direction;
      pet.position.set(x, -0.23 + (reduced ? 0 : Math.abs(Math.sin(elapsed * 6 + info.angleOffset)) * 0.02), z);
      if (reduced) return;
      pet.rotation.y = Math.atan2(Math.cos(nextT) * info.radius - x, Math.sin(nextT) * info.radius - z);
      const stride = elapsed * 9 + info.angleOffset;
      info.tail.rotation.y = Math.sin(stride * 0.9) * 0.32;
      info.tail.rotation.x = Math.sin(stride * 0.45) * 0.10;
      // Head bobs against the stride and leans into the turn.
      info.head.rotation.x = Math.sin(stride * 2) * 0.05;
      info.head.rotation.z = -info.direction * 0.06;
      info.legs.forEach((leg) => {
        const swing = stride + leg.phase;
        leg.hip.rotation.x = leg.rest + Math.sin(swing) * 0.55;
        leg.knee.rotation.x = Math.max(0, Math.sin(swing - 0.9)) * 0.5;
      });
    });

    exterior.visible = !rooms;
    interior.visible = rooms;
    petsGroup.visible = !rooms;
    plants.visible = !rooms;
    grass.visible = !rooms;
    ground.visible = !rooms;

    if (rooms) {
      // Room view is a fixed plan shot, so the orbit is parked while it shows.
      // Looking straight down makes lookAt degenerate against a +Y up vector,
      // which is what left the plan tumbling to an arbitrary angle; pointing up
      // along -Z instead pins the layout square to the screen.
      camera.position.copy(cameraTarget.set(0, 8.6, 0.001));
      camera.lookAt(0, 0, 0);
    } else {
      controls.update();
    }
    renderer.render(scene, camera);
  };

  const shouldRun = () => state.visible && !document.hidden && !state.disposed;
  let running = false;
  const sync = () => {
    if (shouldRun() && !running) {
      running = true;
      renderer.setAnimationLoop(frame);
    } else if (!shouldRun() && running) {
      running = false;
      renderer.setAnimationLoop(null);
    }
  };
  state.onVisibility = sync;
  state.onInteract = () => { if (!running) frame(); };
  state.onViewChange = (next) => {
    if (next === 'rooms') {
      savedOrbit.position.copy(camera.position);
      savedOrbit.target.copy(controls.target);
      controls.enabled = false;
      camera.up.set(0, 0, -1);
    } else {
      // Restore the world up before handing control back — OrbitControls reads
      // camera.up on every update.
      camera.up.set(0, 1, 0);
      camera.position.copy(savedOrbit.position);
      controls.target.copy(savedOrbit.target);
      controls.enabled = true;
      controls.update();
    }
    if (!running) frame();
  };
  // Keep the picture current when the user orbits while the loop is parked
  // (component scrolled off screen, or reduced motion with nothing animating).
  controls.addEventListener('change', () => { if (!running) frame(); });
  sync();

  return {
    apply: (data) => {
      geometryInfo = buildExterior(data);
      buildInterior(data);
      buildPets(data);
      // Re-frame for the new floor count so a 9-storey tower and a 4-storey
      // block are both fully in shot without the user having to zoom.
      frameBuilding(geometryInfo.floors);
      if (!running) frame();
    },
    dispose: () => {
      // Real teardown: here the shared materials and their canvas textures are
      // genuinely finished with, so unlike a rebuild everything goes.
      renderer.setAnimationLoop(null);
      controls.dispose();
      resizeObserver.disconnect();
      clearGroup(scene, true);
      [brick.map, brick.normalMap, trimNormal, furNormal, skyTexture, ground.material.map].forEach((texture) => {
        if (texture) texture.dispose();
      });
      Object.values(materials).forEach((material) => material.dispose());
      Object.values(interiorMaterials).forEach((material) => material.dispose());
      [grassMaterial, foliageMaterial, barkMaterial].forEach((material) => material.dispose());
      environment.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
}

/* ------------------------------------------------- 2D fallback (no WebGL) */

function startCanvasFallback(canvas, state, setView) {
  // No camera here, just a flat projection that spins — so the hint says
  // "explore", not "orbit".
  state.flat = true;
  const ctx = canvas.getContext('2d');
  let data = { bedrooms: 0, bathrooms: 1, squareFeet: 850, hasCats: false, hasDogs: false };
  let raf = 0;
  const start = performance.now();

  const roundRect = (x, y, w, h, radius) => {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, radius);
    ctx.fill();
    ctx.stroke();
  };

  const drawPets = (elapsed, bx, by, bw, bh, scale) => {
    const baseY = by + bh + 16 * scale;
    const unit = 15 * scale;
    const kinds = [];
    if (data.hasCats) kinds.push('cat');
    if (data.hasDogs) kinds.push('dog');
    kinds.forEach((kind, i) => {
      const cat = kind === 'cat';
      const speed = 0.35 + i * 0.12;
      const dir = i % 2 === 0 ? 1 : -1;
      const range = bw * 0.9 + i * 10;
      const t = elapsed * speed * dir + i;
      const px = bx + bw / 2 + (Math.sin(t) * range) / 2;
      const py = baseY + (i % 2) * 16 * scale;
      const flip = Math.cos(t) < 0 ? -1 : 1;
      const stride = Math.sin(elapsed * 8 + i);
      ctx.save();
      ctx.translate(px, py - Math.abs(stride) * unit * 0.1);
      ctx.scale(flip, 1);
      const bodyColor = cat ? '#f0c9a0' : '#c98a52';
      const bellyColor = cat ? '#fff3e6' : '#f0dcc0';
      const bodyLen = cat ? unit * 1.7 : unit * 2.0;
      const bodyH = cat ? unit * 0.62 : unit * 0.85;
      ctx.strokeStyle = bodyColor;
      ctx.lineWidth = unit * 0.3;
      ctx.lineCap = 'round';
      [[-bodyLen * 0.32, stride * unit * 0.16], [bodyLen * 0.32, -stride * unit * 0.16]].forEach(([lx, sw]) => {
        ctx.beginPath();
        ctx.moveTo(lx, -bodyH * 0.15);
        ctx.lineTo(lx + sw, bodyH * 0.55);
        ctx.stroke();
      });
      const tailWag = Math.sin(elapsed * 8 + i) * 0.25;
      ctx.beginPath();
      ctx.moveTo(-bodyLen * 0.48, -bodyH * 0.1);
      ctx.quadraticCurveTo(-bodyLen * (0.78 + tailWag * 0.2), -bodyH * (cat ? 1.0 : 0.3), -bodyLen * (0.64 + tailWag * 0.2), -bodyH * (cat ? 1.35 : 0.5));
      ctx.lineWidth = unit * (cat ? 0.16 : 0.24);
      ctx.stroke();
      ctx.fillStyle = bodyColor;
      ctx.strokeStyle = 'rgba(0,0,0,.12)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.ellipse(0, -bodyH * 0.3, bodyLen * 0.5, bodyH * 0.5, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = bellyColor;
      ctx.beginPath();
      ctx.ellipse(0, -bodyH * 0.05, bodyLen * 0.42, bodyH * 0.28, 0, 0, Math.PI * 2);
      ctx.fill();
      const headR = unit * 0.62;
      const headCx = bodyLen * 0.5 + headR * 0.5;
      const headCy = -bodyH * 0.55;
      ctx.fillStyle = bodyColor;
      ctx.strokeStyle = 'rgba(0,0,0,.12)';
      ctx.beginPath();
      ctx.arc(headCx, headCy, headR, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      if (cat) {
        ctx.moveTo(headCx - headR * 0.55, headCy - headR * 0.6);
        ctx.lineTo(headCx - headR * 0.7, headCy - headR * 1.5);
        ctx.lineTo(headCx - headR * 0.05, headCy - headR * 0.8);
        ctx.closePath();
        ctx.moveTo(headCx + headR * 0.3, headCy - headR * 0.75);
        ctx.lineTo(headCx + headR * 0.5, headCy - headR * 1.55);
        ctx.lineTo(headCx + headR * 0.9, headCy - headR * 0.5);
        ctx.closePath();
      } else {
        ctx.ellipse(headCx - headR * 0.7, headCy - headR * 0.05, headR * 0.28, headR * 0.5, -0.4, 0, Math.PI * 2);
        ctx.ellipse(headCx + headR * 0.75, headCy - headR * 0.05, headR * 0.28, headR * 0.5, 0.4, 0, Math.PI * 2);
      }
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = bellyColor;
      ctx.beginPath();
      ctx.ellipse(headCx + headR * 0.55, headCy + headR * 0.25, headR * 0.46, headR * 0.36, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#2b2320';
      ctx.beginPath();
      ctx.arc(headCx + headR * 0.55, headCy - headR * 0.08, headR * 0.14, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = cat ? '#d97a8f' : '#3a2a22';
      ctx.beginPath();
      ctx.arc(headCx + headR * 0.95, headCy + headR * 0.18, headR * 0.11, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  };

  const drawBuilding = (w, h, elapsed, groundY) => {
    const floors = Math.min(9, Math.max(4, Math.round(data.squareFeet / 185)));
    const scale = Math.min(1.38, Math.max(0.68, data.squareFeet / 850)) * state.zoom;
    const bw = 118 * scale;
    const bh = floors * 30 * scale;
    const side = Math.sin(state.targetRotation) * 42 * scale;
    const x = w / 2 - bw / 2;
    const y = groundY - bh;
    ctx.fillStyle = '#c9a876';
    ctx.strokeStyle = '#8a6f4e';
    ctx.lineWidth = 1.25;
    roundRect(x, y, bw, bh, 8);
    ctx.fillStyle = 'rgba(150,120,80,.55)';
    ctx.beginPath();
    ctx.moveTo(x + bw, y);
    ctx.lineTo(x + bw + side, y - 18);
    ctx.lineTo(x + bw + side, y + bh - 18);
    ctx.lineTo(x + bw, y + bh);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    for (let floor = 0; floor < floors; floor += 1) {
      for (let column = 0; column < 3; column += 1) {
        ctx.fillStyle = column < data.bedrooms ? '#cfe9f2' : '#33424d';
        ctx.fillRect(x + 14 * scale + column * 34 * scale, y + 10 * scale + floor * 30 * scale, 21 * scale, 13 * scale);
      }
    }
    ctx.fillStyle = '#54524c';
    ctx.fillRect(x - 8 * scale, y - 7 * scale, bw + 16 * scale, 7 * scale);
    if (data.hasCats || data.hasDogs) drawPets(elapsed, x, y, bw, bh, scale);
  };

  const drawRooms = (w, h) => {
    const bathCount = Math.ceil(data.bathrooms);
    const unitW = Math.min(w * 0.78, Math.max(260, data.squareFeet / 3.6)) * state.zoom;
    const unitH = Math.min(h * 0.64, Math.max(180, data.squareFeet / 5.5)) * state.zoom;
    const x = w / 2 - unitW / 2;
    const y = h / 2 - unitH / 2 + 24;
    ctx.fillStyle = '#232a33';
    ctx.strokeStyle = 'rgba(255,255,255,.28)';
    ctx.lineWidth = 2;
    roundRect(x, y, unitW, unitH, 8);
    const rows = Math.max(1, Math.ceil(Math.max(1, data.bedrooms) / 2));
    const bedW = (unitW * 0.62) / 2 - 8;
    const bedH = (unitH * 0.58) / rows - 8;
    for (let room = 0; room < data.bedrooms; room += 1) {
      const rx = x + 9 + (room % 2) * (bedW + 8);
      const ry = y + 9 + Math.floor(room / 2) * (bedH + 8);
      ctx.fillStyle = 'rgba(226,160,107,.74)';
      ctx.strokeStyle = 'rgba(255,212,173,.80)';
      roundRect(rx, ry, bedW, bedH, 6);
      ctx.fillStyle = '#fff3e6';
      ctx.font = '600 11px system-ui';
      ctx.fillText('Bedroom ' + (room + 1), rx + 7, ry + 19);
    }
    const bathW = unitW * 0.29;
    const bathH = (unitH * 0.48) / bathCount - 7;
    for (let bath = 0; bath < bathCount; bath += 1) {
      const bx = x + unitW * 0.69;
      const by = y + 9 + bath * (bathH + 8);
      ctx.fillStyle = 'rgba(185,217,234,.72)';
      ctx.strokeStyle = 'rgba(220,241,255,.78)';
      roundRect(bx, by, bathW, bathH, 6);
      ctx.fillStyle = '#f1f9ff';
      ctx.font = '600 10px system-ui';
      ctx.fillText(data.bathrooms % 1 && bath === bathCount - 1 ? 'Half bath' : 'Bath ' + (bath + 1), bx + 7, by + 18);
    }
    ctx.fillStyle = 'rgba(99,116,139,.72)';
    ctx.strokeStyle = 'rgba(185,202,225,.52)';
    roundRect(x + 9, y + unitH * 0.7, unitW * 0.62 - 9, unitH * 0.22, 6);
    ctx.fillStyle = '#edf3ff';
    ctx.font = '600 11px system-ui';
    ctx.fillText('Living / dining', x + 18, y + unitH * 0.7 + 20);
    roundRect(x + unitW * 0.69, y + unitH * 0.58, bathW, unitH * 0.34, 6);
    ctx.fillText('Entry / kitchen', x + unitW * 0.7, y + unitH * 0.58 + 20);
  };

  const draw = () => {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
    if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
      canvas.width = w * ratio;
      canvas.height = h * ratio;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const elapsed = (performance.now() - start) / 1000;
    if (state.view === 'building') {
      const sky = ctx.createLinearGradient(0, 0, 0, h);
      sky.addColorStop(0, '#4a90d9');
      sky.addColorStop(0.72, '#bfe0f5');
      sky.addColorStop(1, '#e7f4fb');
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, w, h);
      const groundY = h * 0.78;
      const grass = ctx.createLinearGradient(0, groundY, 0, h);
      grass.addColorStop(0, '#6fae4a');
      grass.addColorStop(1, '#3f7a2e');
      ctx.fillStyle = grass;
      ctx.fillRect(0, groundY, w, h - groundY);
      for (let i = 0; i < 90; i += 1) {
        const gx = (i * 53 + 17) % w;
        const gy = groundY + (((i * 97) % 1000) / 1000) * (h - groundY);
        ctx.strokeStyle = i % 3 === 0 ? 'rgba(255,255,255,.10)' : 'rgba(0,0,0,.12)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(gx, gy);
        ctx.lineTo(gx + 2, gy - 5);
        ctx.stroke();
      }
      drawBuilding(w, h, elapsed, groundY);
    } else {
      const bg = ctx.createLinearGradient(0, 0, w, h);
      bg.addColorStop(0, '#2a3038');
      bg.addColorStop(1, '#12161b');
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);
      drawRooms(w, h);
    }
  };

  const animated = () => (data.hasCats || data.hasDogs) && state.view === 'building' && !reducedMotion() && state.visible && !document.hidden;
  const loop = () => {
    if (!animated()) { raf = 0; return; }
    draw();
    raf = requestAnimationFrame(loop);
  };
  const sync = () => {
    draw();
    if (animated() && !raf) raf = requestAnimationFrame(loop);
  };
  state.onVisibility = sync;
  state.onInteract = draw;
  state.onViewChange = sync;

  // The fallback has no camera to orbit, so it keeps the original drag-to-spin
  // and scroll-to-zoom on its flat projection.
  const onPointerDown = (event) => {
    state.dragging = true;
    state.lastX = event.clientX;
    canvas.setPointerCapture(event.pointerId);
  };
  const onPointerMove = (event) => {
    if (!state.dragging) return;
    state.targetRotation += (event.clientX - state.lastX) * 0.018;
    state.lastX = event.clientX;
    draw();
  };
  const onPointerUp = () => { state.dragging = false; };
  const onWheel = (event) => {
    event.preventDefault();
    state.zoom = Math.min(1.22, Math.max(0.78, state.zoom - event.deltaY * 0.0007));
    draw();
  };
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });

  const resizeObserver = new ResizeObserver(draw);
  resizeObserver.observe(canvas);

  return {
    apply: (next) => { data = next; sync(); },
    dispose: () => {
      if (raf) cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('wheel', onWheel);
    },
  };
}
"""


_PROPERTY_VIEWER = st.components.v2.component(
    "property_viewer",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def render_property_viewer(
    *,
    city: str,
    state: str,
    bedrooms: float,
    bathrooms: float,
    square_feet: int,
    predicted_rent: float,
    pets_allowed: str = "Not Specified",
    height: int = 520,
) -> None:
    """Render the interactive preview from the live predictor values."""
    _PROPERTY_VIEWER(
        data={
            "location": f"{city}, {state}",
            "bedrooms": max(0, int(bedrooms)),
            "bathrooms": float(bathrooms),
            "squareFeet": int(square_feet),
            "rent": f"${predicted_rent:,.0f} / month",
            "hasCats": "Cats" in pets_allowed,
            "hasDogs": "Dogs" in pets_allowed,
            "height": height,
        },
        height=height,
    )
