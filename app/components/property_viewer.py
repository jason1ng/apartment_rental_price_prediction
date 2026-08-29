"""Browser-compatible interactive property preview for the predictor page."""

from __future__ import annotations

from functools import lru_cache
import html
import json
from pathlib import Path

import streamlit.components.v1 as components


THREE_RUNTIME = Path(__file__).resolve().parents[1] / "static" / "three.min.js"


@lru_cache
def _three_source() -> str:
    return THREE_RUNTIME.read_text(encoding="utf-8")


def render_property_viewer(
    *,
    city: str,
    state: str,
    bedrooms: float,
    bathrooms: float,
    square_feet: int,
    predicted_rent: float,
    height: int = 520,
) -> None:
    """Render an interactive Canvas preview from the live predictor values."""
    data = {
        "location": f"{city}, {state}",
        "bedrooms": max(0, int(bedrooms)),
        "bathrooms": float(bathrooms),
        "squareFeet": int(square_feet),
        "rent": f"${predicted_rent:,.0f} / month",
    }
    payload = json.dumps(data).replace("</", "<\\/")
    try:
        three_source = _three_source()
    except OSError:
        three_source = ""
    viewer = f"""
    <div class="preview-shell">
      <canvas id="property-canvas" aria-label="Interactive property preview"></canvas>
      <div class="location">{html.escape(data['location'])}</div>
      <div class="rent"><span>Predicted rent</span><strong>{data['rent']}</strong></div>
      <div class="view-toggle"><button data-view="building" class="active">Building</button><button data-view="rooms">Room view</button></div>
      <div class="summary">{data['bedrooms']} bed&nbsp; · &nbsp;{data['bathrooms']:g} bath&nbsp; · &nbsp;{data['squareFeet']:,} sq ft</div>
      <div class="room-labels" id="room-labels"></div>
      <div class="hint" id="hint">Drag to explore · Scroll to zoom</div>
    </div>
    <style>
      html, body {{ margin: 0; overflow: hidden; background: transparent; }}
      .preview-shell {{ position: relative; height: {height}px; overflow: hidden; border-radius: 14px; background: #0c1320; }}
      canvas {{ display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }} canvas:active {{ cursor: grabbing; }}
      .location, .rent, .summary, .hint, .view-toggle, .room-labels {{ position: absolute; z-index: 2; border: 1px solid rgba(255,255,255,.20); border-radius: 10px; color: #f4f7ff; background: rgba(24,31,45,.58); box-shadow: 0 8px 24px rgba(0,0,0,.16); backdrop-filter: blur(12px); font: 600 11px/1.25 system-ui, sans-serif; }}
      .location {{ left: 16px; top: 16px; padding: 8px 10px; }} .rent {{ right: 16px; top: 16px; padding: 8px 10px; text-align: right; }} .rent span {{ display: block; color: #c3cad9; font-size: 10px; letter-spacing: .06em; text-transform: uppercase; }} .rent strong {{ display: block; margin-top: 3px; font-size: 17px; }}
      .view-toggle {{ top: 70px; left: 16px; display: flex; padding: 3px; }} .view-toggle button {{ border: 0; border-radius: 7px; padding: 6px 8px; color: #d6dceb; background: transparent; font: 650 11px/1 system-ui, sans-serif; cursor: pointer; }} .view-toggle button.active {{ color: white; background: rgba(255,255,255,.18); }}
      .summary {{ left: 16px; bottom: 16px; padding: 7px 9px; }} .hint {{ right: 16px; bottom: 16px; padding: 7px 9px; color: #d1d8e8; }}
      .room-labels {{ display: none; top: 112px; left: 16px; max-width: 230px; padding: 8px 10px; line-height: 1.65; }} .room-labels strong {{ display: block; font-size: 12px; }} .room-labels span {{ display: inline-block; margin: 2px 4px 2px 0; padding: 2px 5px; border-radius: 5px; }} .room-labels .bedroom {{ color: #ffd4ad; background: rgba(208, 123, 67, .22); }} .room-labels .bathroom {{ color: #c7efff; background: rgba(112, 195, 232, .20); }}
      .switching {{ animation: switch .24s ease-out both; }} @keyframes switch {{ from {{ opacity: .22; transform: scale(.97); }} to {{ opacity: 1; transform: scale(1); }} }}
      @media (prefers-reduced-motion: reduce) {{ .hint {{ display: none; }} .switching {{ animation: none; }} }}
    </style>
    <script>{three_source}</script>
    <script>
      (() => {{
        const data = {payload};
        const canvas = document.getElementById('property-canvas');
        const startThreePreview = () => {{
          const THREE = window.THREE;
          const renderer = new THREE.WebGLRenderer({{canvas: canvas, antialias: true, alpha: true}});
          renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
          renderer.outputColorSpace = THREE.SRGBColorSpace;
          const scene = new THREE.Scene();
          const seed = Array.from(data.location).reduce((sum, char) => (sum * 31 + char.charCodeAt(0)) % 360, 0);
          const hue = 185 + seed % 105;
          const accent = new THREE.Color().setHSL(hue / 360, .54, .67);
          renderer.setClearColor(new THREE.Color().setHSL(hue / 360, .30, .11), .55);
          const camera = new THREE.PerspectiveCamera(38, 1, .1, 100);
          const exterior = new THREE.Group();
          const interior = new THREE.Group();
          const facade = new THREE.MeshStandardMaterial({{color: new THREE.Color().setHSL(hue / 360, .22, .42), roughness: .54, metalness: .16}});
          const trim = new THREE.MeshStandardMaterial({{color: new THREE.Color().setHSL(hue / 360, .30, .72), roughness: .35, metalness: .30}});
          const glass = new THREE.MeshStandardMaterial({{color: accent, roughness: .18, metalness: .26, emissive: accent, emissiveIntensity: .45}});
          const litGlass = new THREE.MeshStandardMaterial({{color: 0xdaf4f4, roughness: .20, emissive: accent, emissiveIntensity: 1.25}});
          const scale = Math.min(1.38, Math.max(.68, data.squareFeet / 850));
          const floors = Math.min(9, Math.max(4, Math.round(data.squareFeet / 185)));
          exterior.add(new THREE.Mesh(new THREE.BoxGeometry(3.55 * scale, .42, 2.8 * scale), trim));
          for (let floor = 0; floor < floors; floor += 1) {{
            const shell = new THREE.Mesh(new THREE.BoxGeometry(3.1 * scale, .70, 2.35 * scale), facade);
            shell.position.y = .42 + floor * .74; exterior.add(shell);
            const balcony = new THREE.Mesh(new THREE.BoxGeometry(3.42 * scale, .08, .32), trim);
            balcony.position.set(0, .70 + floor * .74, 1.30 * scale); exterior.add(balcony);
            for (let column = 0; column < 3; column += 1) {{
              const windowMesh = new THREE.Mesh(new THREE.PlaneGeometry(.58 * scale, .38), column < data.bedrooms ? litGlass : glass);
              windowMesh.position.set(-.94 * scale + column * .94 * scale, .47 + floor * .74, 1.18 * scale + .01); exterior.add(windowMesh);
            }}
          }}
          const roof = new THREE.Mesh(new THREE.BoxGeometry(3.34 * scale, .18, 2.56 * scale), trim);
          roof.position.y = .42 + floors * .74; exterior.add(roof); exterior.rotation.y = -.48; scene.add(exterior);
          const unitW = Math.min(5.8, Math.max(3.8, data.squareFeet / 220)), unitD = Math.min(4.8, Math.max(3.1, data.squareFeet / 285));
          const floorPlan = new THREE.Mesh(new THREE.BoxGeometry(unitW, .12, unitD), new THREE.MeshStandardMaterial({{color: new THREE.Color().setHSL(hue / 360, .20, .34), roughness: .8}}));
          interior.add(floorPlan);
          const roomMaterial = new THREE.MeshStandardMaterial({{color: 0xe2a06b, roughness: .42, emissive: 0x6d3e20, emissiveIntensity: .16}});
          const bathMaterial = new THREE.MeshStandardMaterial({{color: 0xb9d9ea, roughness: .32, emissive: 0x325267, emissiveIntensity: .25}});
          const livingMaterial = new THREE.MeshStandardMaterial({{color: 0x62738b, roughness: .58, emissive: 0x182333, emissiveIntensity: .18}});
          const bedroomRows = Math.max(1, Math.ceil(Math.max(1, data.bedrooms) / 2));
          const bedroomZoneW = unitW * .62, bedroomWidth = bedroomZoneW / 2 - .08, bedroomDepth = unitD * .58 / bedroomRows - .08;
          for (let room = 0; room < data.bedrooms; room += 1) {{
            const row = Math.floor(room / 2), col = room % 2;
            const block = new THREE.Mesh(new THREE.BoxGeometry(bedroomWidth, .18, bedroomDepth), roomMaterial);
            block.position.set(-unitW / 2 + bedroomWidth / 2 + .08 + col * (bedroomWidth + .08), .13, unitD / 2 - bedroomDepth / 2 - .08 - row * (bedroomDepth + .08)); interior.add(block);
          }}
          const bathCount = Math.ceil(data.bathrooms);
          for (let bath = 0; bath < bathCount; bath += 1) {{
            const block = new THREE.Mesh(new THREE.BoxGeometry(unitW * .29, .2, unitD * .48 / bathCount - .07), bathMaterial);
            block.position.set(unitW * .34, .14, unitD / 2 - unitD * .24 / bathCount - .10 - bath * (unitD * .48 / bathCount)); interior.add(block);
          }}
          const living = new THREE.Mesh(new THREE.BoxGeometry(unitW * .62, .15, unitD * .31), livingMaterial);
          living.position.set(-unitW * .18, .11, -unitD * .31); interior.add(living);
          const entry = new THREE.Mesh(new THREE.BoxGeometry(unitW * .29, .15, unitD * .36), livingMaterial);
          entry.position.set(unitW * .34, .11, -unitD * .27); interior.add(entry);
          interior.visible = false; scene.add(interior);
          const ground = new THREE.Mesh(new THREE.CircleGeometry(5.8, 48), new THREE.MeshStandardMaterial({{color: 0x182334, roughness: 1}}));
          ground.rotation.x = -Math.PI / 2; ground.position.y = -.23; scene.add(ground);
          scene.add(new THREE.HemisphereLight(0xd7ecff, 0x142033, 2.2));
          const key = new THREE.DirectionalLight(0xe2f1ff, 2.5); key.position.set(4, 7, 5); scene.add(key);
          const glow = new THREE.PointLight(accent, 7, 12); glow.position.set(-4, 3, 2); scene.add(glow);
          const buttons = document.querySelectorAll('.view-toggle button'), labels = document.getElementById('room-labels'), hint = document.getElementById('hint');
          const roomNames = Array.from({{length: data.bedrooms}}, (_, i) => 'Bedroom ' + (i + 1));
          const bathNames = Array.from({{length: bathCount}}, (_, i) => data.bathrooms % 1 && i === bathCount - 1 ? 'Half bath' : 'Bathroom ' + (i + 1));
          labels.innerHTML = '<strong>Room layout</strong>' + roomNames.map((name) => '<span class="bedroom">' + name + '</span>').concat(bathNames.map((name) => '<span class="bathroom">' + name + '</span>')).join('');
          let view = 'building', dragging = false, lastX = 0, targetRotation = exterior.rotation.y, zoom = 1;
          const setView = (next) => {{ view = next; const rooms = view === 'rooms'; exterior.visible = !rooms; interior.visible = rooms; labels.style.display = rooms ? 'block' : 'none'; hint.textContent = rooms ? 'Room layout · Scroll to zoom' : 'Drag to explore · Scroll to zoom'; buttons.forEach((button) => button.classList.toggle('active', button.dataset.view === view)); canvas.classList.remove('switching'); void canvas.offsetWidth; canvas.classList.add('switching'); }};
          buttons.forEach((button) => button.addEventListener('click', () => setView(button.dataset.view)));
          canvas.addEventListener('pointerdown', (event) => {{ dragging = true; lastX = event.clientX; canvas.setPointerCapture(event.pointerId); }});
          canvas.addEventListener('pointermove', (event) => {{ if (dragging) {{ targetRotation += (event.clientX - lastX) * .012; lastX = event.clientX; }} }});
          canvas.addEventListener('pointerup', () => {{ dragging = false; }});
          canvas.addEventListener('wheel', (event) => {{ event.preventDefault(); zoom = Math.min(1.22, Math.max(.78, zoom - event.deltaY * .0007)); }}, {{passive: false}});
          const resize = () => {{ const rect = canvas.getBoundingClientRect(); renderer.setSize(rect.width, rect.height, false); camera.aspect = rect.width / rect.height; camera.updateProjectionMatrix(); }};
          new ResizeObserver(resize).observe(canvas); resize();
          const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches, clock = new THREE.Clock();
          const render = () => {{ const elapsed = clock.getElapsedTime(); exterior.rotation.y += (targetRotation - exterior.rotation.y) * .08; exterior.position.y = reduced ? 0 : Math.sin(elapsed * .75) * .08; const target = view === 'rooms' ? new THREE.Vector3(0, 8.3 * zoom, .12) : new THREE.Vector3(6.8 * zoom, (4.4 + floors * .18) * zoom, 8.8 * zoom); camera.position.lerp(target, .10); camera.lookAt(0, view === 'rooms' ? 0 : 1.9 + floors * .28, 0); renderer.render(scene, camera); requestAnimationFrame(render); }};
          render(); return true;
        }};
        if (window.THREE) {{ try {{ if (startThreePreview()) return; }} catch (error) {{ console.warn('Three preview failed; using Canvas fallback', error); }} }}
        const ctx = canvas.getContext('2d');
        const buttons = document.querySelectorAll('.view-toggle button');
        const labels = document.getElementById('room-labels');
        const hint = document.getElementById('hint');
        const bathroomCount = Math.ceil(data.bathrooms);
        const roomNames = Array.from({{length: data.bedrooms}}, (_, i) => 'Bedroom ' + (i + 1));
        const bathNames = Array.from({{length: bathroomCount}}, (_, i) => data.bathrooms % 1 && i === bathroomCount - 1 ? 'Half bath' : 'Bathroom ' + (i + 1));
        labels.innerHTML = '<strong>Room layout</strong>' + roomNames.map((name) => '<span class="bedroom">' + name + '</span>').concat(bathNames.map((name) => '<span class="bathroom">' + name + '</span>')).join('');
        const seed = Array.from(data.location).reduce((sum, char) => (sum * 31 + char.charCodeAt(0)) % 360, 0);
        const hue = 185 + seed % 105;
        let view = 'building', rotation = -.4, zoom = 1, dragging = false, startX = 0;
        const roundRect = (x, y, w, h, radius) => {{ ctx.beginPath(); ctx.roundRect(x, y, w, h, radius); ctx.fill(); ctx.stroke(); }};
        const drawBuilding = (w, h) => {{
          const floors = Math.min(9, Math.max(4, Math.round(data.squareFeet / 185)));
          const scale = Math.min(1.38, Math.max(.68, data.squareFeet / 850)) * zoom;
          const bw = 118 * scale, bh = floors * 30 * scale, side = Math.sin(rotation) * 42 * scale;
          const x = w / 2 - bw / 2, y = h / 2 - bh / 2 + 18;
          ctx.fillStyle = 'hsl(' + hue + ', 28%, 31%)'; ctx.strokeStyle = 'hsla(' + hue + ', 58%, 78%, .6)'; ctx.lineWidth = 1.25; roundRect(x, y, bw, bh, 8);
          ctx.fillStyle = 'hsla(' + hue + ', 42%, 62%, .35)'; ctx.beginPath(); ctx.moveTo(x + bw, y); ctx.lineTo(x + bw + side, y - 18); ctx.lineTo(x + bw + side, y + bh - 18); ctx.lineTo(x + bw, y + bh); ctx.closePath(); ctx.fill(); ctx.stroke();
          for (let floor = 0; floor < floors; floor += 1) for (let column = 0; column < 3; column += 1) {{ ctx.fillStyle = column < data.bedrooms ? 'hsla(' + hue + ', 78%, 86%, .95)' : 'hsla(' + hue + ', 54%, 58%, .55)'; ctx.fillRect(x + 14 * scale + column * 34 * scale, y + 10 * scale + floor * 30 * scale, 21 * scale, 13 * scale); }}
          ctx.fillStyle = 'hsla(' + hue + ', 40%, 82%, .7)'; ctx.fillRect(x - 8 * scale, y + bh, bw + 16 * scale, 7 * scale);
        }};
        const drawRooms = (w, h) => {{
          const unitW = Math.min(w * .78, Math.max(260, data.squareFeet / 3.6)) * zoom, unitH = Math.min(h * .64, Math.max(180, data.squareFeet / 5.5)) * zoom;
          const x = w / 2 - unitW / 2, y = h / 2 - unitH / 2 + 24;
          ctx.fillStyle = 'hsl(' + hue + ', 24%, 22%)'; ctx.strokeStyle = 'hsla(' + hue + ', 65%, 78%, .7)'; ctx.lineWidth = 2; roundRect(x, y, unitW, unitH, 8);
          const rows = Math.max(1, Math.ceil(Math.max(1, data.bedrooms) / 2)), bedW = unitW * .62 / 2 - 8, bedH = unitH * .58 / rows - 8;
          for (let room = 0; room < data.bedrooms; room += 1) {{ const row = Math.floor(room / 2), col = room % 2, rx = x + 9 + col * (bedW + 8), ry = y + 9 + row * (bedH + 8); ctx.fillStyle = 'rgba(226,160,107,.74)'; ctx.strokeStyle = 'rgba(255,212,173,.80)'; roundRect(rx, ry, bedW, bedH, 6); ctx.fillStyle = '#fff3e6'; ctx.font = '600 11px system-ui'; ctx.fillText('Bedroom ' + (room + 1), rx + 7, ry + 19); }}
          const bathW = unitW * .29, bathH = unitH * .48 / bathroomCount - 7;
          for (let bath = 0; bath < bathroomCount; bath += 1) {{ const bx = x + unitW * .69, by = y + 9 + bath * (bathH + 8); ctx.fillStyle = 'rgba(185,217,234,.72)'; ctx.strokeStyle = 'rgba(220,241,255,.78)'; roundRect(bx, by, bathW, bathH, 6); ctx.fillStyle = '#f1f9ff'; ctx.font = '600 10px system-ui'; ctx.fillText(data.bathrooms % 1 && bath === bathroomCount - 1 ? 'Half bath' : 'Bath ' + (bath + 1), bx + 7, by + 18); }}
          ctx.fillStyle = 'rgba(99,116,139,.72)'; ctx.strokeStyle = 'rgba(185,202,225,.52)'; roundRect(x + 9, y + unitH * .70, unitW * .62 - 9, unitH * .22, 6); ctx.fillStyle = '#edf3ff'; ctx.font = '600 11px system-ui'; ctx.fillText('Living / dining', x + 18, y + unitH * .70 + 20);
          roundRect(x + unitW * .69, y + unitH * .58, bathW, unitH * .34, 6); ctx.fillText('Entry / kitchen', x + unitW * .70, y + unitH * .58 + 20);
        }};
        const draw = () => {{ const w = canvas.clientWidth, h = canvas.clientHeight, ratio = Math.min(window.devicePixelRatio || 1, 1.5); if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {{ canvas.width = w * ratio; canvas.height = h * ratio; }} ctx.setTransform(ratio, 0, 0, ratio, 0, 0); const bg = ctx.createLinearGradient(0, 0, w, h); bg.addColorStop(0, 'hsl(' + hue + ', 32%, 19%)'); bg.addColorStop(1, 'hsl(222, 30%, 8%)'); ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h); view === 'building' ? drawBuilding(w, h) : drawRooms(w, h); }};
        const setView = (next) => {{ view = next; labels.style.display = view === 'rooms' ? 'block' : 'none'; hint.textContent = view === 'rooms' ? 'Room layout · Scroll to zoom' : 'Drag to explore · Scroll to zoom'; buttons.forEach((button) => button.classList.toggle('active', button.dataset.view === view)); canvas.classList.remove('switching'); void canvas.offsetWidth; canvas.classList.add('switching'); draw(); }};
        buttons.forEach((button) => button.addEventListener('click', () => setView(button.dataset.view)));
        canvas.addEventListener('pointerdown', (event) => {{ dragging = true; startX = event.clientX; canvas.setPointerCapture(event.pointerId); }});
        canvas.addEventListener('pointermove', (event) => {{ if (dragging) {{ rotation += (event.clientX - startX) * .018; startX = event.clientX; draw(); }} }});
        canvas.addEventListener('pointerup', () => {{ dragging = false; }});
        canvas.addEventListener('wheel', (event) => {{ event.preventDefault(); zoom = Math.min(1.22, Math.max(.78, zoom - event.deltaY * .0007)); draw(); }}, {{passive: false}});
        new ResizeObserver(draw).observe(canvas); draw();
      }})();
    </script>
    """
    components.html(viewer, height=height, scrolling=False)
