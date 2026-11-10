// 3D head preview for the Characters tab (three.js, loaded as an ES module).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const REGION_LABEL = {
  BROW_L: "Brows", BROW_R: "Brows", EYE_L: "Eyes", EYE_R: "Eyes", NOSE_C: "Nose", EAR_L: "Ears", EAR_R: "Ears",
  CHEEK_L: "Cheeks", CHEEK_R: "Cheeks", MOUTH_C: "Mouth", JAWLINE_C: "Jaw", CROWN_C: "Head shape", NECK_C: "Neck",
};
const texCache = {};
let view = null;

function f32(b64) { const s = atob(b64); const u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; }
function prettyHead(n) {
  let m = /^(male|female)_archetype_v0*(\d+)_t\d$/.exec(n); if (m) return "Archetype " + m[2];
  m = /^(male|female)0*(\d+)_t\d$/.exec(n); if (m) return (m[1] === "male" ? "Male " : "Female ") + m[2];
  return n.replace(/_t\d$/, "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
function geometry(g) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(f32(g.positions)), 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(new Float32Array(f32(g.uvs)), 2));
  geo.setIndex(new THREE.BufferAttribute(new Uint32Array(f32(g.indices)), 1));
  geo.computeVertexNormals();
  // UV seams duplicate vertices; average normals across vertices at the same spot so seams vanish
  const p = geo.attributes.position, n = geo.attributes.normal, groups = new Map();
  for (let i = 0; i < p.count; i++) {
    const k = `${Math.round(p.getX(i) * 1e4)},${Math.round(p.getY(i) * 1e4)},${Math.round(p.getZ(i) * 1e4)}`;
    (groups.get(k) || groups.set(k, []).get(k)).push(i);
  }
  for (const ids of groups.values()) {
    if (ids.length < 2) continue;
    let x = 0, y = 0, z = 0;
    for (const i of ids) { x += n.getX(i); y += n.getY(i); z += n.getZ(i); }
    const l = Math.hypot(x, y, z) || 1;
    for (const i of ids) n.setXYZ(i, x / l, y / l, z / l);
  }
  return geo;
}
function browAlpha(geo) {
  // game space: x = across the face, z = up. Per slice of |x|, find the band's top/bottom.
  const p = geo.attributes.position, n = p.count, BINS = 48;
  let maxX = 0; for (let i = 0; i < n; i++) maxX = Math.max(maxX, Math.abs(p.getX(i)));
  const lo = new Array(BINS).fill(Infinity), hi = new Array(BINS).fill(-Infinity);
  const bin = (x) => Math.min(BINS - 1, Math.floor(Math.abs(x) / (maxX || 1) * BINS));
  for (let i = 0; i < n; i++) { const b = bin(p.getX(i)); lo[b] = Math.min(lo[b], p.getZ(i)); hi[b] = Math.max(hi[b], p.getZ(i)); }
  const ss = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const col = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const ax = Math.abs(p.getX(i)) / (maxX || 1), b = bin(p.getX(i));
    const h = hi[b] - lo[b], t = h > 1e-5 ? (p.getZ(i) - lo[b]) / h : 0.5;
    const across = Math.pow(Math.sin(Math.PI * Math.min(1, Math.max(0, t))), 0.6);   // thick in the middle of the band
    const inner = ss(0.12, 0.3, ax);                                                   // gap over the nose
    const outer = 1 - 0.75 * ss(0.75, 1.0, ax);                                        // thinner tails
    const grain = 0.8 + 0.2 * Math.abs(Math.sin(i * 12.9898) * 43758.5453 % 1);         // a little texture
    col.set([1, 1, 1, Math.min(1, 1.15 * across * inner * outer * grain)], i * 4);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(col, 4));
}
function avgColour(img) {
  const c = document.createElement("canvas"); c.width = c.height = 32;
  const x = c.getContext("2d"); x.drawImage(img, 0, 0, 32, 32);
  const d = x.getImageData(8, 8, 16, 16).data; let r = 0, g = 0, b = 0;
  for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
  const n = d.length / 4; return [r / n, g / n, b / n];
}
function skinTexture(gender) {
  if (!texCache[gender]) texCache[gender] = new Promise((res, rej) => {
    new THREE.TextureLoader().load(`/api/headtex/${gender}`, (t) => { t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; t.flipY = false; /* game UVs are DirectX-style (v=0 at top) */ res(t); }, undefined, rej);
  });
  return texCache[gender];
}
let maskImg = null;
const loadMask = () => (maskImg ||= new Promise((res) => {
  const i = new Image(); i.onload = () => res(i); i.onerror = () => { maskImg = null; res(null); }; i.src = "/api/headtex/male/makeup";
}));
// Skin texture with this character's tone, makeup and freckles baked in, using the game's own
// makeup masks (same UV layout as the skin).
// Returns null when there's nothing to add, so the plain tinted texture is kept.
async function paintSkin(img, tint, data) {
  const mk = data.makeup || {}, sd = data.skinDetail || {};
  const areas = [["eyes", 0, 0.85], ["cheeks", 1, 0.4], ["lips", 2, 0.9]]
    .filter(([a]) => (mk[a]?.opacity || 0) > 0.01);
  const freck = (sd.FrecklesAmount || 0) * (sd.FrecklesOpacity || 0), spots = (sd.SunSpotsAmount || 0) * (sd.SunSpotsOpacity || 0);
  if (!areas.length && freck < 0.01 && spots < 0.01) return null;
  const mask = await loadMask(); if (!mask) return null;
  const W = img.width, H = img.height;
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const x = c.getContext("2d", { willReadFrequently: true });
  // the mask is two squares stacked: makeup on top, freckle area below
  x.drawImage(mask, 0, 0, mask.width, mask.width, 0, 0, W, H); const m = x.getImageData(0, 0, W, H).data;
  x.drawImage(mask, 0, mask.width, mask.width, mask.width, 0, 0, W, H); const fm = x.getImageData(0, 0, W, H).data;
  x.drawImage(img, 0, 0); const id = x.getImageData(0, 0, W, H), p = id.data;
  // the material's linear tint, as an sRGB multiplier
  const k = [tint.r, tint.g, tint.b].map((v) => Math.pow(v, 1 / 2.2));
  const cols = areas.map(([a, ch, s]) => { const h = mk[a].colour; return [ch, mk[a].opacity * s, [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16))]; });
  for (let i = 0; i < p.length; i += 4) {
    let r = Math.min(255, p[i] * k[0]), g = Math.min(255, p[i + 1] * k[1]), b = Math.min(255, p[i + 2] * k[2]);
    for (const [ch, o, col] of cols) {
      const a = (m[i + ch] / 255) * o; if (a <= 0) continue;
      // part pigment, part multiply, so the skin's shading and texture still show through
      r += a * (0.6 * col[0] + 0.4 * r * col[0] / 255 - r); g += a * (0.6 * col[1] + 0.4 * g * col[1] / 255 - g); b += a * (0.6 * col[2] + 0.4 * b * col[2] / 255 - b);
    }
    p[i] = r; p[i + 1] = g; p[i + 2] = b;
  }
  x.putImageData(id, 0, 0);
  // freckles and sun spots: small brown dots scattered over the freckle area (fixed seed, so stable)
  let seed = 41; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  x.globalCompositeOperation = "multiply";
  const dots = (n, rMin, rMax, alpha) => {
    for (let tries = 0, done = 0; done < n && tries < n * 40; tries++) {
      const u = rnd(), v = rnd(); const w = fm[(Math.floor(v * H) * W + Math.floor(u * W)) * 4] / 255;
      if (rnd() > w) continue;
      done++; const rad = (rMin + rnd() * (rMax - rMin)) * W / 1024;
      x.fillStyle = `rgba(150, 95, 60, ${alpha * (0.5 + 0.5 * rnd())})`;
      x.beginPath(); x.ellipse(u * W, v * H, rad, rad * (0.7 + 0.3 * rnd()), rnd() * 3, 0, Math.PI * 2); x.fill();
    }
  };
  if (freck >= 0.01) dots(Math.round(150 + 1000 * (sd.FrecklesAmount || 0)), 0.8, 1.8, 0.25 + 0.6 * (sd.FrecklesOpacity || 0));
  if (spots >= 0.01) dots(Math.round(20 + 120 * (sd.SunSpotsAmount || 0)), 2, 5, 0.12 + 0.35 * (sd.SunSpotsOpacity || 0));
  x.globalCompositeOperation = "source-over";
  return c;
}
function eyeTexture(iris) {
  const c = document.createElement("canvas"); c.width = c.height = 256;
  const x = c.getContext("2d");
  // sclera: off-white, getting greyer and pinker towards the corners where the lids shade it
  const s = x.createRadialGradient(128, 128, 40, 128, 128, 150);
  s.addColorStop(0, "#d9d2cb"); s.addColorStop(0.55, "#c9bdb5"); s.addColorStop(1, "#8f7a72");
  x.fillStyle = s; x.fillRect(0, 0, 256, 256);
  // iris: darker rim, lighter fibres towards the pupil, dark limbal ring
  const g = x.createRadialGradient(128, 128, 10, 128, 128, 46);
  const ir = new THREE.Color(iris), light = "#" + ir.clone().lerp(new THREE.Color(0xffffff), 0.18).getHexString(), dark = "#" + ir.clone().multiplyScalar(0.55).getHexString();
  g.addColorStop(0, "#0b0b0b"); g.addColorStop(0.27, "#0b0b0b"); g.addColorStop(0.31, light); g.addColorStop(0.6, iris); g.addColorStop(0.88, dark); g.addColorStop(1, "#1d1815");
  x.fillStyle = g; x.beginPath(); x.arc(128, 128, 46, 0, Math.PI * 2); x.fill();
  x.globalAlpha = 0.18; x.strokeStyle = "#000";   // radial fibres
  for (let a = 0; a < Math.PI * 2; a += Math.PI / 36) { x.beginPath(); x.moveTo(128 + Math.cos(a) * 15, 128 + Math.sin(a) * 15); x.lineTo(128 + Math.cos(a) * 44, 128 + Math.sin(a) * 44); x.stroke(); }
  x.globalAlpha = 1;
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

function dispose() {
  if (!view) return;
  cancelAnimationFrame(view.raf); clearInterval(view.timer);
  view.ro.disconnect(); view.controls.dispose(); view.renderer.dispose();
  view.scene.traverse((o) => {
    o.geometry?.dispose();
    for (const m of [].concat(o.material || [])) {
      m.map?.dispose?.(); m.alphaMap?.dispose?.(); m.normalMap?.dispose?.(); m.specularIntensityMap?.dispose?.(); m.dispose();
    }
  });
  view.scene.environment?.dispose?.();
  view.renderer.domElement.remove();
  view = null;
}

async function show(host, recipeHost, id) {
  dispose();
  host.innerHTML = `<div class="hv-status">Building head from your character file…</div>`;
  recipeHost.innerHTML = "";
  let data;
  if (typeof id === "object") data = id;  // ready-made head payload (photo fitting)
  else try {
    const r = await fetch(`/api/chars/${id}/head`); data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || r.statusText);
  } catch (e) { host.innerHTML = `<div class="hv-status bad">Couldn't build a 3D head: ${e.message}</div>`; return; }
  if (host.dataset.forId !== String(id)) { /* stale */ }
  host.innerHTML = "";

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  // filmic tone mapping + soft shadows: closer to how the game's renderer presents skin
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.15;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  // a soft studio environment for reflections and ambient light (like the customizer's lit room)
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;
  const camera = new THREE.PerspectiveCamera(24, 1, 0.01, 10);
  scene.add(new THREE.HemisphereLight(0xe8eeff, 0x40342c, 0.25));
  const key = new THREE.DirectionalLight(0xfff1e4, 2.6); scene.add(key); scene.add(key.target);
  key.castShadow = true; key.shadow.mapSize.set(2048, 2048); key.shadow.radius = 4;
  key.shadow.bias = -0.0004; key.shadow.normalBias = 0.003;
  const fill = new THREE.DirectionalLight(0xdde6ff, 0.45); scene.add(fill); scene.add(fill.target);
  const rim = new THREE.DirectionalLight(0xa8dcff, 1.4); scene.add(rim); scene.add(rim.target);

  const group = new THREE.Group(); group.rotation.x = -Math.PI / 2; scene.add(group); // game is Z-up
  // skin: physically based, with a soft reddish sheen standing in for light scattering under the skin
  const skinMat = new THREE.MeshPhysicalMaterial({ color: data.colours.skin, roughness: 0.52, metalness: 0,
    sheen: 0.45, sheenColor: new THREE.Color(0xff8c70), sheenRoughness: 0.55,
    clearcoat: 0.06, clearcoatRoughness: 0.45, specularIntensity: 0.55, envMapIntensity: 0.55 });
  const head = new THREE.Mesh(geometry(data), skinMat);
  head.castShadow = true; head.receiveShadow = true;
  group.add(head);
  if (data.eyes) {
    const eyes = new THREE.Mesh(geometry(data.eyes), new THREE.MeshPhysicalMaterial({ map: eyeTexture(data.colours.iris),
      roughness: 0.35, clearcoat: 1, clearcoatRoughness: 0.03, envMapIntensity: 0.6 }));  // wet cornea highlight
    eyes.receiveShadow = true; group.add(eyes);
  }

  // hair, beard and brows: cards cut out by their opacity texture, scalp caps as a soft tinted layer
  const hairGroup = new THREE.Group(); group.add(hairGroup);
  const loader = new THREE.TextureLoader();
  for (const a of data.attachments || []) {
    const geo = geometry(a);
    if (a.kind === "stubble") {
      // dense fine strand cards: drawn as a soft, see-through shadow of hair over the jaw
      const m = new THREE.MeshStandardMaterial({ color: new THREE.Color(a.colour).multiplyScalar(0.8), roughness: 0.9,
        transparent: true, opacity: 0.38, depthWrite: false, side: THREE.DoubleSide,
        polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -6 });
      const mesh = new THREE.Mesh(geo, m); mesh.renderOrder = 2; hairGroup.add(mesh);
      continue;
    }
    if (a.kind === "brows") {
      // The game places brow hairs with a card-ID shader we don't reproduce, so shade the brow shell
      // itself: opaque through the middle of its height, fading at the edges, the nose bridge and the tips.
      browAlpha(geo);
      const mats = a.parts.map((p, i) => {
        geo.addGroup(p.first, p.count, i);
        return p.cap ? new THREE.MeshBasicMaterial({ visible: false })
          : new THREE.MeshStandardMaterial({ color: a.colour, roughness: 0.7, vertexColors: true, transparent: true,
              depthWrite: false, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -8 });
      });
      const mesh = new THREE.Mesh(geo, mats); mesh.renderOrder = 2; hairGroup.add(mesh);
      continue;
    }
    const mats = a.parts.map((p, i) => {
      geo.addGroup(p.first, p.count, i);
      // polygonOffset: brows and stubble lie right on the skin, so pull them forward in depth
      // hair: anisotropic highlight along the strands, like the game's hair shader
      const m = new THREE.MeshPhysicalMaterial({ color: a.colour, roughness: 0.45, metalness: 0, side: THREE.DoubleSide,
        anisotropy: p.cap ? 0 : 0.7, sheen: 0.3, sheenColor: new THREE.Color(a.colour).lerp(new THREE.Color(0xffffff), 0.35),
        sheenRoughness: 0.4, envMapIntensity: 0.45, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 });
      // caps (scalp shadow, brow shape) have faint masks by design, so boost them;
      // strand cards are cut out with a low threshold so fine hairs survive
      if (p.cap) { m.transparent = true; m.opacity = a.kind === "hair" ? 0.5 : 0.95; m.depthWrite = false; }
      else m.alphaTest = a.kind === "hair" ? 0.25 : 0.12;
      if (p.tex) loader.load(`/api/headtex/${p.tex}`, (t) => {
        t.flipY = false; t.colorSpace = THREE.NoColorSpace;
        t.wrapS = t.wrapT = THREE.RepeatWrapping; // beard UVs run past 1 and tile
        m.alphaMap = t; m.needsUpdate = true;
      });
      return m;
    });
    const mesh = new THREE.Mesh(geo, mats);
    mesh.renderOrder = 1;
    mesh.castShadow = a.kind === "hair";   // hair shades the forehead and temples
    hairGroup.add(mesh);
  }

  // centre camera on the face
  const box = new THREE.Box3().setFromObject(group); const c = box.getCenter(new THREE.Vector3()); const sz = box.getSize(new THREE.Vector3());
  // three-point lighting around the face (the face looks down -Z): warm key high front-right,
  // cool fill front-left, bright rim from behind
  const place = (l, x, y, z) => { l.position.set(c.x + x, c.y + y, c.z + z); l.target.position.copy(c); };
  place(key, 0.45, 0.4, -0.8); place(fill, -0.8, 0.05, -0.5); place(rim, -0.5, 0.45, 0.8);
  const r = Math.max(sz.x, sz.y, sz.z) * 0.75;
  Object.assign(key.shadow.camera, { left: -r, right: r, top: r, bottom: -r, near: 0.1, far: 3 });
  key.shadow.camera.updateProjectionMatrix();
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(c); controls.enableDamping = true; controls.minDistance = 0.25; controls.maxDistance = 2;
  // the game's faces look down -Z after the Z-up -> Y-up turn, so start the camera there
  camera.position.set(c.x, c.y + sz.y * 0.05, c.z - Math.max(sz.y, sz.x) * 2.6);
  controls.autoRotate = !window.HeadView?.still; controls.autoRotateSpeed = 1.2;
  renderer.domElement.addEventListener("pointerdown", () => (controls.autoRotate = false));

  skinTexture(data.gender).then((t) => {
    const avg = avgColour(t.image); const target = new THREE.Color(data.colours.skin);
    // tint the stock skin texture towards this character's skin tone
    // (both in linear space; clamp so we never push a channel past white)
    const lin = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    const tc = new THREE.Color(Math.min(1.6, target.r / lin(avg[0])), Math.min(1.6, target.g / lin(avg[1])), Math.min(1.6, target.b / lin(avg[2])));
    head.material.map = t; head.material.color = tc; head.material.needsUpdate = true;
    return paintSkin(t.image, tc, data).then((c) => {
      if (!c || view?.head !== head) return;
      const pt = new THREE.CanvasTexture(c); pt.colorSpace = THREE.SRGBColorSpace; pt.anisotropy = 8; pt.flipY = false;
      head.material.map = pt; head.material.color = new THREE.Color(1, 1, 1); head.material.needsUpdate = true;
    });
  }).catch(() => {});
  // the game's own skin detail: normal map (pores, wrinkles, lips) and specular map (oily vs matte areas)
  const texLoader = new THREE.TextureLoader();
  texLoader.load(`/api/headtex/${data.gender}/normal`, (t) => {
    t.flipY = false; t.colorSpace = THREE.NoColorSpace; t.anisotropy = 8;
    skinMat.normalMap = t; skinMat.normalScale.set(0.9, -0.9); skinMat.needsUpdate = true;  // DirectX-style green channel
  });
  texLoader.load(`/api/headtex/${data.gender}/spec`, (t) => {
    t.flipY = false; t.colorSpace = THREE.NoColorSpace;
    skinMat.specularIntensityMap = t; skinMat.specularIntensity = 0.75; skinMat.needsUpdate = true;
  });

  const resize = () => { const w = host.clientWidth, h = host.clientHeight || w * 0.75; renderer.setSize(w, h, false); renderer.domElement.style.width = "100%"; renderer.domElement.style.height = "100%"; camera.aspect = w / h; camera.updateProjectionMatrix(); };
  const ro = new ResizeObserver(resize); ro.observe(host); resize();
  // timer rather than rAF so it keeps drawing in background panes too
  const tick = () => { controls.update(); renderer.render(scene, camera); };
  view = { renderer, scene, controls, ro, timer: setInterval(tick, 33), raf: 0, hairGroup, head, camera };
  tick();

  // recipe readout
  const byPart = new Map();
  for (const u of data.used) {
    const part = REGION_LABEL[u.region] || u.region;
    if (!byPart.has(part)) byPart.set(part, new Map());
    const m = byPart.get(part); m.set(u.head, Math.max(m.get(u.head) || 0, u.weight));
  }
  const att = data.attachments || [];
  const label = { hair: "Hair", beard: "Beard", brows: "Brows", stubble: "Stubble" };
  recipeHost.innerHTML = `<div class="hv-swatches">
      <span><i style="background:${data.colours.skin}"></i>Skin</span>
      <span><i style="background:${data.colours.iris}"></i>Eyes</span>
      ${att.map((a) => `<span title="${a.item}"><i style="background:${a.colour}"></i>${label[a.kind]}</span>`).join("")}
      ${att.length ? `<label class="hv-toggle"><input type="checkbox" id="hvHair" checked> Show hair</label>` : ""}</div>
    <table class="hv-recipe">${[...byPart].map(([part, m]) => `<tr><td>${part}</td><td>${[...m].sort((a, b) => b[1] - a[1]).map(([h, w]) => `${prettyHead(h)} <b>${Math.round(w * 100)}%</b>`).join(" · ")}</td></tr>`).join("")}</table>
    <div class="tiny muted">Preview: face shape is blended from the game's stock heads using your file's recipe, with the game's own skin textures, hair models and colours. Lighting and shaders are an approximation of the game's, so expect small differences in game.${data.missing.length ? " Skipped: " + data.missing.join(", ") + "." : ""}</div>`;
  recipeHost.querySelector("#hvHair")?.addEventListener("change", (e) => { if (view) view.hairGroup.visible = e.target.checked; });
}

window.HeadView = { show, dispose, get view() { return view; }, geometry, eyeTexture, skinTexture, avgColour };
window.dispatchEvent(new Event("headview-ready"));
