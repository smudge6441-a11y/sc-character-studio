// "Create from photo": detect the face with MediaPipe (runs locally in the browser; only the model
// file is downloaded), fit a Star Citizen face recipe on the server, preview it, save to the library.
import * as THREE from "three";
import { FaceLandmarker, ImageSegmenter, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const $ = (s) => document.querySelector(s);
const st = { img: null, photoUrl: null, landmarks: null, result: null, lm: null };

async function landmarker() {
  if (!st.lm) {
    const files = await FilesetResolver.forVisionTasks(WASM);
    st.lm = await FaceLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: MODEL, delegate: "CPU" }, runningMode: "IMAGE", numFaces: 1,
    });
  }
  return st.lm;
}
// hair / skin / clothes segmentation (0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other)
const SEG_MODEL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";
async function segmenter() {
  if (!st.seg) {
    const files = await FilesetResolver.forVisionTasks(WASM);
    st.seg = await ImageSegmenter.createFromOptions(files, {
      baseOptions: { modelAssetPath: SEG_MODEL, delegate: "CPU" }, runningMode: "IMAGE",
      outputCategoryMask: true, outputConfidenceMasks: false,
    });
  }
  return st.seg;
}
/** Segmentation of the head area: {mw, mh, w, ox, oy, mask: base64 category bytes}. */
async function segmentHead(img, lms) {
  // crop to the head (with room for long hair) so the segmenter sees it at full resolution
  const xs = lms.map((p) => p[0]), ys = lms.map((p) => p[1]);
  const fx0 = Math.min(...xs), fx1 = Math.max(...xs), fy0 = Math.min(...ys), fy1 = Math.max(...ys);
  const fw = fx1 - fx0, fh = fy1 - fy0, side = Math.max(fw * 2.4, fh * 2.3);
  const ox = Math.max(0, Math.round((fx0 + fx1) / 2 - side / 2)), oy = Math.max(0, Math.round(fy0 - fh * 0.85));
  const cw = Math.min(img.naturalWidth - ox, Math.round(side)), ch = Math.min(img.naturalHeight - oy, Math.round(side));
  const k = Math.min(1, 512 / Math.max(cw, ch));
  const c = document.createElement("canvas"); c.width = Math.max(1, Math.round(cw * k)); c.height = Math.max(1, Math.round(ch * k));
  c.getContext("2d").drawImage(img, ox, oy, cw, ch, 0, 0, c.width, c.height);
  const res = (await segmenter()).segment(c);
  const cat = res.categoryMask, bytes = cat.getAsUint8Array().slice();
  cat.close?.();
  let bin = ""; for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return { mw: c.width, mh: c.height, w: cw, ox, oy, mask: btoa(bin) };
}
/** Ask the server which hair styles best match the photo; show them and pick the best one. */
async function suggestHair(gender) {
  const box = $("#pfHairSuggest");
  if (!st.seg_res) return;
  status("Finding the hair styles closest to your photo (the first time takes about a minute)…");
  const r = await api("/api/hairmatch", Object.assign({ gender, landmarks: st.landmarks }, st.seg_res));
  const opts = [...$("#pfHairStyle").options].map((o) => o.value);
  const list = r.styles.filter((s) => opts.includes(s.name)).slice(0, 5);
  st.suggested = gender;
  if (!list.length) return;
  box.innerHTML = `<span>Closest matches:</span>` + list.map((s, i) =>
    `<button class="small ${i ? "ghost" : ""}" data-hair="${s.name}">${i ? "" : "★ "}${pretty(s.name)}</button>`).join("");
  box.hidden = false;
  box.querySelectorAll("[data-hair]").forEach((b) => b.onclick = (e) => {
    e.preventDefault(); $("#pfHairStyle").value = b.dataset.hair; markSuggest(); restyle();
  });
  if (!$("#pfHairStyle").value) $("#pfHairStyle").value = list[0].name;  // never override a manual choice
  markSuggest();
}
function markSuggest() {
  $("#pfHairSuggest").querySelectorAll("[data-hair]").forEach((x) => x.classList.toggle("ghost", x.dataset.hair !== $("#pfHairStyle").value));
}
async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const j = await r.json().catch(() => ({}));
  if (!r.ok || j.error) throw Object.assign(new Error(j.error || r.statusText), { status: r.status });
  return j;
}
function status(msg, bad) { const s = $("#pfStatus"); s.textContent = msg; s.className = "pf-status" + (bad ? " bad" : ""); }

/* ---------- one-off: map MediaPipe's 478 landmarks onto the game's base head topology ---------- */
async function landmarkMap(gender) {
  try { return (await api(`/api/landmarkmap/${gender}`)).map; } catch (e) { if (e.status !== 404) throw e; }
  status(`First run for ${gender} heads: mapping face landmarks onto the game's head (one-off)…`);
  // render the plain base head in the normal viewer (front-on, still), then detect on that image
  const data = Object.assign(await api(`/api/basehead/${gender}`), { attachments: [], used: [], missing: [], items: [] });
  const wasStill = HeadView.still; HeadView.still = true;
  await HeadView.show($("#pfHead"), $("#pfRecipe"), data);
  HeadView.still = wasStill;
  const v = HeadView.view;
  await new Promise((r) => setTimeout(r, 1500)); // let the skin texture arrive
  v.renderer.setSize(800, 800, false); v.camera.aspect = 1; v.camera.updateProjectionMatrix();
  v.controls.update(); v.renderer.render(v.scene, v.camera);
  const cam = v.camera, head = v.head;
  const res = (await landmarker()).detect(v.renderer.domElement);
  if (!res.faceLandmarks?.length) throw new Error("Couldn't find landmarks on the game's base head render");
  const ray = new THREE.Raycaster(); const tri = new THREE.Triangle(); const bary = new THREE.Vector3();
  const pos = head.geometry.attributes.position;
  const A = new THREE.Vector3(), B = new THREE.Vector3(), C = new THREE.Vector3();
  const map = res.faceLandmarks[0].map((p) => {
    ray.setFromCamera(new THREE.Vector2(p.x * 2 - 1, -(p.y * 2 - 1)), cam);
    const hit = ray.intersectObject(head, false)[0];
    if (!hit) return null;
    const { a, b, c: cc } = hit.face;
    const local = head.worldToLocal(hit.point.clone());
    A.fromBufferAttribute(pos, a); B.fromBufferAttribute(pos, b); C.fromBufferAttribute(pos, cc);
    tri.set(A, B, C).getBarycoord(local, bary);
    return { tri: [a, b, cc], bary: [bary.x, bary.y, bary.z] };
  });
  HeadView.dispose();
  await api(`/api/landmarkmap/${gender}`, { map });
  return map;
}

/* ---------- photo: landmarks + colour sampling ---------- */
function sample(ctx, x, y, r) {
  const d = ctx.getImageData(Math.max(0, x - r), Math.max(0, y - r), r * 2, r * 2).data;
  const px = [];
  for (let i = 0; i < d.length; i += 4) px.push([d[i], d[i + 1], d[i + 2]]);
  px.sort((p, q) => (p[0] + p[1] + p[2]) - (q[0] + q[1] + q[2]));
  const mid = px.slice(Math.floor(px.length * 0.2), Math.ceil(px.length * 0.8)); // trim shadows & highlights
  const avg = [0, 1, 2].map((k) => mid.reduce((s, p) => s + p[k], 0) / Math.max(1, mid.length));
  return avg.map(Math.round);
}
const hex = (c) => "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
/* colour maths for makeup: sRGB <-> linear, CIE Lab, and "what pigment over this skin gives that colour" */
const toLin = (c) => c.map((v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
const toSrgb = (l) => l.map((v) => { v = Math.min(1, Math.max(0, v)); return Math.round(255 * (v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055)); });
function lab(c) {
  const [r, g, b] = toLin(c), f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const x = f((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047), y = f(0.2126 * r + 0.7152 * g + 0.0722 * b), z = f((0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883);
  return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}
const dE = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
function labToRgb([L, A, B]) {
  const fy = (L + 16) / 116, fx = fy + A / 500, fz = fy - B / 200, inv = (t) => (t ** 3 > 0.008856 ? t ** 3 : (t - 16 / 116) / 7.787);
  const x = inv(fx) * 0.95047, y = inv(fy), z = inv(fz) * 1.08883;
  return toSrgb([3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z, 0.0557 * x - 0.2040 * y + 1.0570 * z]);
}
/** the makeup's own colour: the observed colour pushed further from the skin (makeup is seen
 * blended with skin), in Lab so it stays a believable shade rather than a pure primary */
const pigment = (obs, skin, push = 1.6) => { const o = lab(obs), s = lab(skin); return labToRgb(o.map((v, i) => s[i] + (v - s[i]) * push)); };
const chroma = (l) => Math.hypot(l[1], l[2]);
const median = (cs) => [0, 1, 2].map((k) => { const v = cs.map((c) => c[k]).sort((p, q) => p - q); return Math.round(v[v.length >> 1]); });
const mean = (cs) => [0, 1, 2].map((k) => Math.round(cs.reduce((s, c) => s + c[k], 0) / cs.length));

/** Read makeup from the photo: lipstick, eye makeup (upper lids) and blush (cheek apples), each
 * judged against the person's own skin so natural colouring isn't mistaken for makeup.
 * Returns {lips|eyes|cheeks: {colour: [r,g,b], opacity: 0..1}} plus the raw samples. */
function readMakeup(skin, lips, lids, cheeks) {
  const S = lab(skin), out = {};
  // lips: natural lips are a little darker and redder than the skin around them
  const Lp = lab(lips), naturalLips = [S[0] - 10, S[1] + 14, S[2] + 4], dl = dE(Lp, naturalLips);
  // a smooth ramp: natural lips get a hint of their own colour, clear lipstick gets more
  out.lips = dl > 13 ? { colour: pigment(lips, skin), opacity: clamp(0.12 + (dl - 13) / 25, 0.12, 0.9) }
                     : { colour: lips, opacity: 0.12 };
  // upper lids: shadow makes them darker but not more colourful. Eye makeup = clearly coloured
  // (eyeshadow), or very much darker (liner / smoky eyes)
  // (lids are naturally a little redder and less yellow than the forehead: up to +10 redness and
  // -12 yellowness doesn't count)
  const E = lab(lids), da0 = E[1] - S[1], db0 = E[2] - S[2], dDark = S[0] - E[0];
  const dChroma = Math.hypot(db0 > 0 ? db0 : Math.min(0, db0 + 12), da0 < 0 ? da0 : Math.max(0, da0 - 10));
  const eyeScore = Math.max(dChroma - 9, (dDark - 28) * 0.8);
  out.eyes = eyeScore > 0 ? { colour: pigment(lids, skin, 1.4), opacity: clamp(0.25 + eyeScore / 25, 0.25, 0.6) }
                          : { colour: null, opacity: 0 };
  // blush: cheeks noticeably redder than the reference skin
  const da = lab(cheeks)[1] - S[1];
  out.cheeks = da > 5 ? { colour: pigment(cheeks, skin, 2.0), opacity: clamp((da - 3) / 20, 0.15, 0.6) }
                      : { colour: null, opacity: 0 };
  return out;
}
/* ---------- skin tone strip: exactly the game's skin tones (light -> dark), click to choose ---------- */
async function skinStrip() {
  if (st.skinPal) return st.skinPal;
  const { png } = await api("/api/palette/skin");
  const im = new Image();
  await new Promise((res, rej) => { im.onload = res; im.onerror = rej; im.src = png; });
  const full = document.createElement("canvas"); full.width = im.width; full.height = im.height;
  const fctx = full.getContext("2d", { willReadFrequently: true }); fctx.drawImage(im, 0, 0);
  const cv = $("#pfSkinCanvas"); cv.getContext("2d").drawImage(im, 0, 0, cv.width, cv.height);
  const data = fctx.getImageData(0, 0, im.width, im.height).data, labs = [];
  for (let y = 0; y < im.height; y += 4) for (let x = 0; x < im.width; x += 2) {
    const i = (y * im.width + x) * 4; labs.push([x, y, lab([data[i], data[i + 1], data[i + 2]])]);
  }
  st.skinPal = { w: im.width, h: im.height, data, labs };
  return st.skinPal;
}
function markSkin() {
  const p = st.skinPal; if (!p) return;
  const t = lab(rgb($("#pfSkin").value));
  let best = null;
  for (const [x, y, l] of p.labs) { const d = dE(l, t); if (!best || d < best[0]) best = [d, x, y]; }
  $("#pfSkinMark").style.left = (100 * best[1] / (p.w - 1)) + "%";
}
$("#pfSkinStrip").addEventListener("click", (e) => {
  const p = st.skinPal; if (!p) return;
  const r = e.currentTarget.getBoundingClientRect();
  const x = Math.round(clamp((e.clientX - r.left) / r.width, 0, 1) * (p.w - 1)), y = Math.round(clamp((e.clientY - r.top) / r.height, 0, 1) * (p.h - 1));
  const i = (y * p.w + x) * 4;
  $("#pfSkin").value = hex([p.data[i], p.data[i + 1], p.data[i + 2]]);
  markSkin(); restyle();
});
$("#pfSkin").addEventListener("input", markSkin);

/* ---------- freckles: small spots darker than the skin right around them ---------- */
const FACE_PX = 360;   // the face is rescaled to this width so spot sizes mean the same in every photo
function detectFreckles(img, L) {
  const W = img.naturalWidth, H = img.naturalHeight;
  const faceW = Math.hypot(L[454][0] - L[234][0], L[454][1] - L[234][1]);
  if (faceW < 140) return { tooSmall: true };          // freckles are a pixel or two: can't see them
  const k = FACE_PX / faceW;
  const xs = L.map((p) => p[0]), ys = L.map((p) => p[1]);
  const x0 = Math.max(0, Math.min(...xs)), y0 = Math.max(0, Math.min(...ys));
  const cw = Math.min(W, Math.max(...xs)) - x0, ch = Math.min(H, Math.max(...ys)) - y0;
  const c = document.createElement("canvas"); c.width = Math.round(cw * k); c.height = Math.round(ch * k);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, x0, y0, cw, ch, 0, 0, c.width, c.height);
  const P = (i) => [(L[i][0] - x0) * k, (L[i][1] - y0) * k];
  // cheeks below the eyes, upper cheeks and the nose bridge (not the lower cheeks: stubble)
  const centres = [50, 280, 117, 346, 101, 330, 197, 6].map(P);
  const half = Math.round(FACE_PX * 0.055), blurR = 5, T = 4.5;  // T: L* units darker than surroundings
  let spots = 0, total = 0, depth = 0;
  for (const [cx, cy] of centres) {
    const sx = Math.round(cx - half - blurR), sy = Math.round(cy - half - blurR), n = 2 * (half + blurR);
    if (sx < 0 || sy < 0 || sx + n > c.width || sy + n > c.height) continue;
    const d = ctx.getImageData(sx, sy, n, n).data, Lv = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) Lv[i] = lab([d[i * 4], d[i * 4 + 1], d[i * 4 + 2]])[0];
    // integral image for the local average
    const S = new Float64Array((n + 1) * (n + 1));
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) S[(y + 1) * (n + 1) + x + 1] = Lv[y * n + x] + S[y * (n + 1) + x + 1] + S[(y + 1) * (n + 1) + x] - S[y * (n + 1) + x];
    const box = (x, y) => { const a = x - blurR, b = y - blurR, e = x + blurR + 1, f = y + blurR + 1;
      return (S[f * (n + 1) + e] - S[b * (n + 1) + e] - S[f * (n + 1) + a] + S[b * (n + 1) + a]) / ((2 * blurR + 1) ** 2); };
    for (let y = blurR; y < n - blurR; y++) for (let x = blurR; x < n - blurR; x++) {
      const diff = Lv[y * n + x] - box(x, y);
      total++;
      if (diff < -T && diff > -25) { spots++; depth -= diff; }   // much darker = hair / brow / nostril, not a freckle
    }
  }
  if (!total) return { tooSmall: true };
  const density = spots / total, contrast = spots ? depth / spots : 0;
  return { density, contrast };
}
/** freckle measurements -> the customizer's FrecklesAmount / FrecklesOpacity */
function frecklesToGame(f) {
  if (!f || f.tooSmall) return null;
  // clear skin measures ~0.02% spot pixels, moderate freckles ~1.5%, heavy freckles 3%+
  const amount = clamp((f.density - 0.002) / 0.03, 0, 1);
  const opacity = amount > 0 ? clamp(0.25 + (f.contrast - T_FRECKLE_MIN) / 6, 0.2, 0.8) * Math.min(1, amount * 2.5) : 0.05;
  return { FrecklesAmount: +amount.toFixed(2), FrecklesOpacity: +opacity.toFixed(2) };
}
const T_FRECKLE_MIN = 4.5;
function setFreckleControls(v) {
  if (!v) return;
  $("#pfFreckAmt").value = Math.round(v.FrecklesAmount * 100); $("#pfFreckOp").value = Math.round(v.FrecklesOpacity * 100);
}
const frecklesFromControls = () => ({ FrecklesAmount: Number($("#pfFreckAmt").value) / 100, FrecklesOpacity: Number($("#pfFreckOp").value) / 100 });

function setMakeupControls(mk) {
  const set = (c, amt, v) => { if (v.colour) $(c).value = hex(v.colour); $(amt).value = Math.round(v.opacity * 100); };
  set("#pfLips", "#pfLipsAmt", mk.lips); set("#pfEyeMk", "#pfEyeMkAmt", mk.eyes); set("#pfBlush", "#pfBlushAmt", mk.cheeks);
}
function makeupFromControls() {
  const v = (c, amt) => ({ colour: rgb($(c).value), opacity: Number($(amt).value) / 100 });
  return { lips: v("#pfLips", "#pfLipsAmt"), eyes: v("#pfEyeMk", "#pfEyeMkAmt"), cheeks: v("#pfBlush", "#pfBlushAmt") };
}

async function loadPhoto(file) {
  if (!file || !file.type.startsWith("image/")) return;
  st.photoUrl = URL.createObjectURL(file);
  const img = new Image();
  await new Promise((res, rej) => { img.onload = res; img.onerror = () => rej(new Error("That file isn't an image I can read")); img.src = st.photoUrl; });
  st.img = img;
  status("Finding the face…");
  const res = (await landmarker()).detect(img);
  const cv = $("#pfCanvas"); const W = img.naturalWidth, H = img.naturalHeight;
  const scale = Math.min(1, 900 / Math.max(W, H));
  cv.width = Math.round(W * scale); cv.height = Math.round(H * scale);
  const ctx = cv.getContext("2d", { willReadFrequently: true }); ctx.drawImage(img, 0, 0, cv.width, cv.height);
  if (!res.faceLandmarks?.length) { st.landmarks = null; status("No face found. Try a clear, front-facing photo with even lighting.", true); updateButtons(); return; }
  const L = res.faceLandmarks[0];
  st.landmarks = L.map((p) => [p.x * W, p.y * H]);
  st.suggested = null; $("#pfHairSuggest").hidden = true; $("#pfHairStyle").value = "";
  try { st.seg_res = await segmentHead(img, st.landmarks); } catch (e) { st.seg_res = null; console.warn("hair segmentation failed", e); }
  // colours (sampled on the displayed canvas)
  const P = (i) => [L[i].x * cv.width, L[i].y * cv.height];
  const faceH = Math.hypot(P(10)[0] - P(152)[0], P(10)[1] - P(152)[1]);
  const r = Math.max(3, Math.round(faceH * 0.03));
  // white balance: if the background is a bright, nearly neutral wall, use it to cancel a warm/cool light cast
  const bgs = [[0.06, 0.04], [0.94, 0.04], [0.5, 0.02]].map(([fx, fy]) => sample(ctx, Math.round(cv.width * fx), Math.max(r, Math.round(cv.height * fy)), r));
  const bgAvg = [0, 1, 2].map((k) => bgs.reduce((s, c) => s + c[k], 0) / bgs.length);
  const bgMax = Math.max(...bgAvg), bgMin = Math.min(...bgAvg), bgMean = (bgAvg[0] + bgAvg[1] + bgAvg[2]) / 3;
  const neutralWall = bgMean > 150 && (bgMax - bgMin) / bgMax < 0.25;
  const gain = neutralWall ? bgAvg.map((v) => Math.min(1.3, Math.max(0.8, bgMean / v))) : [1, 1, 1];
  const wb = (c) => c.map((v, k) => Math.min(255, Math.round(v * gain[k])));
  st.whiteBalance = neutralWall ? gain.map((g) => +g.toFixed(2)) : null;
  // skin: forehead, under-eye cheeks and outer cheeks; the median ignores a spot with blush,
  // shine or shadow (the cheek apples alone would read blush as skin tone)
  const skin = [151, 337, 117, 346, 50, 280].map((i) => sample(ctx, ...P(i).map(Math.round), r));
  const skinRef = wb(median(skin));
  $("#pfSkin").value = hex(skinRef);
  skinStrip().then(markSkin).catch(() => {});
  // a very bright or very dark photo shifts every skin tone: say so next to the strip
  const faceY = toLin(skinRef).reduce((s, v, k) => s + v * [0.2126, 0.7152, 0.0722][k], 0);
  $("#pfSkinNote").textContent = bgMean > 200 && faceY > 0.3 ? "Bright photo: skin may read lighter than it is. Click a darker tone if needed."
    : bgMean < 60 && faceY < 0.08 ? "Dark photo: skin may read darker than it is. Click a lighter tone if needed."
    : "Click the strip to pick a lighter or darker tone.";
  // makeup: lips (middle of each lip), upper lids (a third of the way from lid to brow), cheek apples
  const at = (i, j, t) => [P(i)[0] + (P(j)[0] - P(i)[0]) * t, P(i)[1] + (P(j)[1] - P(i)[1]) * t].map(Math.round);
  const rs = Math.max(2, Math.round(r * 0.5));
  const lipC = wb(mean([[0, 13], [37, 82], [267, 312], [17, 14], [84, 87], [314, 317]].map(([i, j]) => sample(ctx, ...at(i, j, 0.5), rs))));
  const lidC = wb(mean([[159, 52], [158, 53], [386, 282], [385, 283]].map(([i, j]) => sample(ctx, ...at(i, j, 0.33), rs))));
  const cheekC = wb(mean([205, 425, 50, 280].map((i) => sample(ctx, ...P(i).map(Math.round), r))));
  st.makeup = readMakeup(skinRef, lipC, lidC, cheekC);
  setMakeupControls(st.makeup);
  st.freckles = detectFreckles(img, st.landmarks);
  const fg = frecklesToGame(st.freckles);
  setFreckleControls(fg || { FrecklesAmount: 0.5, FrecklesOpacity: 0.25 });   // can't tell: the game's default
  $("#pfFreckNote").textContent = st.freckles.tooSmall ? "face too small in this photo to see freckles: game default used, set them by hand"
    : fg.FrecklesAmount > 0 ? "read from the photo" : "none seen in the photo";
  if (L.length > 473) {
    // iris colour from a ring around the centre, skipping the pupil and the eyelid edges
    const ring = [];
    for (const [c, edge] of [[468, 469], [473, 474]]) {
      const [cx, cy] = P(c); const R = Math.hypot(P(edge)[0] - cx, P(edge)[1] - cy);
      for (let a = 0; a < Math.PI * 2; a += Math.PI / 8) {
        if (Math.abs(Math.sin(a)) > 0.75) continue; // top and bottom are often under the lids
        ring.push(sample(ctx, Math.round(cx + Math.cos(a) * R * 0.65), Math.round(cy + Math.sin(a) * R * 0.65), Math.max(1, Math.round(R * 0.12))));
      }
    }
    $("#pfIris").value = hex(wb([0, 1, 2].map((k) => Math.round(ring.reduce((s, c) => s + c[k], 0) / ring.length))));
  }
  // hair: try spots above the forehead and at the temples; keep the one least like skin and background
  const skinC = [0, 1, 2].map((k) => skin.reduce((s, c) => s + c[k], 0) / skin.length);
  const bg = bgs[0];
  const faceW = Math.hypot(P(234)[0] - P(454)[0], P(234)[1] - P(454)[1]);
  const cands = [];
  for (const f of [0.06, 0.12, 0.18, 0.24]) cands.push([P(10)[0], P(10)[1] - faceH * f]);
  for (const [i, dir] of [[162, -1], [389, 1], [54, -1], [284, 1]]) cands.push([P(i)[0] + dir * faceW * 0.05, P(i)[1] - faceH * 0.05]);
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  let best = null;
  for (const [x, y] of cands) {
    if (x < r || y < r || x > cv.width - r || y > cv.height - r) continue;
    const c = sample(ctx, Math.round(x), Math.round(y), r);
    const score = Math.min(dist(c, skinC), dist(c, bg));
    if (!best || score > best[0]) best = [score, c];
  }
  if (best) $("#pfHair").value = hex(wb(best[1]));
  // overlay landmarks
  ctx.fillStyle = "rgba(56,198,232,0.9)";
  for (const p of L) { ctx.beginPath(); ctx.arc(p.x * cv.width, p.y * cv.height, Math.max(1, cv.width / 500), 0, Math.PI * 2); ctx.fill(); }
  const yaw = Math.abs((P(234)[0] + P(454)[0]) / 2 - P(1)[0]) / Math.abs(P(454)[0] - P(234)[0] || 1);
  status(yaw > 0.12 ? "Face found, but it's turned. A straight-on photo gives a much better match." : "Face found. Check the colours, then press Generate.", yaw > 0.12);
  updateButtons();
}

async function generate() {
  const gender = $("#pfGender").value;
  $("#pfGenerate").disabled = true;
  try {
    await st.stylesReady;  // style pickers must be filled before we read them
    await landmarkMap(gender);
    if (st.suggested !== gender) {
      try { await suggestHair(gender); } catch (e) { console.warn("hair suggestion failed", e); }
    }
    status("Matching your face to the game's head shapes…");
    const tpl = $("#pfTemplate").value;
    const r = await api("/api/facefit", {
      gender, template: tpl, landmarks: st.landmarks,
      skin: rgb($("#pfSkin").value), iris: rgb($("#pfIris").value), hair: rgb($("#pfHair").value),
      styles: { hair: $("#pfHairStyle").value, brows: $("#pfBrowStyle").value, beard: $("#pfBeardStyle").value },
      makeup: gender === "female" ? makeupFromControls() : undefined,
      skinDetail: frecklesFromControls(),
    });
    if (r.gameColours?.skin) { $("#pfSkin").value = r.gameColours.skin; markSkin(); }  // the tone the game will use
    if (r.makeup) {  // show the palette colours the game will actually use
      setMakeupControls({ lips: mk(r.makeup.lips), eyes: mk(r.makeup.eyes), cheeks: mk(r.makeup.cheeks) });
    }
    $("#pfMakeupNote").textContent = gender !== "female" ? "" : r.makeup ? "read from the photo; 0 = none"
      : "this base character has no makeup to change";
    st.result = r;
    await HeadView.show($("#pfHead"), $("#pfRecipe"), r.head);
    const fits = Object.values(r.detail).filter((d) => d.before !== undefined);
    const gain = fits.length ? Math.round(100 * (1 - fits.reduce((s, d) => s + d.after, 0) / fits.reduce((s, d) => s + d.before, 0))) : 0;
    const ex = r.expression || {};
    const note = ex.smiling || ex.mouthOpen
      ? ` You're ${ex.smiling ? "smiling" : "open-mouthed"} in this photo, so the mouth${ex.mouthOpen ? " and chin" : ""} stayed neutral rather than being read as bigger lips. A photo with a relaxed, closed mouth gives the best match.`
      : "";
    status(`Done. The blended head is ${gain}% closer to your photo's landmarks than the base head.${note} Adjust colours and regenerate, or save.`, !!note);
  } catch (e) { status(e.message, true); }
  $("#pfGenerate").disabled = false; updateButtons();
}

async function save() {
  if (!st.result) return;
  const c = document.createElement("canvas"); const s = Math.min(1, 1200 / Math.max(st.img.naturalWidth, st.img.naturalHeight));
  c.width = st.img.naturalWidth * s; c.height = st.img.naturalHeight * s; c.getContext("2d").drawImage(st.img, 0, 0, c.width, c.height);
  try {
    const r = await api("/api/facefit/save", { chf: st.result.chf, name: $("#pfName").value.trim() || "Photo character", photo: c.toDataURL("image/jpeg", 0.88) });
    close();
    await charsLoad(); charsOpen(r.id);
    toast("Saved to your library. Install it from the character panel.");
  } catch (e) { status(e.message, true); }
}

function updateButtons() {
  $("#pfGenerate").disabled = !st.landmarks || $("#pfHairStyle").options.length === 0;
  $("#pfSave").disabled = !st.result;
}
const pretty = (n) => n.replace(/^facial_hair_0*/, "Beard ").replace(/^hair_0*/, "Hair ").replace(/^brows_0*/, "Brows ").replace(/_/g, " ");
async function fillStyles() {
  const g = $("#pfGender").value;
  let s;
  try { s = await api(`/api/styles/${g}`); } catch (e) { status("Couldn't list hair styles: " + e.message, true); return; }
  const fill = (sel, list) => {
    const keep = $(sel).value;
    $(sel).innerHTML = `<option value="">Same as base character</option>` + list.map((n) => `<option value="${n}">${pretty(n)}</option>`).join("");
    if (list.includes(keep)) $(sel).value = keep;
  };
  st.suggested = null; $("#pfHairSuggest").hidden = true;
  fill("#pfHairStyle", s.hair); fill("#pfBrowStyle", s.brows); fill("#pfBeardStyle", s.beard);
  // the customizer's own "stubble only" look (most NPC beards render much fuller in game)
  $("#pfBeardStyle").options[0].insertAdjacentHTML("afterend", `<option value="__stubble__">Stubble only (like the customizer)</option>`);
  updateButtons();
}
let restyleT = null;
function restyle() {  // re-run the fit with the new style so the preview and file match
  markSuggest();
  if (!st.result) return;
  clearTimeout(restyleT); restyleT = setTimeout(generate, 250);
}
const mk = (v) => (v ? { colour: rgb(v.colour), opacity: v.opacity } : { colour: null, opacity: 0 });
function fillTemplates() {
  st.stylesReady = fillStyles();
  const g = $("#pfGender").value;
  $("#pfMakeupRow").hidden = g !== "female";  // the game only stores makeup on female characters
  const opts = [`<option value="default">Game default ${g}</option>`].concat(
    (typeof C !== "undefined" ? C.lib : []).map((c) => `<option value="${c.id}">${c.name.replace(/[<&]/g, "")}</option>`));
  $("#pfTemplate").innerHTML = opts.join("");
}
function open() {
  $("#photoFit").hidden = false; fillTemplates(); updateButtons();
  status("Choose a clear, front-facing photo: drop it here, paste it, or pick a file.");
}
function close() { $("#photoFit").hidden = true; HeadView.dispose(); }

$("#pfOpen").onclick = open;
$("#pfClose").onclick = close;
$("#pfFile").onchange = (e) => loadPhoto(e.target.files[0]).catch((x) => status(x.message, true));
$("#pfGender").onchange = fillTemplates;
$("#pfGenerate").onclick = generate;
for (const id of ["#pfHairStyle", "#pfBrowStyle", "#pfBeardStyle", "#pfLips", "#pfLipsAmt", "#pfEyeMk", "#pfEyeMkAmt", "#pfBlush", "#pfBlushAmt", "#pfFreckAmt", "#pfFreckOp"]) $(id).onchange = restyle;
document.querySelectorAll("[data-step]").forEach((b) => b.addEventListener("click", (e) => {
  e.preventDefault();
  const [id, d] = b.dataset.step.split(":"); const sel = document.getElementById(id);
  const n = sel.options.length; if (!n) return;
  sel.selectedIndex = (sel.selectedIndex + Number(d) + n) % n; restyle();
}));
$("#pfSave").onclick = save;
$("#pfDrop").addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); });
$("#pfDrop").addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); loadPhoto(e.dataTransfer.files[0]).catch((x) => status(x.message, true)); });
document.addEventListener("paste", (e) => {
  if ($("#photoFit").hidden) return;
  const it = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (it) { e.preventDefault(); e.stopImmediatePropagation(); loadPhoto(it.getAsFile()).catch((x) => status(x.message, true)); }
}, true);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#photoFit").hidden) close(); });
window.PhotoFit = { open, loadPhoto, generate, state: st };
