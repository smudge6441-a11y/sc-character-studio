/* Character library. Uses $, $$, esc, api, toast, lsGet/lsSet from common.js. */
"use strict";

const C = { lib: [], game: {}, shots: [], sel: null, imgBust: {} };

async function charsLoad() {
  try {
    const r = await api("/api/chars");
    C.lib = r.library; C.game = r.game; C.shots = r.screenshots;
  } catch (e) { toast("Couldn't read characters: " + e.message, true); return; }
  charsRender();
}

function fmtDate(v) {
  const d = typeof v === "number" ? new Date(v * 1000) : new Date(v);
  return isNaN(d) ? "" : d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}
function installedWhere(c) {
  const out = [];
  for (const [b, info] of Object.entries(C.game)) for (const f of info.files) if (f.libraryId === c.id) out.push(`${b}: ${f.file}`);
  return out;
}
function imgUrl(c) { return `/api/chars/${c.id}/image?v=${C.imgBust[c.id] || 0}`; }
function initials(name) { return (name || "?").split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase(); }

function charsRender() {
  // game folders
  $("#charGame").innerHTML = Object.entries(C.game).map(([b, info]) => {
    const files = info.files.map((f) => `<div class="gamefile">
        <div><b>${esc(f.file)}</b> <span class="muted tiny">${fmtDate(f.mtime)}</span><br>
        ${!f.valid ? `<span class="bad-chip">⚠ ${esc(f.problem)}</span>`
          : f.libraryId ? `<span class="ok-chip">✓ In your library as “${esc(C.lib.find((c) => c.id === f.libraryId)?.name || "")}”</span>`
          : `<span class="warn-chip">Not backed up</span>`}</div>
        ${f.valid && !f.libraryId ? `<button class="small" data-savegame="${esc(b)}|${esc(f.file)}">Save to library</button>`
          : f.libraryId ? `<button class="small ghost" data-opencard="${f.libraryId}">Open</button>` : ""}
      </div>`).join("");
    return `<div class="gamebox"><h4>${esc(b)}</h4>${files || `<div class="muted tiny">${info.exists ? "No characters saved here yet." : "No characters folder yet. Installing a character here will create it."}</div>`}</div>`;
  }).join("");

  // library grid
  const q = $("#charSearch").value.trim().toLowerCase();
  const list = C.lib.filter((c) => !q || (c.name + " " + c.notes + " " + c.body + " " + c.source).toLowerCase().includes(q));
  $("#charCount").textContent = `${C.lib.length} saved`;
  $("#charGrid").innerHTML = list.map((c) => {
    const where = installedWhere(c);
    return `<button class="charcard ${C.sel === c.id ? "sel" : ""}" data-opencard="${c.id}">
      <div class="charpic" ${c.hasImage ? `style="background-image:url('${imgUrl(c)}')"` : ""}>${c.hasImage ? "" : esc(initials(c.name))}</div>
      <div class="info"><div class="nm">${esc(c.name)}</div>
      <div class="meta">${c.body ? esc(c.body) + " · " : ""}added ${fmtDate(c.added)}</div>
      ${where.length ? `<div class="where">● In game: ${esc(where.join(", "))}</div>` : ""}</div></button>`;
  }).join("") || `<div class="muted">${C.lib.length ? "No matches." : "Your library is empty. Click “Back up all game characters” to start, or import .chf files."}</div>`;

  if (C.sel) charsRenderDrawer();
}

function charsOpen(id) {
  C.sel = id;
  const c = C.lib.find((x) => x.id === id); if (!c) return;
  $("#cdName").value = c.name; $("#cdBody").value = c.body || ""; $("#cdNotes").value = c.notes || "";
  $("#cdFileName").value = c.name.replace(/[^A-Za-z0-9 _\-\.']/g, "").trim() || "character";
  $("#cdBranch").innerHTML = Object.keys(C.game).map((b) => `<option ${b === (lsGet("studio.charBranch") || "LIVE") ? "selected" : ""}>${b}</option>`).join("");
  $("#cdResult").textContent = "";
  $("#charDrawer").hidden = false;
  charsRender();
  charsShow3D(id);
}
function charsShow3D(id) {
  const go = () => window.HeadView.show($("#cdHead"), $("#cdRecipe"), id);
  if (window.HeadView) go();
  else {
    $("#cdHead").innerHTML = `<div class="hv-status">Loading 3D viewer…</div>`;
    window.addEventListener("headview-ready", go, { once: true });
  }
}
function charsRenderDrawer() {
  const c = C.lib.find((x) => x.id === C.sel);
  if (!c) { $("#charDrawer").hidden = true; C.sel = null; return; }
  $("#cdTitle").textContent = c.name;
  $("#cdSource").textContent = `From ${c.source} · added ${fmtDate(c.added)}`;
  const im = $("#cdImage");
  im.style.backgroundImage = c.hasImage ? `url('${imgUrl(c)}')` : "";
  im.textContent = c.hasImage ? "" : "No picture yet: paste a screenshot (Ctrl+V), drop an image here, or upload one.";
  $("#cdShots").innerHTML = C.shots.length
    ? `<span class="tiny muted">From your Star Citizen screenshots:</span>` + C.shots.slice(0, 20).map((p) => `<img src="/api/shot?p=${encodeURIComponent(p)}" data-shot="${esc(p)}" loading="lazy" title="Use this picture">`).join("")
    : "";
  $("#cdDownload").href = `/api/chars/${c.id}/file`;
}
function charsClose() { $("#charDrawer").hidden = true; C.sel = null; window.HeadView?.dispose(); charsRender(); }

let charSaveT = null;
function charsSaveMeta() {
  const id = C.sel; if (!id) return;
  const fields = { name: $("#cdName").value.trim() || "Unnamed character", body: $("#cdBody").value, notes: $("#cdNotes").value };
  const c = C.lib.find((x) => x.id === id); Object.assign(c, fields);
  clearTimeout(charSaveT);
  charSaveT = setTimeout(async () => {
    try { await api(`/api/chars/${id}`, { method: "PUT", body: JSON.stringify(fields) }); charsRender(); }
    catch (e) { toast(e.message, true); }
  }, 500);
}

/* ---------- images: downscale big screenshots (e.g. 5120x1440) before storing */
function fileToDataUrl(file, maxW = 1600) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => {
      const s = Math.min(1, maxW / img.width);
      const cv = document.createElement("canvas");
      cv.width = Math.round(img.width * s); cv.height = Math.round(img.height * s);
      cv.getContext("2d").drawImage(img, 0, 0, cv.width, cv.height);
      URL.revokeObjectURL(img.src);
      res(cv.toDataURL("image/jpeg", 0.88));
    };
    img.onerror = () => rej(new Error("That file isn't an image I can read"));
    img.src = URL.createObjectURL(file);
  });
}
async function charsSetImage(body) {
  const id = C.sel; if (!id) return;
  try {
    await api(`/api/chars/${id}/image`, { method: "POST", body: JSON.stringify(body) });
    C.lib.find((x) => x.id === id).hasImage = true; C.imgBust[id] = Date.now();
    charsRender(); toast("Picture saved");
  } catch (e) { toast(e.message, true); }
}
async function charsImageFile(file) { charsSetImage({ dataUrl: await fileToDataUrl(file) }); }

/* ---------- importing .chf files */
function bufToB64(buf) {
  let s = ""; const b = new Uint8Array(buf);
  for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
  return btoa(s);
}
async function charsImportFiles(files) {
  let added = 0, dup = 0, lastId = null;
  for (const f of files) {
    if (!f.name.toLowerCase().endsWith(".chf")) { toast(`${f.name} isn't a .chf file`, true); continue; }
    try {
      const r = await api("/api/chars/import", { method: "POST", body: JSON.stringify({ filename: f.name, upload: bufToB64(await f.arrayBuffer()) }) });
      r.new ? added++ : dup++; lastId = r.id;
    } catch (e) { toast(`${f.name}: ${e.message}`, true); }
  }
  await charsLoad();
  if (added || dup) toast(`${added} added${dup ? `, ${dup} already in your library` : ""}`);
  if (lastId && files.length === 1) charsOpen(lastId);
}

async function charsInstall(overwrite = false) {
  const branch = $("#cdBranch").value, fileName = $("#cdFileName").value.trim();
  lsSet("studio.charBranch", branch);
  try {
    const r = await api(`/api/chars/${C.sel}/install`, { method: "POST", body: JSON.stringify({ branch, fileName, overwrite }) });
    if (r.conflict) {
      if (confirm(`${branch} already has a different character called "${fileName}.chf".\n\nReplace it? The current one will be saved into your library first, so nothing is lost.`)) return charsInstall(true);
      return;
    }
    $("#cdResult").innerHTML = r.unchanged ? `<span class="ok-chip">Already installed: ${esc(r.path)}</span>`
      : `<span class="ok-chip">Installed ✓ ${esc(r.path)}</span>${r.backedUp ? `<br><span class="warn-chip">The file it replaced was saved to your library first.</span>` : ""}`;
    await charsLoad();
  } catch (e) { $("#cdResult").innerHTML = `<span class="bad-chip">${esc(e.message)}</span>`; }
}

/* ---------- wiring */
document.addEventListener("click", async (e) => {
  const t = e.target.closest("[data-opencard],[data-savegame],[data-shot]");
  if (!t) return;
  if (t.dataset.opencard) charsOpen(t.dataset.opencard);
  else if (t.dataset.shot) charsSetImage({ screenshot: t.dataset.shot });
  else if (t.dataset.savegame) {
    const [branch, file] = t.dataset.savegame.split("|");
    try {
      const r = await api("/api/chars/import", { method: "POST", body: JSON.stringify({ branch, file }) });
      await charsLoad(); toast("Saved to your library"); charsOpen(r.id);
    } catch (x) { toast(x.message, true); }
  }
});
$("#charSearch").oninput = charsRender;
$("#charBackupAll").onclick = async () => {
  try { const r = await api("/api/chars/backup-all", { method: "POST", body: "{}" }); await charsLoad(); toast(r.added ? `${r.added} character${r.added > 1 ? "s" : ""} backed up` : "Everything in the game is already in your library"); }
  catch (e) { toast(e.message, true); }
};
$("#charFile").onchange = (e) => { charsImportFiles([...e.target.files]); e.target.value = ""; };
$("#cdClose").onclick = charsClose;
$("#cdName").oninput = charsSaveMeta;
$("#cdBody").onchange = charsSaveMeta;
$("#cdNotes").oninput = charsSaveMeta;
$("#cdInstall").onclick = () => charsInstall(false);
$("#cdImgFile").onchange = (e) => { if (e.target.files[0]) charsImageFile(e.target.files[0]); e.target.value = ""; };
$("#cdDelete").onclick = async () => {
  const c = C.lib.find((x) => x.id === C.sel); if (!c) return;
  const where = installedWhere(c);
  if (!confirm(`Delete "${c.name}" from your library? This can't be undone.${where.length ? `\n\nThe copy installed in the game (${where.join(", ")}) is NOT touched.` : ""}`)) return;
  try { await api(`/api/chars/${c.id}`, { method: "DELETE" }); C.sel = null; $("#charDrawer").hidden = true; await charsLoad(); toast("Deleted from library"); }
  catch (e) { toast(e.message, true); }
};
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#charDrawer").hidden) charsClose(); });
document.addEventListener("paste", (e) => {
  if (!C.sel || $("#charDrawer").hidden) return;
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) { e.preventDefault(); charsImageFile(item.getAsFile()); }
});
// drag & drop: .chf anywhere → import; image onto the drawer picture → set picture
let dragDepth = 0;
document.addEventListener("dragenter", (e) => { if ([...e.dataTransfer.types].includes("Files")) { dragDepth++; if (!e.target.closest?.("#charDrawer") && $("#photoFit").hidden) document.body.classList.add("dropping"); } });
document.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dropping"); } });
document.addEventListener("dragover", (e) => { if ([...e.dataTransfer.types].includes("Files")) e.preventDefault(); });
document.addEventListener("drop", (e) => {
  e.preventDefault(); dragDepth = 0; document.body.classList.remove("dropping");
  const files = [...e.dataTransfer.files];
  const img = files.find((f) => f.type.startsWith("image/"));
  if (img && C.sel && e.target.closest?.("#charDrawer")) return charsImageFile(img);
  const chf = files.filter((f) => f.name.toLowerCase().endsWith(".chf"));
  if (chf.length) charsImportFiles(chf);
});

/* ---------- first run: make sure we know where the game is */
async function boot() {
  let s;
  try { s = await api("/api/setup"); } catch (e) { toast("Can't reach the local server: " + e.message, true); return; }
  if (s.missingPackages?.length) toast("Missing Python packages: " + s.missingPackages.join(", ") + ". Install them with pip, or use the SC-Character-Studio.exe release, which includes everything.", true);
  $("#setup").hidden = s.found;
  $("#tab-chars").hidden = !s.found;
  if (s.found) charsLoad();
}
$("#setupSave").onclick = async () => {
  try {
    await api("/api/setup", { method: "POST", body: JSON.stringify({ scRoot: $("#setupPath").value }) });
    toast("Found it"); boot();
  } catch (e) { $("#setupMsg").innerHTML = `<span class="bad-chip">${esc(e.message)}</span>`; }
};
boot();
