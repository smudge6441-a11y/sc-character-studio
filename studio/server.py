"""SC Character Studio - local server.

Serves the web app on http://localhost:8752 and gives it a small API:
  /api/chars...           character library (list, import, install, pictures, 3D head)
  /api/headtex/...        decoded skin / hair textures (cached in data/textures)
  /api/basehead, /api/landmarkmap, /api/facefit...   create a character from a photo
Reads Data.p4k read-only. Only writes to its own data folder (see studio/paths.py) and the
game's CustomCharacters folders (and backs up any character file before replacing it).
Started by "SC Character Studio.py" (desktop window, or --serve for a web browser).
"""
import base64
import json
try:
    import numpy as np
except ImportError:  # only needed for the 3D head preview
    np = None
import os
import re
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from studio import gamedata, characters, paths

PORT = int(os.environ.get("PORT", 8752))
WEB = paths.WEB
CACHE = paths.CACHE
LIB = characters.Library(paths.LIBRARY)
_HEADS = {}


def head_builder(branch="LIVE"):
    """Lazily created: loading head meshes needs numpy and reads the game archive."""
    if branch not in _HEADS:
        from studio import heads
        _HEADS[branch] = heads.HeadBuilder(branch)
    return _HEADS[branch]
CHAR_ID = r"([0-9a-f]{12})"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB, **kw)

    def log_message(self, fmt, *args):
        if "/api/" in (str(args[0]) if args else ""):
            sys.stderr.write("%s\n" % (fmt % args))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ---------------------------------------------------------------- helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, mime, download=None):
        with open(path, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % download)
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    # ---------------------------------------------------------------- routes
    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/setup":
                return self._json(setup_info())
            if u.path.startswith(("/api/basehead/", "/api/landmarkmap/", "/api/styles/", "/api/palette/")):
                r = facefit_get(u.path)
                return self._json(r) if r is not None else self._json({"error": "not built yet"}, 404)
            if u.path.startswith(("/api/chars", "/api/headtex")) or u.path == "/api/shot":
                return self._chars_get(u)
        except Exception as e:  # surface errors to the UI
            return self._json({"error": str(e)}, 500)
        return super().do_GET()

    def do_PUT(self):
        u = urlparse(self.path)
        try:
            m = re.match(r"^/api/chars/%s$" % CHAR_ID, u.path)
            if m:
                return self._json(LIB.update(m.group(1), self._body()))
        except Exception as e:
            return self._json({"error": str(e)}, 400)
        self._json({"error": "not found"}, 404)

    def do_DELETE(self):
        u = urlparse(self.path)
        try:
            m = re.match(r"^/api/chars/%s$" % CHAR_ID, u.path)
            if m:
                LIB.delete(m.group(1))
                return self._json({"ok": True})
        except Exception as e:
            return self._json({"error": str(e)}, 400)
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/setup":
            try:
                gamedata.set_sc_root(self._body().get("scRoot", ""))
                _HEADS.clear()  # rebuild head data from the new install
                return self._json(setup_info())
            except Exception as e:
                return self._json({"error": str(e)}, 400)
        if u.path.startswith(("/api/facefit", "/api/landmarkmap/", "/api/hairmatch")):
            try:
                return self._json(facefit_post(u.path, self._body()))
            except Exception as e:
                return self._json({"error": str(e)}, 400)
        if u.path.startswith("/api/chars"):
            try:
                return self._json(self._chars_post(u.path, self._body()))
            except Exception as e:
                return self._json({"error": str(e)}, 400)
        return self._json({"error": "not found"}, 404)

    # ---------------------------------------------------------------- characters
    def _chars_get(self, u):
        if u.path == "/api/chars":
            return self._json({"library": LIB.list(), "game": LIB.game_files(),
                               "screenshots": LIB.screenshots()})
        if u.path == "/api/shot":
            path = parse_qs(u.query).get("p", [""])[0]
            if path not in LIB.screenshots(500):
                return self._json({"error": "not a screenshot"}, 404)
            return self._file(path, "image/png" if path.lower().endswith(".png") else "image/jpeg")
        m = re.match(r"^/api/headtex/(t_[0-9a-f]{16}\.png)$", u.path)
        if m:
            p = os.path.join(CACHE, "textures", m.group(1))
            if not os.path.exists(p):
                return self._json({"error": "not found"}, 404)
            return self._file(p, "image/png")
        m = re.match(r"^/api/headtex/(male|female)(?:/(normal|spec|trans|makeup))?$", u.path)
        if m:
            png = head_builder().skin_png(m.group(1), os.path.join(CACHE, "textures"), m.group(2) or "diffuse")
            if not png:
                return self._json({"error": "no skin texture"}, 404)
            return self._file(png, "image/png")
        m = re.match(r"^/api/chars/%s/(file|image|head)$" % CHAR_ID, u.path)
        if not m:
            return self._json({"error": "not found"}, 404)
        cid, what = m.groups()
        if what == "head":
            with open(LIB.chf_path(cid), "rb") as fh:
                return self._json(head_payload(fh.read()))
        if what == "file":
            meta = next((c for c in LIB.list() if c["id"] == cid), None)
            if not meta:
                return self._json({"error": "not found"}, 404)
            name = re.sub(r"[^A-Za-z0-9 _\-]", "", meta["name"]).strip() or cid
            return self._file(LIB.chf_path(cid), "application/octet-stream", name + ".chf")
        ip = LIB.image_path(cid)
        if not ip:
            return self._json({"error": "no image"}, 404)
        ext = ip.rsplit(".", 1)[1]
        return self._file(ip, "image/" + ("jpeg" if ext == "jpg" else ext))

    def _chars_post(self, path, data):
        if path == "/api/chars/import":
            if data.get("upload"):
                fname = os.path.basename(data.get("filename", ""))
                cid, new = LIB.add(base64.b64decode(data["upload"]),
                                   os.path.splitext(fname)[0] or "Imported character", "Imported file: " + fname)
            else:
                cid, new = LIB.add_from_game(data.get("branch", ""), data.get("file", ""))
            return {"id": cid, "new": new}
        if path == "/api/chars/backup-all":
            return {"added": LIB.backup_all()}
        m = re.match(r"^/api/chars/%s/(install|image)$" % CHAR_ID, path)
        if not m:
            raise ValueError("not found")
        cid, what = m.groups()
        if what == "install":
            return LIB.install(cid, data.get("branch", ""), data.get("fileName", ""), bool(data.get("overwrite")))
        if data.get("screenshot"):
            LIB.image_from_screenshot(cid, data["screenshot"])
        else:
            LIB.set_image(cid, *characters.decode_data_url(data.get("dataUrl", "")))
        return {"ok": True}


def setup_info():
    missing = [m for m, ok in (("zstandard", gamedata.zstandard is not None), ("numpy", np is not None)) if not ok]
    try:
        import PIL  # noqa: F401
    except ImportError:
        missing.append("pillow")
    return {"scRoot": gamedata.SC_ROOT, "found": bool(gamedata.SC_ROOT), "branches": gamedata.branches(),
            "missingPackages": missing}


def _b64(a, t):
    return base64.b64encode(np.ascontiguousarray(a, dtype=t).tobytes()).decode()


def head_payload(chf_bytes):
    """3D head for a .chf as JSON (arrays base64-encoded little-endian)."""
    res = head_builder().build(chf_bytes, os.path.join(CACHE, "textures"))
    out = {"gender": res["gender"], "colours": res["colours"], "used": res["used"],
           "missing": res["missing"], "maskSource": res["maskSource"],
           "positions": _b64(res["positions"], "<f4"), "uvs": _b64(res["uvs"], "<f4"),
           "indices": _b64(res["indices"], "<u4"), "items": res.get("items", [])}
    try:  # makeup and freckles, drawn onto the preview's skin
        from studio import heads
        raw = heads.chf_payload(chf_bytes)
        # male files carry makeup values too, but the game only shows makeup on female heads
        out["makeup"] = heads.parse_makeup(raw) if res["gender"] == "female" else {}
        out["skinDetail"] = heads.parse_skin_params(raw)
    except Exception:
        pass
    if res["eyes"]:
        out["eyes"] = {k: _b64(v, "<u4" if k == "indices" else "<f4") for k, v in res["eyes"].items()}
    out["attachments"] = [{
        "kind": a["kind"], "item": a["item"], "colour": a["colour"], "parts": a["parts"],
        "positions": _b64(a["positions"], "<f4"), "uvs": _b64(a["uvs"], "<f4"),
        "indices": _b64(a["indices"], "<u4")} for a in res.get("attachments", [])]
    return out


# ---------------------------------------------------------------- photo -> character
DEFAULT_TEMPLATES = {"male": r"Data\Libs\CharacterCustomizer\masculinedefault.chf",
                     "female": r"Data\Libs\CharacterCustomizer\femininedefault.chf"}


def _lmap_path(g):
    return os.path.join(CACHE, "landmarks_%s.json" % g)


def facefit_get(path):
    m = re.match(r"^/api/basehead/(male|female)$", path)
    if m:
        info = head_builder().gender(m.group(1))
        out = {"gender": m.group(1), "positions": _b64(info["base"], "<f4"), "uvs": _b64(info["uv"], "<f4"),
               "indices": _b64(info["idx"], "<u4"), "colours": {"skin": "#a87a62", "iris": "#4a3020"}}
        if info["eyes"] is not None:
            ep, euv, eidx = info["eyes"]
            out["eyes"] = {"positions": _b64(ep, "<f4"), "uvs": _b64(euv, "<f4"), "indices": _b64(eidx, "<u4")}
        return out
    m = re.match(r"^/api/palette/(skin|makeup|eyes|hair)$", path)
    if m:  # the customizer's own colour pickers, so the page can offer exactly the game's choices
        import io as _io
        from PIL import Image
        pal = head_builder().palette(m.group(1))
        if pal is None:
            raise ValueError("palette not found in this game version")
        buf = _io.BytesIO()
        Image.fromarray(pal).save(buf, "PNG")
        return {"png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}
    m = re.match(r"^/api/styles/(male|female)$", path)
    if m:
        return {k: [n for n, _ in v] for k, v in head_builder().styles(m.group(1)).items()}
    m = re.match(r"^/api/landmarkmap/(male|female)$", path)
    if m:
        p = _lmap_path(m.group(1))
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as fh:
            return {"map": json.load(fh)}
    raise ValueError("not found")


def facefit_post(path, data):
    from studio import facefit, heads
    m = re.match(r"^/api/landmarkmap/(male|female)$", path)
    if m:
        lm = data.get("map")
        if not isinstance(lm, list) or len(lm) < 400:
            raise ValueError("landmark map looks incomplete")
        os.makedirs(CACHE, exist_ok=True)
        with open(_lmap_path(m.group(1)), "w", encoding="utf-8") as fh:
            json.dump(lm, fh)
        return {"ok": True, "found": sum(1 for x in lm if x)}
    if path == "/api/facefit":
        g = data.get("gender")
        if g not in ("male", "female"):
            raise ValueError("choose male or female")
        if not isinstance(data.get("landmarks"), list) or len(data["landmarks"]) < 400:
            raise ValueError("No face landmarks yet. Load a photo with a clearly visible face first.")
        tpl = data.get("template") or "default"
        if tpl == "default":
            template = head_builder()._p().read(DEFAULT_TEMPLATES[g])
        else:
            if not re.match(r"^[0-9a-f]{12}$", tpl):
                raise ValueError("bad template")
            with open(LIB.chf_path(tpl), "rb") as fh:
                template = fh.read()
        with open(_lmap_path(g), encoding="utf-8") as fh:
            lmap = json.load(fh)
        hb = head_builder()
        dna = heads.parse_dna_block(heads.chf_payload(template))
        if dna["rig"] == hb.gender("female")["rig"]:
            tg = "female"
        elif dna["rig"] == hb.gender("male")["rig"]:
            tg = "male"
        else:
            tg = "male" if dna["body"] == "9493d0fcf6676cdd" else "female"
        if tg != g:
            raise ValueError("That base character is %s. Pick a %s one, or the game default." % (tg, g))
        expr = facefit.expression(data["landmarks"])
        blend, detail = facefit.fit_regions(hb, g, lmap, data["landmarks"], expr)
        cols = facefit.game_colours(data.get("skin"), data.get("iris"), data.get("hair"), hb)
        hexrgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
        for k in ("skin", "iris"):  # manual overrides from the colour pickers
            if data.get(k + "Override"):
                cols[k] = hexrgb(data[k + "Override"])
        if data.get("hairOverride"):
            cols["hair"] = facefit.game_colours(None, None, hexrgb(data["hairOverride"]), hb)["hair"]
        avail = hb.styles(g)
        items = {}
        stubble_only = (data.get("styles") or {}).get("beard") == "__stubble__"
        if stubble_only:
            items["beard"] = hb.item_guid("Universal_stubble")
            if not items["beard"]:
                raise ValueError("Couldn't find the stubble item in this game version")
        for slot, name in (data.get("styles") or {}).items():
            if slot == "beard" and stubble_only:
                continue
            if name and slot in avail:
                guid = dict(avail[slot]).get(name)
                if not guid:
                    raise ValueError("Unknown %s style: %s" % (slot, name))
                items[slot] = guid
        makeup = facefit.game_makeup(data.get("makeup"), hb)
        chf = heads.write_chf(template, blend=blend, skin=cols.get("skin"), iris=cols.get("iris"),
                              hair=cols.get("hair"), items=items, stubble_only=stubble_only, makeup=makeup,
                              params={k: v for k, v in (data.get("skinDetail") or {}).items()
                                      if k in heads.SKIN_PARAMS and isinstance(v, (int, float))})
        has_makeup = bool(heads.parse_makeup(heads.chf_payload(template)))
        names = hb.gender(g)["names"]
        return {"head": head_payload(chf), "chf": base64.b64encode(chf).decode(), "detail": detail, "expression": expr,
                "blend": [None if row is None else [(names[h] if h < len(names) else str(h), w) for h, w in row]
                          for row in blend],
                "gameColours": {k: ("#%02x%02x%02x" % v) for k, v in cols.items() if k in ("skin", "iris")},
                "makeup": heads.parse_makeup(heads.chf_payload(chf)) if has_makeup else None,
                "skinDetail": heads.parse_skin_params(heads.chf_payload(chf))}
    if path == "/api/hairmatch":
        from studio import hairmatch
        g = data.get("gender")
        if g not in ("male", "female"):
            raise ValueError("choose male or female")
        mw, mh, w = (int(data.get(k) or 0) for k in ("mw", "mh", "w"))
        cat = np.frombuffer(base64.b64decode(data.get("mask", "")), np.uint8)
        if not (0 < mw <= 1024 and 0 < mh <= 1024 and w > 0) or cat.size != mw * mh:
            raise ValueError("bad hair mask")
        if not isinstance(data.get("landmarks"), list) or len(data["landmarks"]) < 400:
            raise ValueError("no face landmarks")
        with open(_lmap_path(g), encoding="utf-8") as fh:
            lmap = json.load(fh)
        hb = head_builder()
        sx, s, R, t = facefit.align(hb, g, lmap, data["landmarks"])
        # landmarks are in full-photo pixels; the mask is a scaled crop at (ox, oy):
        # photo p = k m + o  ->  b2 = s R p + t = (s k) R m + (s R o + t)
        k = w / mw
        o = np.array([float(data.get("ox") or 0), float(data.get("oy") or 0)])
        tf = (sx, s * k, R, t + s * (R @ o))
        top, grid, valid = hairmatch.rank(hb, g, tf, cat.reshape(mh, mw), mw, mh, CACHE)
        if data.get("debug"):
            np.savez(os.path.join(CACHE, "_hairdebug.npz"), cat=grid, valid=valid)
        return {"styles": [{"name": n, "score": round(sc, 3)} for n, sc in top]}
    if path == "/api/facefit/save":
        raw = base64.b64decode(data.get("chf", ""))
        cid, _ = LIB.add(raw, (data.get("name") or "Photo character")[:60], "Created from a photo")
        if data.get("photo"):
            try:
                LIB.set_image(cid, *characters.decode_data_url(data["photo"]))
            except Exception:
                pass
        return {"id": cid}
    raise ValueError("not found")


def make_server(port=PORT):
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    srv = make_server(PORT)
    url = "http://localhost:%d/" % PORT
    print("SC Character Studio by valkyrie_41 running at", url)
    print("Close this window to stop it.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
