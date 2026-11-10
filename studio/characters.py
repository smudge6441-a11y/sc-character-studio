"""Character library: keeps Star Citizen .chf character files safe, named and
installable into LIVE / PTU. Files are copied byte-for-byte; never modified.

Library layout:  characters/<id>/character.chf, meta.json, image.(jpg|png|webp)
<id> is the first 12 hex chars of the file's SHA-256, so duplicates collapse.
"""
import base64
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import struct

from . import gamedata

CHF_SIZE = 4096
IMG_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
SAFE_FILE = re.compile(r"^[A-Za-z0-9 _\-\.']{1,60}$")


def game_dir(branch):
    return os.path.join(gamedata.SC_ROOT, branch, "user", "client", "0", "CustomCharacters")


def screenshot_dirs():
    out = []
    for b in gamedata.branches():
        d = os.path.join(gamedata.SC_ROOT, b, "ScreenShots")
        if os.path.isdir(d):
            out.append(d)
    return out


def inspect(data):
    """Return (ok, reason). A .chf is 4096 bytes: 0x4242 magic, crc, sizes, zstd body."""
    if len(data) != CHF_SIZE:
        return False, "Not a Star Citizen character file (wrong size: %d bytes)" % len(data)
    magic, _, _, csz, usz = struct.unpack_from("<HHIII", data, 0)
    if magic != 0x4242:
        return False, "Not a Star Citizen character file (bad header)"
    if not (0 < csz <= CHF_SIZE - 16) or not (0 < usz < 1 << 20):
        return False, "Character file header looks damaged"
    try:
        import zstandard
        zstandard.ZstdDecompressor().decompress(data[16:16 + csz], max_output_size=usz)
    except Exception:
        return False, "Character data is damaged (won't decompress)"
    return True, ""


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Library:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    # ------------------------------------------------------------ storage
    def _dir(self, cid):
        if not re.match(r"^[0-9a-f]{12}$", cid or ""):
            raise ValueError("bad character id")
        return os.path.join(self.root, cid)

    def _meta(self, cid):
        with open(os.path.join(self._dir(cid), "meta.json"), encoding="utf-8") as fh:
            return json.load(fh)

    def _save_meta(self, cid, meta):
        p = os.path.join(self._dir(cid), "meta.json")
        with open(p + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        os.replace(p + ".tmp", p)

    def image_path(self, cid):
        d = self._dir(cid)
        for ext in IMG_EXT.values():
            p = os.path.join(d, "image." + ext)
            if os.path.exists(p):
                return p
        return None

    def chf_path(self, cid):
        return os.path.join(self._dir(cid), "character.chf")

    # ------------------------------------------------------------ queries
    def list(self):
        out = []
        for cid in sorted(os.listdir(self.root)):
            if not re.match(r"^[0-9a-f]{12}$", cid) or not os.path.exists(os.path.join(self.root, cid, "meta.json")):
                continue
            m = self._meta(cid)
            m["id"] = cid
            m["hasImage"] = self.image_path(cid) is not None
            out.append(m)
        out.sort(key=lambda m: m.get("added", ""), reverse=True)
        return out

    def by_hash(self):
        return {m["sha"]: m["id"] for m in self.list()}

    def game_files(self):
        known = self.by_hash()
        out = {}
        for b in gamedata.branches():
            d = game_dir(b)
            files = []
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if not f.lower().endswith(".chf"):
                        continue
                    p = os.path.join(d, f)
                    data = open(p, "rb").read()
                    ok, why = inspect(data)
                    h = sha(data)
                    files.append({"file": f, "size": len(data), "mtime": os.path.getmtime(p),
                                  "valid": ok, "problem": why, "sha": h, "libraryId": known.get(h)})
            out[b] = {"exists": os.path.isdir(d), "dir": d, "files": files}
        return out

    # ------------------------------------------------------------ mutations
    def add(self, data, name, source, notes=""):
        ok, why = inspect(data)
        if not ok:
            raise ValueError(why)
        h = sha(data)
        cid = h[:12]
        d = self._dir(cid)
        if os.path.exists(os.path.join(d, "meta.json")):
            return cid, False
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "character.chf"), "wb") as fh:
            fh.write(data)
        self._save_meta(cid, {
            "name": (name or "Unnamed character")[:60], "notes": notes, "body": "", "tags": [],
            "source": source, "sha": h, "added": _dt.datetime.now().isoformat(timespec="seconds"),
        })
        return cid, True

    def add_from_game(self, branch, file):
        if branch not in gamedata.branches() or os.path.basename(file) != file or not file.lower().endswith(".chf"):
            raise ValueError("bad game file")
        data = open(os.path.join(game_dir(branch), file), "rb").read()
        return self.add(data, os.path.splitext(file)[0], "%s\\%s" % (branch, file))

    def backup_all(self):
        added = 0
        for b, info in self.game_files().items():
            for f in info["files"]:
                if f["valid"] and not f["libraryId"]:
                    self.add_from_game(b, f["file"])
                    added += 1
        return added

    def update(self, cid, fields):
        m = self._meta(cid)
        for k in ("name", "notes", "body"):
            if k in fields:
                m[k] = str(fields[k])[:2000 if k == "notes" else 60]
        if "tags" in fields:
            m["tags"] = [str(t)[:30] for t in fields["tags"]][:20]
        self._save_meta(cid, m)
        return m

    def delete(self, cid):
        """OneDrive (or antivirus) can hold a fresh folder for a moment, so retry briefly."""
        import stat
        import time

        def unlock(func, path, _):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        d = self._dir(cid)
        for attempt in range(10):
            try:
                shutil.rmtree(d, onerror=unlock)
                return
            except (PermissionError, OSError):
                if attempt == 9:
                    raise
                time.sleep(0.4)

    def set_image(self, cid, mime, data):
        if mime not in IMG_EXT:
            raise ValueError("Use a JPG, PNG or WebP image")
        if len(data) > 15 * 1024 * 1024:
            raise ValueError("Image is too large (15 MB max)")
        old = self.image_path(cid)
        if old:
            os.remove(old)
        with open(os.path.join(self._dir(cid), "image." + IMG_EXT[mime]), "wb") as fh:
            fh.write(data)

    def install(self, cid, branch, file_name, overwrite=False):
        """Copy into the game's CustomCharacters folder. If a different file would be
        replaced, it is first saved into the library so nothing is ever lost."""
        if branch not in gamedata.branches():
            raise ValueError("Unknown game branch")
        file_name = file_name.strip()
        if file_name.lower().endswith(".chf"):
            file_name = file_name[:-4]
        if not SAFE_FILE.match(file_name) or file_name.startswith("."):
            raise ValueError("File name can use letters, numbers, spaces, ' - _ and .")
        data = open(self.chf_path(cid), "rb").read()
        d = game_dir(branch)
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, file_name + ".chf")
        backed_up = None
        if os.path.exists(dest):
            existing = open(dest, "rb").read()
            if existing == data:
                return {"path": dest, "unchanged": True}
            if not overwrite:
                return {"conflict": True, "path": dest}
            if inspect(existing)[0]:
                backed_up, _ = self.add(existing, "Backup of %s (%s)" % (file_name, branch),
                                        "%s\\%s.chf (auto-backup before replace)" % (branch, file_name))
        with open(dest + ".tmp", "wb") as fh:
            fh.write(data)
        os.replace(dest + ".tmp", dest)
        return {"path": dest, "backedUp": backed_up}

    # ------------------------------------------------------------ screenshots
    @staticmethod
    def screenshots(limit=40):
        files = []
        for d in screenshot_dirs():
            for f in os.listdir(d):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    p = os.path.join(d, f)
                    files.append((os.path.getmtime(p), p))
        files.sort(reverse=True)
        return [p for _, p in files[:limit]]

    def image_from_screenshot(self, cid, path):
        if path not in self.screenshots(500):
            raise ValueError("Not a Star Citizen screenshot")
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        self.set_image(cid, mime, open(path, "rb").read())


def decode_data_url(s):
    m = re.match(r"^data:(image/[a-z]+);base64,(.+)$", s or "", re.S)
    if not m:
        raise ValueError("bad image data")
    return m.group(1), base64.b64decode(m.group(2))
