"""Find the Star Citizen install and read files out of its Data.p4k archive.

Data.p4k is a zip64 archive whose entries are zstd-compressed (method 100); many game XML
files inside are CryXmlB (CryEngine binary XML). Nothing here writes to the game folder.
"""
import json
import os
import string
import struct
import xml.etree.ElementTree as ET

try:
    import zstandard
except ImportError:  # reported nicely by the server
    zstandard = None

from .paths import SETTINGS
BRANCHES = ("LIVE", "PTU", "EPTU", "TECH-PREVIEW", "HOTFIX")


def looks_like_sc(path):
    """A StarCitizen folder holds LIVE/PTU/... subfolders containing Data.p4k."""
    return bool(path) and any(os.path.exists(os.path.join(path, b, "Data.p4k")) for b in BRANCHES)


def _candidates():
    env = os.environ.get("SC_ROOT")
    if env:
        yield env
    try:
        with open(SETTINGS, encoding="utf-8") as fh:
            yield json.load(fh).get("scRoot", "")
    except (OSError, ValueError):
        pass
    tails = [r"Program Files\Roberts Space Industries\StarCitizen", r"Roberts Space Industries\StarCitizen",
             r"Games\Roberts Space Industries\StarCitizen", r"Games\StarCitizen", r"StarCitizen",
             r"Program Files (x86)\Roberts Space Industries\StarCitizen"]
    for d in string.ascii_uppercase:
        drive = d + ":\\"
        if os.path.exists(drive):
            for t in tails:
                yield os.path.join(drive, t)


def find_sc_root():
    for c in _candidates():
        if looks_like_sc(c):
            return os.path.normpath(c)
    return None


def set_sc_root(path):
    """Remember a user-chosen install folder (accepts the StarCitizen folder or one of its branches)."""
    global SC_ROOT
    path = os.path.normpath(path.strip().strip('"'))
    if not looks_like_sc(path) and looks_like_sc(os.path.dirname(path)):
        path = os.path.dirname(path)
    if not looks_like_sc(path):
        raise ValueError("That folder doesn't look like a Star Citizen install (no LIVE\\Data.p4k inside).")
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    with open(SETTINGS, "w", encoding="utf-8") as fh:
        json.dump({"scRoot": path}, fh, indent=1)
    SC_ROOT = path
    return path


SC_ROOT = find_sc_root()


def branches():
    if not SC_ROOT:
        return []
    return [b for b in BRANCHES if os.path.exists(os.path.join(SC_ROOT, b, "Data.p4k"))]


# ---------------------------------------------------------------- p4k reader
class P4K:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.f.seek(0, 2)
        size = self.f.tell()
        self.f.seek(max(0, size - 70000))
        tail = self.f.read()
        loc = tail.rfind(b"PK\x06\x07")
        if loc < 0:
            raise ValueError("Data.p4k: zip64 locator not found")
        z64off = struct.unpack_from("<Q", tail, loc + 8)[0]
        self.f.seek(z64off)
        rec = self.f.read(56)
        _, cdsize, cdoff = struct.unpack_from("<QQQ", rec, 32)
        self.f.seek(cdoff)
        self.cd = self.f.read(cdsize)

    def read(self, name):
        key = name.replace("/", "\\").encode()
        # exact match: the name must be the whole filename field of a central-directory header
        # (a plain find() would match "x.skin" inside "x.skinm")
        k = -1
        lower = getattr(self, "_lower", None)
        start = 0
        while True:
            k = self.cd.find(key, start)
            if k < 0 and lower is None:
                self._lower = lower = self.cd.lower()
            if k < 0:
                k = lower.find(key.lower(), start)
            if k < 0:
                raise FileNotFoundError(name)
            if k >= 46 and self.cd[k - 46:k - 42] == b"PK\x01\x02" and struct.unpack_from("<H", self.cd, k - 18)[0] == len(key):
                break
            start = k + 1
        h = k - 46
        (_, _, _, _, meth, _, _, _, csz, usz, nlen, elen, _, _, _, _, lho) = struct.unpack_from(
            "<IHHHHHHIIIHHHHHII", self.cd, h)
        extra = self.cd[h + 46 + nlen:h + 46 + nlen + elen]
        # zip64 extra (id 0x0001) holds the real sizes/offset when fields are 0xFFFFFFFF
        if extra[:2] == b"\x01\x00":
            vals = list(struct.unpack_from("<%dQ" % (struct.unpack_from("<H", extra, 2)[0] // 8), extra, 4))
            if usz == 0xFFFFFFFF: usz = vals.pop(0)
            if csz == 0xFFFFFFFF: csz = vals.pop(0)
            if lho == 0xFFFFFFFF: lho = vals.pop(0)
        self.f.seek(lho)
        lh = self.f.read(30)
        ln, le = struct.unpack_from("<HH", lh, 26)
        self.f.seek(lho + 30 + ln + le)
        data = self.f.read(csz)
        if meth == 100:
            if zstandard is None:
                raise RuntimeError("Python package 'zstandard' is missing (pip install zstandard)")
            data = zstandard.ZstdDecompressor().decompress(data, max_output_size=usz)
        elif meth == 8:
            import zlib
            data = zlib.decompress(data, -15)
        elif meth != 0:
            raise ValueError("unsupported compression %d" % meth)
        return data

    def close(self):
        self.f.close()


# ---------------------------------------------------------------- CryXmlB
def parse_cryxml(buf):
    if buf[:5] == b"<?xml" or buf[:1] == b"<":
        return ET.fromstring(buf)
    assert buf[:8] == b"CryXmlB\x00", "not CryXmlB"
    (_, noff, ncnt, aoff, acnt, coff, ccnt, soff, _) = struct.unpack_from("<9I", buf, 8)

    def s(o):
        e = buf.index(b"\x00", soff + o)
        return buf[soff + o:e].decode("utf-8", "replace")

    nodes = [struct.unpack_from("<IIHHiiii", buf, noff + i * 28) for i in range(ncnt)]
    attrs = [struct.unpack_from("<II", buf, aoff + i * 8) for i in range(acnt)]
    kids = struct.unpack_from("<%di" % ccnt, buf, coff)
    els = []
    for (tag, _, na, _, _, fa, _, _) in nodes:
        el = ET.Element(s(tag))
        for k, v in attrs[fa:fa + na]:
            el.set(s(k), s(v))
        els.append(el)
    for i, (_, _, _, nc, _, _, fc, _) in enumerate(nodes):
        for cid in kids[fc:fc + nc]:
            els[i].append(els[cid])
    return els[0]
