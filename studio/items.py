"""Resolve item IDs stored in .chf files to item names, using the game's database (Game2.dcb).

Each DataCore record is 36 bytes:
    u32 name_offset, u32 file_offset, u32 tag, u32 struct_index, 16-byte GUID, u16 instance, u16 size
Names live in a string blob whose start we calibrate from a pair the game itself ships:
masculinedefault.chf (binary) lists its body item first, and MasculineDefault.xml names it.
"""
import re
import struct
import zstandard

CAL_CHF = r"Data\Libs\CharacterCustomizer\masculinedefault.chf"
CAL_NAME = b"EntityClassDefinition.body_01_noMagicPocket"


def chf_items(raw):
    """Item GUIDs equipped in a decompressed .chf, in file order (body, head, eyes, hair, ...)."""
    o = 48
    nreg, nslot = struct.unpack_from("<HH", raw, o + 16)
    o += 24 + nreg * nslot * 4
    count = struct.unpack_from("<I", raw, o)[0]
    o += 8
    out = []
    # records: u32 prefix, 16-byte GUID, u32 child count, u32 pad. Count above only covers the top
    # level, so keep reading while records look like records (non-zero GUID).
    while o + 28 <= len(raw) and len(out) < 64:
        g = raw[o + 4:o + 20]
        kids, pad = struct.unpack_from("<II", raw, o + 20)
        if g == b"\0" * 16 or kids > 64 or pad != 0:
            break
        out.append(g)
        o += 28
    return out


class ItemNames:
    def __init__(self, p4k):
        self.p4k = p4k
        self.map = None  # guid bytes -> name

    def _load(self):
        dcb = self.p4k.read(r"Data\Game2.dcb")
        cal = self.p4k.read(CAL_CHF)
        csz, usz = struct.unpack_from("<II", cal, 8)
        cal_raw = zstandard.ZstdDecompressor().decompress(cal[16:16 + csz], max_output_size=usz)
        body_guid = chf_items(cal_raw)[0]
        rec = dcb.find(body_guid) - 16
        name_str = dcb.find(CAL_NAME + b"\0")
        if rec < 0 or name_str < 0:
            raise RuntimeError("couldn't calibrate the game database (body item not found)")
        base = name_str - struct.unpack_from("<I", dcb, rec)[0]
        nstructs = struct.unpack_from("<I", dcb, 16)[0]

        def ok(r):
            if r < 0 or r + 36 > len(dcb):
                return False
            name, _, _, sidx = struct.unpack_from("<4I", dcb, r)
            return sidx < nstructs and 0 <= base + name < len(dcb) and dcb[r + 16:r + 32] != b"\0" * 16
        start = rec
        while ok(start - 36):
            start -= 36
        end = rec
        while ok(end + 36):
            end += 36
        names = {}
        for r in range(start, end + 36, 36):
            no = struct.unpack_from("<I", dcb, r)[0]
            s = base + no
            e = dcb.find(b"\0", s, s + 256)
            if e > s:
                names[bytes(dcb[r + 16:r + 32])] = dcb[s:e].decode("utf-8", "replace")
        self.map = names

    def name(self, guid):
        if self.map is None:
            self._load()
        n = self.map.get(bytes(guid))
        return n.split(".", 1)[1] if n and "." in n else n
