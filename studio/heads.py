"""Rebuild a Star Citizen character head from a .chf file.

How the game does it (as far as decoded):
  * Every player head shares one topology: protos_human_<g>_face_t1_pu_head (5584 verts male).
  * The protos .dna rig file lists ~60 stock heads (archetypes + named NPC heads) in a fixed order.
  * A .chf holds a "DNA" block: for each of 13 face regions (BROW_L ... NECK_C) up to 4
    (weight u16, head-index u16) slots, stored slot-major: entries[slot * regions + region].
  * The customizer's region-highlight atlas (13 tiles, 256x256 each, red channel) masks each region.
  Face = base + sum_region mask_r(v) * sum_slot w * (head_k(v) - base(v)).
This is an approximation of the game's blend (no skin shader, no hair yet).
"""
import io
import os
import re
import struct
import threading
import zstandard
import numpy as np

from . import gamedata
from .gamedata import P4K

REGIONS = ["BROW_L", "BROW_R", "EYE_L", "EYE_R", "NOSE_C", "EAR_L", "EAR_R",
           "CHEEK_L", "CHEEK_R", "MOUTH_C", "JAWLINE_C", "CROWN_C", "NECK_C"]
HEADS = r"Data\Objects\Characters\Human\heads"
ATLAS = HEADS + r"\highlighting\protos_human_face_highlighting_blend_region_atlas.dds"
S_INDICES, S_VERTSUVS = 0xEECDC168, 0x91329AE9


# ------------------------------------------------------------------ mesh
def read_ivo_subsets(data):
    """Submesh table: [(material_id, first_index, num_indices)] (48-byte entries at mesh+168)."""
    n, tbl = struct.unpack_from("<II", data, 8)
    for i in range(n):
        t, _, off = struct.unpack_from("<IIQ", data, tbl + i * 16)
        if t == 0xB8757777:
            _, _, nv, ni, nsub = struct.unpack_from("<5I", data, off)
            out = []
            for k in range(min(nsub, 32)):
                mat, first, count = struct.unpack_from("<3I", data, off + 168 + k * 48)
                if first + count <= ni:
                    out.append((mat, first, count))
            return out or [(0, 0, ni)]
    return []


def read_ivo_mesh(data):
    """#ivo .skinm -> (positions float32[N,3], uvs float32[N,2], indices uint32[M])."""
    assert data[:4] == b"#ivo", "not an #ivo file"
    n, tbl = struct.unpack_from("<II", data, 8)
    mesh_off = None
    for i in range(n):
        t, _, off = struct.unpack_from("<IIQ", data, tbl + i * 16)
        if t == 0xB8757777:
            mesh_off = off
    if mesh_off is None:
        raise ValueError("no mesh chunk")
    _, _, nv, ni = struct.unpack_from("<4I", data, mesh_off)
    pos = uv = idx = None
    o = mesh_off
    end = min(len(data), mesh_off + 64 + nv * 200 + ni * 4)
    # streams are (u32 type, u32 elemsize, data) aligned to 8; find them by scanning
    while o < end - 8:
        t, sz = struct.unpack_from("<II", data, o)
        if t == S_INDICES and sz in (2, 4) and idx is None:
            idx = np.frombuffer(data, dtype="<u2" if sz == 2 else "<u4", count=ni, offset=o + 8).astype(np.uint32)
            o = (o + 8 + ni * sz + 7) & ~7
            continue
        if t == S_VERTSUVS and sz == 20 and pos is None:
            raw = np.frombuffer(data, dtype=np.uint8, count=nv * 20, offset=o + 8).reshape(nv, 20)
            pos = raw[:, :12].copy().view("<f4").reshape(nv, 3)
            uv = raw[:, 16:20].copy().view("<f2").reshape(nv, 2).astype(np.float32)
            o = (o + 8 + nv * 20 + 7) & ~7
            continue
        o += 4
        if pos is not None and idx is not None:
            break
    if pos is None or idx is None:
        raise ValueError("mesh streams not found")
    return pos.astype(np.float32), uv, idx


def read_dds_rgba(data):
    """Uncompressed 32-bit DDS -> uint8[H,W,4] (the region atlas is plain BGRA/RGBA)."""
    h, w = struct.unpack_from("<II", data, 12)
    flags, = struct.unpack_from("<I", data, 80)
    rmask, = struct.unpack_from("<I", data, 92)
    px = np.frombuffer(data, dtype=np.uint8, count=w * h * 4, offset=128).reshape(h, w, 4)
    if rmask == 0x00FF0000:  # BGRA
        px = px[:, :, [2, 1, 0, 3]]
    return px


def split_dds_to_png(p4k, path, max_size=1024, mask=False, normalize=False, normal=False):
    """Star Citizen splits big textures: <x>.dds holds the header + small mips, <x>.dds.N the big
    ones (highest N = full size). Rebuild one mip <= max_size as a standalone DDS and decode it."""
    from PIL import Image
    head = p4k.read(path)
    h, w, _, _, mips = struct.unpack_from("<5I", head, 12)
    dx10 = head[84:88] == b"DX10"
    hdr_len = 148 if dx10 else 128
    fourcc = head[84:88]
    dxgi = struct.unpack_from("<I", head, 128)[0] if dx10 else None
    bpb = 8 if (fourcc in (b"DXT1", b"ATI1", b"BC4U") or dxgi in (70, 71, 72, 79, 80, 81)) else 16
    # how many split parts exist?
    n = 0
    while True:
        try:
            p4k.read("%s.%d" % (path, n + 1))
            n += 1
        except FileNotFoundError:
            break
    # mip k (0 = full) lives in part (n - k) for k < n; smaller mips are in the base file
    k = 0
    while (w >> k) > max_size and k < n:
        k += 1
    mw, mh = max(1, w >> k), max(1, h >> k)
    size = max(1, (mw + 3) // 4) * max(1, (mh + 3) // 4) * bpb
    if k < n:
        data = p4k.read("%s.%d" % (path, n - k))[:size]
    else:
        data = head[hdr_len:hdr_len + size]
    hdr = bytearray(head[:hdr_len])
    struct.pack_into("<II", hdr, 12, mh, mw)
    struct.pack_into("<I", hdr, 28, 1)  # one mip
    img = Image.open(io.BytesIO(bytes(hdr) + data))
    img.load()
    out = io.BytesIO()
    if normal:
        # BC5 normal maps keep X and Y only: rebuild Z so it's a standard RGB tangent-space normal map
        a = np.asarray(img.convert("RGB"), np.float32) / 127.5 - 1.0
        z = np.sqrt(np.clip(1.0 - a[..., 0] ** 2 - a[..., 1] ** 2, 0, 1))
        rgb = np.stack([a[..., 0], a[..., 1], z], -1)
        Image.fromarray(np.clip((rgb + 1) * 127.5, 0, 255).astype(np.uint8)).save(out, "PNG")
    elif mask:
        # one greyscale coverage mask: the alpha channel when it carries data, else red
        # (BC4/BC5 opacity maps keep coverage in red; BC3/BC7 scalp maps in alpha)
        bands = img.getbands()
        a = img.getchannel("A") if "A" in bands else None
        if a is not None and a.getextrema()[0] < 250:
            m = a
        else:
            m = img.getchannel("R") if "R" in bands else img.convert("L")
        if normalize:
            lo, hi = m.getextrema()
            if 0 < hi < 250:
                m = m.point(lambda v: min(255, int(v * 255 / hi)))
        m.save(out, "PNG")
    else:
        img.convert("RGB").save(out, "PNG")
    return out.getvalue(), (mw, mh), (fourcc.decode("latin-1"), dxgi)


# ------------------------------------------------------------------ chf
def chf_payload(chf_bytes):
    csz, usz = struct.unpack_from("<II", chf_bytes, 8)
    return zstandard.ZstdDecompressor().decompress(chf_bytes[16:16 + csz], max_output_size=usz)


HAIR_KEYS = {"5ac1f64a": "dyeAmount", "a300fab9": "melanin", "aff0fb62": "redness"}


def parse_hair_blocks(raw):
    """Hair-type material parameter blocks, in file order (hair, beard if worn, brows)."""
    out = []
    o = raw.find(bytes.fromhex("5ac1f64a"))
    while o >= 0:
        vals = {}
        p = o
        known = set(HAIR_KEYS) | {"76400806", "c8a79aa5", "d90b37c3"}
        while p + 8 <= len(raw) and raw[p:p + 4].hex() in known:  # (key, f32, pad) records
            k = raw[p:p + 4].hex()
            if k in HAIR_KEYS:
                vals[HAIR_KEYS[k]] = struct.unpack_from("<f", raw, p + 4)[0]
            p += 12
        c = raw.find(bytes.fromhex("1408e915"), o, o + 200)
        vals["dye"] = tuple(raw[c + 4:c + 7]) if c >= 0 else (128, 100, 80)
        out.append(vals)
        o = raw.find(bytes.fromhex("5ac1f64a"), o + 1)
    return out


PALETTES = {  # the character customizer's own colour pickers (UI textures)
    "hair": r"Data\UI\Textures\I_InteractiveScreens\CharacterCustomizer\root_color_graph_31_03_2025.dds",
    "eyes": r"Data\UI\Textures\I_InteractiveScreens\CharacterCustomizer\eye_color_graph_11_04_2024.dds",
    "skin": r"Data\UI\Textures\I_InteractiveScreens\CharacterCustomizer\skin_tone_lut_12_02_2024.dds",
    # makeup colours: black at the top, full colour in the middle, white at the bottom
    "makeup": r"Data\UI\Textures\I_InteractiveScreens\CharacterCustomizer\color_graph_04_03_2024.dds",
}


def hair_colour(block, graph=None):
    """Natural hair colour from melanin + redness, mixed with dye. Returns '#rrggbb'.
    graph: the customizer's hair palette (x = melanin 0..1, y = redness 0..1) as uint8[H, W, 3];
    when given, the colour is read straight off it, which is what the game shows."""
    m = min(1.0, max(0.0, block.get("melanin", 0.8)))
    red = min(1.0, max(0.0, block.get("redness", 0.1)))
    dye_amt = min(1.0, max(0.0, block.get("dyeAmount", 0.0)))
    if graph is not None:
        h, w = graph.shape[:2]
        base_srgb = graph[int(round(red * (h - 1))), int(round(m * (w - 1)))].astype(float) / 255.0
        lin = base_srgb ** 2.2
        dye = (np.array(block.get("dye", (128, 100, 80))) / 255.0) ** 2.2
        lin = lin * (1 - dye_amt) + dye * dye_amt
        return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in np.clip(lin, 0, 1) ** (1 / 2.2))
    light = np.array([0.80, 0.62, 0.38])   # linear-ish blond
    dark = np.array([0.012, 0.008, 0.006])  # near-black
    t = m ** 0.45
    base = light * (1 - t) + dark * t
    auburn = np.array([0.30, 0.06, 0.015]) * (1 - 0.7 * m) + dark * 0.7 * m
    base = base * (1 - red * 0.75) + auburn * (red * 0.75)
    dye = (np.array(block.get("dye", (128, 100, 80))) / 255.0) ** 2.2
    lin = base * (1 - dye_amt) + dye * dye_amt
    srgb = np.clip(lin, 0, 1) ** (1 / 2.2)
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in srgb)


COLOUR_KEYS = {"skin": bytes.fromhex("970753bd"), "iris": bytes.fromhex("ac342a44"), "dye": bytes.fromhex("1408e915")}

# Makeup (decoded from characters saved by the in-game customizer): three shader slots with an
# R/G/B mask channel each. The customizer's Eyes, Cheeks and Lips are these channels; each has a
# colour record (a pixel of the makeup palette) and an opacity float keyed by CRC32C(name).
# Older files (like the game's own built-in defaults) have one opacity per slot ("Makeup1Opacity");
# the game upgrades them to per-channel values on load, so either name works.
MAKEUP = {"eyes": ("2f0e23e3", ("Makeup1OpacityG", "Makeup1Opacity")),
          "cheeks": ("35e05386", ("Makeup2OpacityB", "Makeup2Opacity")),
          "lips": ("341ddde1", ("Makeup3OpacityB", "Makeup3Opacity"))}


def param_key(raw, names):
    """Offset of a material parameter record (key = CRC32C of its name), or -1.
    names: one name, or several to try in order."""
    for name in ([names] if isinstance(names, str) else names):
        v = crc32c(name.encode()) & 0xffffffff
        for k in (v.to_bytes(4, "big"), v.to_bytes(4, "little")):
            o = raw.find(k)
            if o >= 0:
                return o
    return -1


def parse_makeup(raw):
    """{area: {"colour": "#rrggbb", "opacity": 0..1}} for the makeup this file has."""
    out = {}
    for area, (ckey, oname) in MAKEUP.items():
        c, o = raw.find(bytes.fromhex(ckey)), param_key(raw, oname)
        if c >= 0 and o >= 0:
            out[area] = {"colour": "#%02x%02x%02x" % tuple(raw[c + 4:c + 7]),
                         "opacity": round(struct.unpack_from("<f", raw, o + 4)[0], 3)}
    return out


def parse_colours(raw):
    """Colours are stored as (u32 key, r, g, b, a) with sRGB bytes. Dye appears once per
    hair material in order: hair, beard, brows (when present)."""
    out = {}
    for name, key in COLOUR_KEYS.items():
        found = []
        o = raw.find(key)
        while o >= 0:
            r, g, b, a = raw[o + 4:o + 8]
            found.append("#%02x%02x%02x" % (r, g, b))
            o = raw.find(key, o + 1)
        out[name] = found
    return {"skin": (out["skin"] or ["#b07a62"])[0], "iris": (out["iris"] or ["#4a3a2a"])[0],
            "hair": (out["dye"] or ["#3a2a20"])[0], "allDyes": out["dye"]}


def parse_dna_block(raw):
    """Find and decode the DNA block inside a decompressed .chf."""
    # block starts at 48 in every known file: u64 body id, u32 rig id, u32 0, u16 regions, u16 slots, u16, u16
    o = 48
    body = raw[o:o + 8].hex()
    rig = raw[o + 8:o + 12].hex()
    nreg, nslot, _, _ = struct.unpack_from("<4H", raw, o + 16)
    if not (1 <= nreg <= 32 and 1 <= nslot <= 8):
        raise ValueError("no face DNA in this character file")
    ents = struct.unpack_from("<%dH" % (nreg * nslot * 2), raw, o + 24)
    slots = []
    for r in range(nreg):
        row = []
        for s in range(nslot):
            k = (s * nreg + r) * 2
            w, head = ents[k], ents[k + 1]
            if w:
                row.append((head, w / 65535.0))
        slots.append(row)
    return {"body": body, "rig": rig, "regions": REGIONS[:nreg], "blend": slots}


# ------------------------------------------------------------------ writing .chf
_CRC32C = None


def crc32c(data):
    global _CRC32C
    if _CRC32C is None:
        t = []
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ (0x82F63B78 if c & 1 else 0)
            t.append(c)
        _CRC32C = t
    c = 0xFFFFFFFF
    for x in data:
        c = _CRC32C[(c ^ x) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def _set_colour(raw, key, rgb):
    o = raw.find(key)
    while o >= 0:
        raw[o + 4:o + 7] = bytes(rgb)
        o = raw.find(key, o + 1)


SLOTS = {"hair": "951a6013", "brows": "22ee8717", "beard": "1cbbef98"}  # item-port ids, same in every file
STUBBLE_MESH = r"Data\Objects\Characters\Human\heads\shared\facialhair\m_stubble_facialhair_01.skinm"


def item_records(raw):
    """Item list after the DNA block: u32 count, then count x 28-byte records
    [u32 0][u32 port id][16-byte item GUID][u32 child count]. Returns [(offset, port_hex, guid)]."""
    o = 48
    nreg, nslot = struct.unpack_from("<HH", raw, o + 16)
    o += 24 + nreg * nslot * 4
    count = struct.unpack_from("<I", raw, o)[0]
    out = []
    for i in range(min(count, 64)):
        r = o + 4 + 28 * i
        out.append((r, raw[r + 4:r + 8].hex(), bytes(raw[r + 8:r + 24])))
    return out


def _list_start(raw):
    o = 48
    nreg, nslot = struct.unpack_from("<HH", raw, o + 16)
    return o + 24 + nreg * nslot * 4


item_records.list_start = _list_start


# Skin detail sliders on the customizer's DNA tab (floats 0..1, every character has them)
SKIN_PARAMS = ("FrecklesAmount", "FrecklesOpacity", "SunSpotsAmount", "SunSpotsOpacity")


def parse_skin_params(raw):
    out = {}
    for name in SKIN_PARAMS:
        o = param_key(raw, name)
        if o >= 0:
            out[name] = round(struct.unpack_from("<f", raw, o + 4)[0], 3)
    return out


def write_chf(template, blend=None, skin=None, iris=None, hair=None, items=None, stubble_only=False, makeup=None,
              params=None):
    """New .chf bytes from a template .chf.
    blend: list per region of [(head_index, weight 0..1)] (max slots per region from the template)
    skin / iris: (r, g, b) sRGB bytes.  hair: dict(melanin=, redness=) for the first hair block.
    items: {"hair"|"brows"|"beard": 16-byte GUID} swaps the item in that slot (slot must exist).
    stubble_only: items["beard"] is the stubble item; drop the beard's child records.
    makeup: {"eyes"|"cheeks"|"lips": {"colour": (r, g, b) or None, "opacity": 0..1 or None}};
    only changes makeup the template has (the makeup style itself is kept).
    params: {name: 0..1} for SKIN_PARAMS (freckles, sun spots).
    Header: u16 0x4242, u16 (kept from template), u32 CRC32C of bytes 16..4096, u32 csz, u32 usz."""
    raw = bytearray(chf_payload(template))
    o = 48
    nreg, nslot = struct.unpack_from("<HH", raw, o + 16)
    for slot, guid in (items or {}).items():
        if not guid:
            continue
        hits = [r for r, port, _ in item_records(raw) if port == SLOTS[slot]]
        if not hits:
            raise ValueError("The base character has no %s slot to change" % slot)
        r = hits[0]
        raw[r + 8:r + 24] = guid
        if slot == "beard" and stubble_only:
            # what the customizer writes for "stubble only": the stubble item sits in the beard
            # slot itself, with no child records (a beard's stubble is normally its child)
            kids = struct.unpack_from("<I", raw, r + 24)[0]
            if kids:
                del raw[r + 28:r + 28 + 28 * kids]
                struct.pack_into("<I", raw, r + 24, 0)
                lst = item_records.list_start(raw)
                struct.pack_into("<I", raw, lst, struct.unpack_from("<I", raw, lst)[0] - kids)
    if blend is not None:
        base = o + 24
        for r in range(nreg):
            if r < len(blend) and blend[r] is None:
                continue  # None = keep the template's own recipe for this region
            row = sorted(blend[r] if r < len(blend) else [], key=lambda x: -x[1])[:nslot]
            for s in range(nslot):
                head, w = row[s] if s < len(row) else (0, 0.0)
                k = base + (s * nreg + r) * 4
                struct.pack_into("<HH", raw, k, int(round(max(0.0, min(1.0, w)) * 65535)) if w > 0 else 0, head if w > 0 else 0)
    for name, val in (params or {}).items():
        if name in SKIN_PARAMS and val is not None:
            o = param_key(raw, name)
            if o >= 0:
                struct.pack_into("<f", raw, o + 4, float(min(1.0, max(0.0, val))))
    for area, val in (makeup or {}).items():
        ckey, oname = MAKEUP[area]
        if val.get("colour") is not None:
            _set_colour(raw, bytes.fromhex(ckey), val["colour"])
        if val.get("opacity") is not None:
            o = param_key(raw, oname)
            if o >= 0:
                struct.pack_into("<f", raw, o + 4, float(min(1.0, max(0.0, val["opacity"]))))
    if skin is not None:
        _set_colour(raw, COLOUR_KEYS["skin"], skin)
    if iris is not None:
        _set_colour(raw, COLOUR_KEYS["iris"], iris)
    if hair is not None:
        # every hair-type material block (hair, beard, brows) gets the same natural colour
        p = raw.find(bytes.fromhex("5ac1f64a"))
        while p >= 0:
            q = p
            while q + 8 <= len(raw) and raw[q:q + 4].hex() in ("5ac1f64a", "a300fab9", "aff0fb62", "76400806", "c8a79aa5", "d90b37c3"):
                k = raw[q:q + 4].hex()
                if k == "a300fab9" and "melanin" in hair:
                    struct.pack_into("<f", raw, q + 4, float(hair["melanin"]))
                elif k == "aff0fb62" and "redness" in hair:
                    struct.pack_into("<f", raw, q + 4, float(hair["redness"]))
                elif k == "5ac1f64a":
                    struct.pack_into("<f", raw, q + 4, 0.0)  # natural colour, no dye
                q += 12
            p = raw.find(bytes.fromhex("5ac1f64a"), q)
    comp = zstandard.ZstdCompressor(level=19, write_content_size=True, write_checksum=False).compress(bytes(raw))
    if 16 + len(comp) > 4096:
        raise ValueError("character data too large for a .chf")
    body = comp.ljust(4096 - 16, b"\0")
    out = bytearray(template[:4]) + struct.pack("<III", crc32c(body), len(comp), len(raw)) + body
    return bytes(out)


# ------------------------------------------------------------------ head builder
class HeadBuilder:
    def __init__(self, branch="LIVE"):
        self.branch = branch
        self.p4k = None
        self.lock = threading.Lock()
        self.genders = {}   # gender -> dict(base, uv, idx, heads[list], rig_id, masks)
        self.mesh_cache = {}

    def _p(self):
        if self.p4k is None:
            if not gamedata.SC_ROOT:
                raise RuntimeError("Star Citizen install folder not set")
            self.p4k = P4K(os.path.join(gamedata.SC_ROOT, self.branch, "Data.p4k"))
        return self.p4k

    def _find(self, suffix):
        """Find an archive path ending with suffix (case-insensitive)."""
        p = self._p()
        if not hasattr(p, "_lower"):
            p._lower = p.cd.lower()
        key = suffix.lower().replace("/", "\\").encode()
        k = p._lower.find(key)
        while k >= 0:
            h = p.cd.rfind(b"PK\x01\x02", 0, k)
            nlen, = struct.unpack_from("<H", p.cd, h + 28)
            name_end = h + 46 + nlen
            if name_end == k + len(key):
                return p.cd[h + 46:name_end].decode("utf-8", "replace")
            k = p._lower.find(key, k + 1)
        return None

    def _mesh(self, path):
        if path not in self.mesh_cache:
            self.mesh_cache[path] = read_ivo_mesh(self._p().read(path))
        return self.mesh_cache[path]

    def gender(self, g):
        with self.lock:
            if g in self.genders:
                return self.genders[g]
            proto = "protos_human_%s_face_t1_pu" % g
            folder = "%s\\%s\\pu\\%s\\%s" % (HEADS, g, proto, proto)
            dna = self._p().read(folder + "_head.dna")
            rig_id = dna[16:20].hex()
            # stock head names: fixed-size records; locate them from the first two archetype names
            first = dna.find(b"%s_archetype_v001_t" % g.encode(), 0, 65536)
            second = dna.find(b"%s_archetype_v002_t" % g.encode(), first + 1, 65536)
            if first < 0 or second < 0:
                raise ValueError("couldn't find the stock head list in the %s rig file" % g)
            stride = second - first
            names = []
            o = first
            while True:
                nm = dna[o:o + stride].split(b"\x00")[0].decode("ascii", "replace")
                if not re.match(r"^[a-z0-9_]+_t\d$", nm):
                    break
                names.append(nm)
                o += stride
            base, uv, idx = self._mesh(folder + "_head.skinm")
            masks = self._region_masks(g, base, uv)
            try:
                eyes = self._mesh(folder + "_eyes.skinm")
            except Exception:
                eyes = None
            self.genders[g] = {"rig": rig_id, "names": names, "base": base, "uv": uv, "idx": idx,
                               "masks": masks, "paths": {}, "eyes": eyes,
                               "skin_tex": self._skin_texture_path(folder + "_head_material.mtl")}
            return self.genders[g]

    def _skin_texture_path(self, mtl_path):
        from .gamedata import parse_cryxml
        try:
            root = parse_cryxml(self._p().read(mtl_path))
            for t in root.iter("Texture"):
                if t.get("Map") == "TexSlot1" and t.get("File"):
                    f = "Data\\" + t.get("File").replace("/", "\\")
                    return re.sub(r"\.tif$", ".dds", f, flags=re.I)
        except Exception:
            pass
        return None

    # the head material's other maps: TexSlot2 normal (BC5 "ddna"), TexSlot4 specular, TexSlot12 translucency
    SKIN_MAPS = {"normal": "TexSlot2", "spec": "TexSlot4", "trans": "TexSlot12"}

    def _skin_map_path(self, g, slot):
        from .gamedata import parse_cryxml
        proto = "protos_human_%s_face_t1_pu" % g
        mtl = "%s\\%s\\pu\\%s\\%s_head_material.mtl" % (HEADS, g, proto, proto)
        try:
            for t in parse_cryxml(self._p().read(mtl)).iter("Texture"):
                if t.get("Map") == slot and t.get("File"):
                    return "Data\\" + re.sub(r"\.tif$", ".dds", t.get("File"), flags=re.I).replace("/", "\\")
        except Exception:
            pass
        return None

    # The customizer's makeup masks (shared by both heads, same UV layout as the skin texture):
    # (file, channel) for eye shadow, blush and lipstick, plus the foundation mask for the freckle area
    MAKEUP_MASKS = (("makeup_eyes_02", 2), ("makeup_blush_01", 2), ("makeup_lips_01", 2), ("makeup_foundation_01", 0))

    def makeup_png(self, cache_dir):
        """1024x2048 PNG in head-texture UV space. Top half: R eye makeup, G blush, B lips.
        Bottom half: where freckles go."""
        from PIL import Image
        os.makedirs(cache_dir, exist_ok=True)
        out = os.path.join(cache_dir, "skin_%s_makeupmask2.png" % self.branch)
        if not os.path.exists(out):
            base = "Data\\Objects\\Characters\\Human\\heads\\shared\\makeup\\"
            ch = []
            for name, c in self.MAKEUP_MASKS:
                png = split_dds_to_png(self._p(), base + name + ".dds", 1024)[0]
                im = Image.open(io.BytesIO(png)).convert("RGB").resize((1024, 1024))
                ch.append(np.asarray(im, np.float32)[..., c] / 255.0)
            eyes, blush, lips, fnd = ch
            # freckles: cheeks and nose, i.e. the blush and foundation areas in the band
            # between the lower eyelids and the upper lip
            rows = np.arange(1024, dtype=np.float32)[:, None]
            band = np.clip((rows - 380) / 40, 0, 1) * np.clip((580 - rows) / 40, 0, 1)
            freck = np.clip(1.2 * blush + 1.2 * fnd, 0, 1) * band
            # no alpha channel: browsers premultiply, which would wipe the colour where alpha is 0
            rgb = np.vstack([np.stack([eyes, blush, lips], -1), np.stack([freck] * 3, -1)])
            Image.fromarray((rgb * 255).astype(np.uint8), "RGB").save(out)
        return out

    def skin_png(self, g, cache_dir, kind="diffuse"):
        """Decoded head texture as a cached PNG: kind = diffuse | normal | spec | trans | makeup."""
        if kind == "makeup":
            return self.makeup_png(cache_dir)
        info = self.gender(g)
        path = info.get("skin_tex") if kind == "diffuse" else self._skin_map_path(g, self.SKIN_MAPS[kind])
        if not path:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        out = os.path.join(cache_dir, "skin_%s_%s%s.png" % (self.branch, g, "" if kind == "diffuse" else "_" + kind))
        if not os.path.exists(out):
            size = 2048 if kind == "normal" else 1024
            png, _, _ = split_dds_to_png(self._p(), path, size, normal=(kind == "normal"), mask=(kind == "trans"))
            with open(out, "wb") as fh:
                fh.write(png)
        return out

    def _region_masks(self, g, base, uv):
        """Per-vertex weight for each region from the highlight atlas (red channel)."""
        atlas = read_dds_rgba(self._p().read(ATLAS)).astype(np.float32) / 255.0
        h, w = atlas.shape[:2]
        tiles = w // h
        hl_path = "%s\\highlighting\\protos_human_%s_face_highlighting_head.skinm" % (HEADS, g)
        try:
            hpos, huv, _ = self._mesh(hl_path)
        except Exception:
            hpos = huv = None
        if huv is not None and len(huv) == len(base):
            u, v = huv[:, 0], huv[:, 1]
            self.mask_source = "highlight mesh UVs"
        else:
            # fallback: front projection of the head
            lo, hi = base.min(0), base.max(0)
            u = (base[:, 0] - lo[0]) / (hi[0] - lo[0])
            v = (hi[2] - base[:, 2]) / (hi[2] - lo[2])
            self.mask_source = "front projection"
        u = np.clip(u - np.floor(u), 0, 0.9999)
        v = np.clip(v - np.floor(v), 0, 0.9999)
        px = (u * h).astype(int)
        py = (v * h).astype(int)
        masks = np.zeros((tiles, len(base)), np.float32)
        for t in range(tiles):
            masks[t] = atlas[py, t * h + px, 0]
        # soft edges overlap: never let regions add up to more than 100% at a vertex
        tot = masks.sum(0)
        over = tot > 1.0
        masks[:, over] /= tot[over]
        return masks

    # ------------------------------------------------------------------ hair & co
    def _names(self):
        if not hasattr(self, "_item_names"):
            from .items import ItemNames
            self._item_names = ItemNames(self._p())
        return self._item_names

    def item_guid(self, name):
        """Database ID for an item name (e.g. 'Universal_stubble'), or None."""
        nm = self._names()
        nm.name(b"\0" * 16)
        want = "EntityClassDefinition." + name
        return next((g for g, n in nm.map.items() if n == want), None)

    def palette(self, key):
        """Customizer palette as uint8[H, W, 3] (sRGB), or None if the game doesn't have it."""
        cache = self.__dict__.setdefault("_palettes", {})
        if key not in cache:
            try:
                from PIL import Image
                png, _, _ = split_dds_to_png(self._p(), PALETTES[key], 1024)
                cache[key] = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            except Exception:
                cache[key] = None
        return cache[key]

    def styles(self, g):
        """Hair / brow / beard items that have a model for this body: {kind: [(name, guid)]}."""
        cache = getattr(self, "_styles", {})
        if g in cache:
            return cache[g]
        p = self._p()
        if not hasattr(p, "_lower"):
            p._lower = p.cd.lower()
        # one pass over the archive's file names instead of a search per item
        models = set(m.decode() for m in re.findall(rb"\\([a-z0-9_]+)\.skinm", p._lower))
        nm = self._names()
        nm.name(b"\0" * 16)  # make sure the database index is loaded
        out = {"hair": [], "brows": [], "beard": []}
        for guid, full in nm.map.items():
            if not full.startswith("EntityClassDefinition."):
                continue
            n = full.split(".", 1)[1]
            low = n.lower()
            kind = "beard" if low.startswith("facial_hair_") else "hair" if low.startswith("hair_") else "brows" if low.startswith("brows_") else None
            if kind and "%s_%s" % (g[0], low) in models and not low.endswith("_prop"):  # _prop = NPC props
                out[kind].append((n, guid))
        for k in out:
            out[k].sort(key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x[0].lower())])
        cache[g] = out
        self._styles = cache
        return out

    def _texture_png(self, path, cache_dir, max_size=1024, normalize=False):
        import hashlib
        key = hashlib.sha1(("mask%s:" % ("n" if normalize else "") + path).lower().encode()).hexdigest()[:16]
        out = os.path.join(cache_dir, "t_%s.png" % key)
        if not os.path.exists(out):
            os.makedirs(cache_dir, exist_ok=True)
            png, _, _ = split_dds_to_png(self._p(), path, max_size, mask=True, normalize=normalize)
            with open(out, "wb") as fh:
                fh.write(png)
        return "t_%s.png" % key

    def _material(self, item, g):
        """Submaterials of an attachment: [{'cap': bool, 'tex': archive path or None}]"""
        from .gamedata import parse_cryxml
        for cand in ("\\%s_%s.mtl" % (g[0], item), "\\sc\\%s\\%s.mtl" % (g, item)):
            path = self._find(cand)
            if path:
                break
        else:
            return []
        root = parse_cryxml(self._p().read(path))
        subs = root.find("SubMaterials")
        mats = list(subs) if subs is not None else [root]
        out = []
        for m in mats:
            tex = None
            for t in m.iter("Texture"):
                if t.get("Map") == "TexSlot1" and t.get("File"):
                    tex = "Data\\" + re.sub(r"\.tif$", ".dds", t.get("File"), flags=re.I).replace("/", "\\")
            gen = (m.get("StringGenMask") or "") + (m.get("GenMask") or "")
            cap = "HAIR_CAP" in gen or "scalp" in (m.get("Name") or "").lower() or "scalp" in (tex or "").lower()
            out.append({"cap": cap, "tex": tex})
        return out

    def attachments(self, raw, g, cache_dir):
        """Hair, beard and brow meshes for the items worn in this character file."""
        from .items import chf_items
        names = [self._names().name(x) or "" for x in chf_items(raw)]
        blocks = parse_hair_blocks(raw)
        kinds = []
        for n in names:
            low = n.lower()
            if low.startswith("hair_"):
                kinds.append(("hair", n))
            elif low.startswith("facial_hair_"):
                kinds.append(("beard", n))
            elif low.startswith("brows_"):
                kinds.append(("brows", n))
        # colour blocks run in attachment order: hair first, brows last, beard in between
        colour = {}
        graph = self.palette("hair")
        if blocks:
            colour["hair"] = hair_colour(blocks[0], graph)
            # brows often carry a leftover (unused) dye; the game shows them in natural colour
            colour["brows"] = hair_colour(dict(blocks[-1], dyeAmount=0.0), graph)
            colour["beard"] = hair_colour(blocks[1] if len(blocks) >= 3 else blocks[0], graph)
        out = []
        # the customizer's "stubble only" item: a shell of fine strand cards over the jaw
        if g == "male" and any(n.lower() == "universal_stubble" for n in names):
            try:
                pos, uv, idx = read_ivo_mesh(self._p().read(STUBBLE_MESH))
                out.append({"kind": "stubble", "item": "Universal_stubble", "colour": colour.get("beard", colour.get("hair", "#3a2a20")),
                            "positions": pos, "uvs": uv, "indices": idx,
                            "parts": [{"first": 0, "count": len(idx), "cap": False, "tex": None}]})
            except Exception:
                pass
        for kind, item in kinds:
            path = self._find("\\%s_%s.skinm" % (g[0], item))
            if not path:
                continue
            data = self._p().read(path)
            pos, uv, idx = read_ivo_mesh(data)
            mats = self._material(item, g)
            parts = []
            for mat, first, count in read_ivo_subsets(data):
                info = mats[mat] if mat < len(mats) else {"cap": False, "tex": None}
                tex = None
                if info["tex"]:
                    try:
                        # brow shape masks peak around 50% by design; stretch them so brows read clearly
                        tex = self._texture_png(info["tex"], cache_dir, 512, normalize=(kind == "brows" and info["cap"]))
                    except Exception:
                        tex = None
                parts.append({"first": first, "count": count, "cap": info["cap"], "tex": tex})
            out.append({"kind": kind, "item": item, "colour": colour.get(kind, "#3a2a20"),
                        "positions": pos, "uvs": uv, "indices": idx, "parts": parts})
        return names, out

    # ------------------------------------------------------------------ photo fitting
    def all_head_meshes(self, g):
        """positions of every stock head, stacked: float32[H, N, 3] (None rows if missing)."""
        info = self.gender(g)
        if "stack" not in info:
            rows, avail = [], []
            for name in info["names"]:
                path = self.head_path(g, name)
                p = self._mesh(path)[0] if path else None
                ok = p is not None and p.shape == info["base"].shape
                rows.append(p if ok else info["base"])
                avail.append(ok)
            info["stack"] = np.stack(rows)
            info["available"] = np.array(avail)  # e.g. imperator_t1 has no standalone mesh
        return info["stack"]

    def landmarks(self, g, lmap):
        """Landmark positions for base + every stock head from a barycentric landmark map
        [{'tri': [a, b, c], 'bary': [u, v, w]} | None] -> (base[L,3], heads[H,L,3], valid mask)."""
        info = self.gender(g)
        valid = np.array([m is not None for m in lmap])
        tri = np.array([m["tri"] if m else [0, 0, 0] for m in lmap])
        bary = np.array([m["bary"] if m else [1, 0, 0] for m in lmap], np.float32)
        def interp(P):
            return (P[..., tri[:, 0], :] * bary[:, 0, None] + P[..., tri[:, 1], :] * bary[:, 1, None]
                    + P[..., tri[:, 2], :] * bary[:, 2, None])
        return interp(info["base"]), interp(self.all_head_meshes(g)), valid

    def landmark_regions(self, g, lmap):
        """Region membership per landmark from the region masks at its triangle."""
        info = self.gender(g)
        out = []
        for m in lmap:
            if not m:
                out.append(None); continue
            w = sum(info["masks"][:, v] * b for v, b in zip(m["tri"], m["bary"]))
            out.append(w)
        return out

    def head_path(self, g, name):
        info = self.gender(g)
        if name not in info["paths"]:
            info["paths"][name] = self._find("\\%s_head.skinm" % name)
        return info["paths"][name]

    def build(self, chf_bytes, cache_dir=None):
        raw = chf_payload(chf_bytes)
        dna = parse_dna_block(raw)
        g = None
        for cand in ("male", "female"):
            if self.gender(cand)["rig"] == dna["rig"]:
                g = cand
        if g is None:
            # older files carry an older rig id; guess from body id used by the male default
            g = "male" if dna["body"] == "9493d0fcf6676cdd" else "female"
        info = self.gender(g)
        base = info["base"]
        pos = base.copy()
        used, missing = [], []
        delta_cache = {}
        nreg = min(len(dna["blend"]), len(info["masks"]))
        for r in range(nreg):
            row = dna["blend"][r]
            if not row:
                continue
            tot = sum(w for _, w in row)
            scale = 1.0 / tot if tot > 1.0 else 1.0
            d = np.zeros_like(base)
            for head, w in row:
                if head >= len(info["names"]):
                    missing.append("head #%d (not in this game version)" % head); continue
                name = info["names"][head]
                if name not in delta_cache:
                    path = self.head_path(g, name)
                    if not path:
                        missing.append(name); delta_cache[name] = None; continue
                    hp, _, _ = self._mesh(path)
                    delta_cache[name] = (hp - base) if hp.shape == base.shape else None
                    if delta_cache[name] is None:
                        missing.append(name)
                if delta_cache[name] is not None:
                    d += delta_cache[name] * (w * scale)
                    used.append({"region": REGIONS[r], "head": name, "weight": round(w * scale, 3)})
            pos += d * info["masks"][r][:, None]
        # eyes are a separate mesh: move each eyeball with the average shift of its eye region
        eyes = None
        if info["eyes"] is not None:
            ep, euv, eidx = info["eyes"]
            ep = ep.copy()
            shift = pos - base
            for r, side in ((2, None), (3, None)):
                m = info["masks"][r]
                if m.sum() < 1e-6:
                    continue
                centre = (base * m[:, None]).sum(0) / m.sum()
                avg = (shift * m[:, None]).sum(0) / m.sum()
                # eyeball verts on the same side as this region's centre
                sel = np.sign(ep[:, 0]) == np.sign(centre[0])
                ep[sel] += avg
            eyes = {"positions": ep, "uvs": euv, "indices": eidx}
        items, attach = [], []
        if cache_dir:
            try:
                items, attach = self.attachments(raw, g, cache_dir)
            except Exception as e:  # hair is a bonus; never break the head
                missing.append("hair (%s)" % e)
        return {"gender": g, "positions": pos, "uvs": info["uv"], "indices": info["idx"], "eyes": eyes,
                "items": items, "attachments": attach,
                "colours": parse_colours(raw),
                "used": used, "missing": sorted(set(map(str, missing))), "maskSource": self.mask_source,
                "dna": dna}
