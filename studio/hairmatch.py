"""Suggest hair styles from a photo.

Every hair style gets a front-view opacity map on a fixed grid around the base head (game X
across, Z up), built from its strand cards and their alpha textures, with strands hidden behind
the head culled. The photo's hair (MediaPipe multiclass segmentation) is mapped onto the same grid
with the face-landmark alignment, and styles are ranked by how well their solid hair matches it.
"""
import io
import os

import numpy as np

from .heads import read_ivo_mesh, read_ivo_subsets, split_dds_to_png

GRID = 96          # cells per side
MARGIN = 0.07      # metres of space around the base head's bounding box
VERSION = 3       # bump when the silhouette format changes (invalidates caches)


def grid_box(base):
    """(x0, x1, z0, z1) of the grid in game units, from the base head mesh."""
    lo, hi = base.min(0), base.max(0)
    cx = (lo[0] + hi[0]) / 2
    top = hi[2] + MARGIN * 0.6
    size = (hi[0] - lo[0]) + 2 * MARGIN
    size = max(size, (top - lo[2]) * 1.05)
    return cx - size / 2, cx + size / 2, top - size, top


def _bary(n):
    b = [(i / n, j / n, 1 - i / n - j / n) for i in range(n + 1) for j in range(n + 1 - i)]
    return np.array(b, np.float32)


def _cells(pts, box):
    x0, x1, z0, z1 = box
    cx = ((pts[:, 0] - x0) / (x1 - x0) * GRID).astype(int)
    cz = ((z1 - pts[:, 2]) / (z1 - z0) * GRID).astype(int)
    ok = (cx >= 0) & (cx < GRID) & (cz >= 0) & (cz < GRID)
    return cz, cx, ok


def head_front(base, idx, box):
    """Front-most head surface depth (+Y = towards the camera) per grid cell; -inf where no head."""
    pts = np.einsum("sk,tkd->tsd", _bary(4), base[idx.reshape(-1, 3)]).reshape(-1, 3)
    cz, cx, ok = _cells(pts, box)
    d = np.full((GRID, GRID), -np.inf, np.float32)
    np.maximum.at(d, (cz[ok], cx[ok]), pts[ok, 1])
    return d


def _mask_image(p4k, path, cache):
    if path not in cache:
        from PIL import Image
        try:
            png, _, _ = split_dds_to_png(p4k, path, 1024, mask=True)
            cache[path] = np.asarray(Image.open(io.BytesIO(png)).convert("L"), np.float32) / 255.0
        except Exception:
            cache[path] = None
    return cache[path]


def silhouette(builder, g, item, box, tex_cache, front):
    """float16[GRID, GRID] (row 0 = top): how opaque the hair strands look from the front, 0..1.
    Each strand card adds its texture's alpha times its projected area; opacity = 1 - exp(-sum).
    Sparse styles (twists, buzz cuts, wisps) stay see-through, dense short hair reads as solid.
    Hair painted on the scalp is left out: in game it reads as skin with a shadow, not as hair."""
    path = builder._find("\\%s_%s.skinm" % (g[0], item))
    if not path:
        return None
    data = builder._p().read(path)
    pos, uv, idx = read_ivo_mesh(data)
    mats = builder._material(item, g)
    x0, x1, z0, z1 = box
    cell_area = ((x1 - x0) / GRID) * ((z1 - z0) / GRID)
    dens = np.zeros((GRID, GRID), np.float64)
    b = _bary(6)  # dense samples per triangle: hair cards are thin, vertices alone leave gaps
    for mat, first, count in read_ivo_subsets(data):
        info = mats[mat] if mat < len(mats) else {"cap": False, "tex": None}
        if info["cap"]:
            continue
        tri = idx[first:first + count].reshape(-1, 3)
        if not len(tri):
            continue
        P = pos[tri]
        # front-projected (X-Z) area of each card triangle, shared by its samples
        e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
        area = np.abs(e1[:, 0] * e2[:, 2] - e1[:, 2] * e2[:, 0]) / 2 / len(b)
        pts = np.einsum("sk,tkd->tsd", b, P).reshape(-1, 3)
        wgt = np.repeat(area, len(b))
        m = _mask_image(builder._p(), info["tex"], tex_cache) if info["tex"] else None
        if m is not None:
            uvs = np.einsum("sk,tkd->tsd", b, uv[tri]).reshape(-1, 2)
            h, w = m.shape
            u = uvs[:, 0] - np.floor(uvs[:, 0])
            v = uvs[:, 1] - np.floor(uvs[:, 1])
            wgt = wgt * m[np.clip((v * h).astype(int), 0, h - 1), np.clip((u * w).astype(int), 0, w - 1)]
        cz, cx, ok = _cells(pts, box)
        cz, cx, pts, wgt = cz[ok], cx[ok], pts[ok], wgt[ok]
        seen = pts[:, 1] >= front[cz, cx] - 0.02   # hidden behind the face/head otherwise
        np.add.at(dens, (cz[seen], cx[seen]), wgt[seen])
    return (1 - np.exp(-dens / cell_area)).astype(np.float16)


def silhouettes(builder, g, cache_dir):
    """{style name: float16[GRID, GRID] opacity} for every hair style, cached on disk per game version."""
    store = builder.__dict__.setdefault("_hairsil", {})
    if g in store:
        return store[g]
    styles = [n for n, _ in builder.styles(g)["hair"]]
    try:
        stamp = "%d" % int(os.path.getmtime(builder._p().path))
    except (OSError, AttributeError):
        stamp = "0"
    f = os.path.join(cache_dir, "hairsil%d_%s_%s_%s.npz" % (VERSION, builder.branch, g, stamp))
    out = {}
    if os.path.exists(f):
        with np.load(f) as z:
            out = {k: z[k] for k in z.files}
    missing = [s for s in styles if s not in out]
    if missing:
        info = builder.gender(g)
        box = grid_box(info["base"])
        front = head_front(info["base"], info["idx"], box)
        tex = {}
        for s in missing:
            try:
                sil = silhouette(builder, g, s, box, tex, front)
            except Exception:
                sil = None
            if sil is not None:
                out[s] = sil
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(f, **out)
    store[g] = {k: v for k, v in out.items() if k in styles}
    return store[g]


def photo_grid(builder, g, transform, cat, mw, mh):
    """Photo segmentation categories resampled onto the grid -> (cat uint8[G, G], valid bool[G, G]).
    MediaPipe multiclass: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other."""
    sx, s, R, t = transform
    x0, x1, z0, z1 = grid_box(builder.gender(g)["base"])
    xs = x0 + (np.arange(GRID) + 0.5) / GRID * (x1 - x0)
    zs = z1 - (np.arange(GRID) + 0.5) / GRID * (z1 - z0)
    X, Z = np.meshgrid(xs, zs)
    b2 = np.stack([sx * X, -Z], -1).reshape(-1, 2)
    p = ((b2 - t) @ R) / s   # inverse of b2 = s R p + t  (R orthonormal)
    px, py = p[:, 0], p[:, 1]
    valid = (px >= 0) & (px < mw) & (py >= 0) & (py < mh)
    out = np.zeros(len(px), np.uint8)
    out[valid] = cat[py[valid].astype(int), px[valid].astype(int)]
    return out.reshape(GRID, GRID), valid.reshape(GRID, GRID)


def _dilate(m, r):
    for _ in range(r):
        p = np.pad(m, 1)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


SOLID = 0.5      # a style's hair counts where it looks at least this opaque from the front
VOLUME = 4       # cells (~2 cm) a style's hair may stand proud of the photo's hair outline


def score(opacity, cat, valid):
    """F1 of the style's solid hair against the photo's hair (MediaPipe category 1).
    precision: how much of the style lands on (or just beside) hair in the photo - hair over
    bare forehead, background or clothes counts against it. recall: how much of the photo's
    hair the style accounts for."""
    photo = (cat == 1) & valid
    style = (opacity.astype(np.float32) > SOLID) & valid
    if photo.sum() < 15:  # (nearly) no hair in the photo: the barest styles win
        return 1.0 / (1.0 + float(style.sum()) / 30.0)
    if not style.any():
        return 0.0
    prec = (style & _dilate(photo, VOLUME)).sum() / style.sum()
    rec = (photo & _dilate(style, 1)).sum() / photo.sum()
    return float(2 * prec * rec / (prec + rec + 1e-9))


def rank(builder, g, transform, cat, mw, mh, cache_dir, top=8):
    """Hair styles ordered by match with the photo: [(name, score)]."""
    sils = silhouettes(builder, g, cache_dir)
    grid, valid = photo_grid(builder, g, transform, cat, mw, mh)
    res = sorted(((n, float(score(s, grid, valid))) for n, s in sils.items()), key=lambda x: -x[1])
    return res[:top], grid, valid
