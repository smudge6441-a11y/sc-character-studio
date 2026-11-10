"""Fit a Star Citizen face recipe to landmarks detected in a photo.

The game builds a face per region as a weighted mix of up to 4 stock heads (weights sum to 1).
We have the same 478 MediaPipe landmarks on the photo and (via a one-off barycentric map made by
running MediaPipe on a render of the base head) on every stock head. Per region we solve
    min_w || sum_h w_h * L_h[R] - photo[R] ||^2   with  w >= 0, sum w = 1
on the frontal 2D projection, then keep the 4 strongest heads.
"""
import numpy as np

from .heads import REGIONS, hair_colour

PAIRS = [(0, 1), (2, 3), (4,), (5, 6), (7, 8), (9,), (10,), (11,), (12,)]


def _procrustes(src, dst):
    """Similarity transform (s, R, t) mapping src points onto dst (no reflection)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, sig, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    D = np.diag([1, d])
    R = (u @ D @ vt).T
    s = (sig * np.diag(D)).sum() / (a ** 2).sum()
    t = md - s * (R @ ms)
    return s, R, t


def _project_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)


def _nnls(A, b, max_iter=500):
    """Lawson-Hanson non-negative least squares: min ||A x - b|| with x >= 0."""
    m, n = A.shape
    P = np.zeros(n, bool)
    x = np.zeros(n)
    w = A.T @ (b - A @ x)
    for _ in range(max_iter):
        if P.all() or w[~P].max(initial=-np.inf) <= 1e-12:
            break
        j = np.argmax(np.where(P, -np.inf, w))
        P[j] = True
        while True:
            z = np.zeros(n)
            z[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (z[P] > 0).all():
                x = z
                break
            neg = P & (z <= 0)
            alpha = np.min(x[neg] / (x[neg] - z[neg] + 1e-30))
            x = x + alpha * (z - x)
            P &= x > 1e-12
        w = A.T @ (b - A @ x)
    return x


def _simplex_lsq(H, T, ridge=0.0):
    """min ||H^T w - T||^2 + ridge*||w||^2 with w >= 0 and sum w = 1 (sum enforced by a heavy row).
    H: [K, D] (each row a head's landmark offsets from the base), T: [D]."""
    K = H.shape[0]
    scale = np.abs(H).max() + 1e-12
    A = np.vstack([H.T / scale, np.full((1, K), 100.0)])
    b = np.concatenate([T / scale, [100.0]])
    if ridge:
        A = np.vstack([A, np.sqrt(ridge) * np.eye(K)])
        b = np.concatenate([b, np.zeros(K)])
    w = _nnls(A, b)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def align(builder, g, lmap, photo_px):
    """Similarity transform from photo pixels to the base head's front view:
    (sx, s, R, t) with  (sx * x_game, -z_game) = s R p + t."""
    base3, _, valid = builder.landmarks(g, lmap)
    photo = np.asarray(photo_px, np.float64)
    ok = valid & np.isfinite(photo).all(1)
    best = None
    for sx in (-1.0, 1.0):
        b2 = np.stack([sx * base3[:, 0], -base3[:, 2]], 1).astype(np.float64)
        s, R, t = _procrustes(photo[ok], b2[ok])
        err = np.linalg.norm((s * (R @ photo[ok].T)).T + t - b2[ok], axis=1).mean()
        if best is None or err < best[0]:
            best = (err, (sx, s, R, t))
    return best[1]


# MediaPipe face-mesh indices used to read the expression
LIP_TOP_IN, LIP_BOT_IN, CORNER_L, CORNER_R, BROW_MID, CHIN = 13, 14, 61, 291, 10, 152
CHEEK_L, CHEEK_R = 234, 454
# lips (outer + inner contours) and the chin/lower jaw: these move when someone smiles or talks
LIPS = {0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191,
        267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415}
LOWER_FACE = {17, 18, 83, 84, 85, 86, 152, 148, 149, 150, 175, 176, 199, 200, 201, 208, 313, 314,
              315, 316, 377, 378, 379, 396, 400, 421, 428}
MOUTH_REGION = REGIONS.index("MOUTH_C")
JAW_REGION = REGIONS.index("JAWLINE_C")
# thresholds from test photos: neutral 0.021 lift / 0.001 gap / 0.40 width;
# closed smile 0.039 lift; toothy smiles 0.05-0.12 gap and 0.48-0.53 width
SMILE_AT = 0.035   # mouth-corner lift / face height
WIDE_AT = 0.45     # mouth width / face width (a smile stretches the mouth)
OPEN_AT = 0.025    # lip gap / face height


def expression(photo_px):
    """How much the photo is smiling / has the mouth open, from the landmarks (face-height units).
    The game's stock heads are all neutral, so an expression would be fitted as a different shape."""
    p = np.asarray(photo_px, np.float64)
    face = np.linalg.norm(p[BROW_MID] - p[CHIN]) or 1.0
    up = (p[CHIN] - p[BROW_MID]) / face                  # "down the face" direction (handles head tilt)
    centre = (p[LIP_TOP_IN] + p[LIP_BOT_IN]) / 2
    corners = (p[CORNER_L] + p[CORNER_R]) / 2
    smile = float((centre - corners) @ up / face)        # corners above the lip centre -> smiling
    opened = float(np.linalg.norm(p[LIP_BOT_IN] - p[LIP_TOP_IN]) / face)
    width = float(np.linalg.norm(p[CORNER_R] - p[CORNER_L]) / (np.linalg.norm(p[CHEEK_R] - p[CHEEK_L]) or 1.0))
    return {"smile": round(smile, 4), "open": round(opened, 4), "width": round(width, 3),
            "smiling": smile > SMILE_AT or width > WIDE_AT, "mouthOpen": opened > OPEN_AT}


def fit_regions(builder, g, lmap, photo_px, expr=None):
    """photo_px: [L, 2] pixel coords (x right, y down) for the same landmarks as lmap.
    expr: from expression(). A smiling or open mouth gets an average, neutral mouth (the mix of
    stock heads closest to the base head), and an open mouth also leaves the chin out of the jaw
    fit; alignment and the whole-face fit (ears/crown/neck) ignore the lips then too."""
    base3, heads3, valid = builder.landmarks(g, lmap)
    member = builder.landmark_regions(g, lmap)
    photo = np.asarray(photo_px, np.float64)
    ok = valid & np.isfinite(photo).all(1)
    expr = expr or {}
    expressive = expr.get("smiling") or expr.get("mouthOpen")
    moving = set()
    if expressive:
        moving |= LIPS
    if expr.get("mouthOpen"):
        moving |= LOWER_FACE
    steady = ok.copy()
    steady[[i for i in moving if i < len(steady)]] = False

    best = None
    for sx in (-1.0, 1.0):  # which way game X runs in the image; pick whichever fits better
        b2 = np.stack([sx * base3[:, 0], -base3[:, 2]], 1).astype(np.float64)
        s, R, t = _procrustes(photo[steady], b2[steady])   # align on the parts an expression doesn't move
        ph = (s * (R @ photo.T)).T + t
        err = np.linalg.norm(ph[steady] - b2[steady], axis=1).mean()
        if best is None or err < best[0]:
            best = (err, sx, ph, b2)
    _, sx, ph, b2 = best
    h2 = np.stack([sx * heads3[:, :, 0], -heads3[:, :, 2]], 2).astype(np.float64)  # [H, L, 2]
    # only heads we can actually show (a head with no standalone mesh would look different in game)
    cand = np.nonzero(builder.gender(g)["available"])[0]

    blend = [[] for _ in REGIONS]
    detail = {}
    all_idx = [i for i in range(len(member)) if steady[i]]
    for group in PAIRS:
        neutral = expressive and MOUTH_REGION in group
        use = ok if neutral else steady   # the neutral mouth targets the base head, so every mouth landmark helps
        idx = [i for i, m in enumerate(member) if m is not None and use[i] and max(m[r] for r in group) > 0.35]
        if len(idx) < 4:
            # ears, crown and neck aren't tracked by the face detector: use the whole-face fit there
            idx = all_idx
        # work in offsets from the base head: sum w = 1 makes this the same objective, but well-conditioned
        H = (h2[cand][:, idx, :] - b2[None, idx, :]).reshape(len(cand), -1)
        T = (ph[idx] - b2[idx]).reshape(-1)
        if neutral:
            # a smile can't be told apart from full lips, and the game's default mouths are fuller
            # than average (the female default especially): aim for the average, neutral mouth
            T = np.zeros_like(T)
        w = _simplex_lsq(H, T, ridge=1e-3)
        top = np.argsort(w)[::-1][:4]
        w4 = _simplex_lsq(H[top], T)
        keep = [(int(cand[top[k]]), float(w4[k])) for k in range(len(top)) if w4[k] > 0.02]
        tot = sum(x for _, x in keep) or 1.0
        keep = [(h, x / tot) for h, x in keep]
        for r in group:
            blend[r] = keep
        resid_before = float(np.linalg.norm(b2[idx] - ph[idx], axis=1).mean())
        fitted = (w4[:, None] * H[top]).sum(0).reshape(-1, 2) + b2[idx]
        resid_after = float(np.linalg.norm(fitted - ph[idx], axis=1).mean())
        if neutral:
            detail[REGIONS[MOUTH_REGION]] = {"neutral": "smiling or open mouth in the photo"}
            continue
        detail["+".join(REGIONS[r] for r in group)] = {"landmarks": len(idx), "before": resid_before, "after": resid_after}
    return blend, detail


def _lin(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(lin):
    lin = np.clip(lin, 0, 1)
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return tuple(int(round(v * 255)) for v in s)


def _lab(rgb):
    """sRGB 0-255 (..., 3) -> CIE Lab (D65)."""
    lin = _lin(rgb)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _nearest(palette, target, l_weight=1.0, step=2):
    """Nearest palette pixel to target (perceptual). Returns (rgb tuple, (x 0..1, y 0..1))."""
    p = palette[::step, ::step, :3].astype(float)
    d = _lab(p) - _lab(np.asarray(target, float))
    d[..., 0] *= l_weight
    dist = (d ** 2).sum(-1)
    y, x = np.unravel_index(np.argmin(dist), dist.shape)
    h, w = p.shape[:2]
    return tuple(int(v) for v in p[y, x]), (x / max(1, w - 1), y / max(1, h - 1))


def game_makeup(makeup, builder=None):
    """{"lips"|"eyes"|"cheeks": {"colour": (r, g, b), "opacity": 0..1}} -> same, with each colour
    snapped to the customizer's makeup palette (the game stores a pixel of it)."""
    pal = builder.palette("makeup") if builder is not None else None
    out = {}
    for area, v in (makeup or {}).items():
        if area not in ("lips", "eyes", "cheeks") or not isinstance(v, dict):
            continue
        col, op = v.get("colour"), v.get("opacity")
        if col is not None and pal is not None:
            col = _nearest(pal, col, step=1)[0]
        out[area] = {"colour": tuple(int(c) for c in col) if col is not None else None,
                     "opacity": None if op is None else float(min(1.0, max(0.0, op)))}
    return out


def game_colours(photo_skin, photo_iris, photo_hair, builder=None):
    """Photo colours (sRGB 0-255) -> values the game's customizer can actually produce.
    With a builder, each colour snaps to the customizer's own palette texture:
      skin -> skin_tone_lut, iris -> eye_color_graph (brightness weighted down, hue matters most),
      hair -> root_color_graph whose x/y ARE melanin/redness."""
    out = {}
    pal = (lambda k: builder.palette(k)) if builder is not None else (lambda k: None)
    if photo_skin is not None:
        sk = pal("skin")
        out["skin"] = _nearest(sk, photo_skin)[0] if sk is not None else _srgb(_lin(photo_skin))
    if photo_iris is not None:
        ey = pal("eyes")
        # irises photograph darker than they look (shaded by the brow, plus the dark limbal ring),
        # so lift the sample before matching it against the palette
        lifted = _srgb(np.clip(_lin(photo_iris) * 2.2, 0, 1))
        if ey is not None:
            lab = _lab(np.asarray(lifted, float))
            if np.hypot(lab[1], lab[2]) < 10:
                # near-colourless iris: photos wash out light eyes; "grey" eyes are nearly always
                # blue-grey (confirmed by a user picking #6d8baf in-game for such a sample)
                target = np.array([lab[0] + 8, -4.0, -16.0])
                d = _lab(ey[::2, ::2, :3].astype(float)) - target
                d[..., 0] *= 0.5
                y, x = np.unravel_index(np.argmin((d ** 2).sum(-1)), d.shape[:2])
                out["iris"] = tuple(int(v) for v in ey[::2, ::2][y, x])
            else:
                out["iris"] = _nearest(ey, lifted, l_weight=0.5)[0]
        else:
            out["iris"] = _srgb(_lin(photo_iris) * 0.6)
    if photo_hair is not None and pal("hair") is not None:
        _, (m, r) = _nearest(pal("hair"), photo_hair)
        out["hair"] = {"melanin": float(m), "redness": float(r)}
        return out
    if photo_hair is not None:
        target = _lin(photo_hair)
        best = None
        for m in np.linspace(0, 1, 41):
            for r in np.linspace(0, 1, 21):
                c = _lin(np.array([int(hair_colour({"melanin": m, "redness": r})[i:i + 2], 16) for i in (1, 3, 5)]))
                d = ((c - target) ** 2).sum()
                if best is None or d < best[0]:
                    best = (d, m, r)
        out["hair"] = {"melanin": float(best[1]), "redness": float(best[2])}
    return out
