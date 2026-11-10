"""Draw the SC Character Studio logo (original artwork): a wireframe head in a hexagon frame.

    python tools/make_logo.py   ->  web/logo.png (256 px, page header + favicon), tools/app.ico (exe icon)
"""
import math
import os

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = 1024                     # drawn large, then scaled down for smooth edges
CYAN = (56, 198, 232)
CYAN_DIM = (56, 198, 232, 110)
ORANGE = (245, 166, 35)
BG_TOP, BG_BOT = (16, 30, 52), (8, 14, 26)


def hexagon(cx, cy, r, rot=math.pi / 6):
    return [(cx + r * math.cos(rot + k * math.pi / 3), cy + r * math.sin(rot + k * math.pi / 3)) for k in range(6)]


def draw():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # rounded-square background with a vertical gradient
    grad = Image.new("RGBA", (S, S))
    gd = ImageDraw.Draw(grad)
    for y in range(S):
        t = y / (S - 1)
        gd.line([(0, y), (S, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOT)) + (255,))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=200, fill=255)
    img.paste(grad, (0, 0), mask)

    d = ImageDraw.Draw(img, "RGBA")
    cx, cy = S / 2, S / 2 + 10
    # hexagon frame with a soft glow
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).line(hexagon(cx, cy, 400) + [hexagon(cx, cy, 400)[0]], fill=CYAN + (160,), width=40, joint="curve")
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(18)))
    d.line(hexagon(cx, cy, 400) + [hexagon(cx, cy, 400)[0]], fill=CYAN + (255,), width=26, joint="curve")

    # faceted (low-poly) head in profile, facing right, with shoulders
    fig = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fig, "RGBA")
    ox, oy, sc = cx - 215, cy - 330, 400               # profile box origin and scale
    P = lambda x, y: (ox + x * sc, oy + y * sc)
    outline = [(0.05, 0.47), (0.2, 0.12), (0.48, 0.0), (0.76, 0.09), (0.88, 0.3), (0.9, 0.42), (0.86, 0.48),
               (0.9, 0.53), (1.02, 0.69), (0.9, 0.73), (0.93, 0.79), (0.88, 0.83), (0.92, 0.87), (0.86, 0.91),
               (0.9, 0.99), (0.76, 1.06), (0.56, 1.03), (0.64, 1.35), (0.26, 1.35), (0.2, 1.02), (0.06, 0.78)]
    fd.ellipse([cx - 340, cy + 190, cx + 340, cy + 660], fill=CYAN + (255,))     # shoulders
    fd.polygon([P(x, y) for x, y in outline], fill=CYAN + (255,))
    # facets: a few inner vertices joined to the outline, like a game head mesh
    ink = BG_BOT + (235,)
    inner = {"a": (0.42, 0.3), "b": (0.66, 0.5), "c": (0.4, 0.68), "d": (0.68, 0.86)}
    edges = [("a", (0.2, 0.12)), ("a", (0.48, 0.0)), ("a", (0.76, 0.09)), ("a", (0.88, 0.3)), ("a", (0.05, 0.47)),
             ("a", "b"), ("a", "c"), ("b", (0.88, 0.3)), ("b", (0.86, 0.48)), ("b", (1.02, 0.69)), ("b", (0.9, 0.73)),
             ("b", "c"), ("b", "d"), ("c", (0.05, 0.47)), ("c", (0.06, 0.78)), ("c", "d"), ("c", (0.56, 1.03)),
             ("d", (0.9, 0.73)), ("d", (0.88, 0.83)), ("d", (0.86, 0.91)), ("d", (0.76, 1.06)), ("d", (0.56, 1.03))]
    pt = lambda v: P(*inner[v]) if isinstance(v, str) else P(*v)
    for u, v in edges:
        fd.line([pt(u), pt(v)], fill=ink, width=8)
    for v in inner:
        x, y = pt(v)
        fd.ellipse([x - 14, y - 14, x + 14, y + 14], fill=ink)
    # eye
    ex, ey = P(0.74, 0.47)
    fd.polygon([(ex - 30, ey), (ex + 8, ey - 14), (ex + 22, ey + 2), (ex - 6, ey + 12)], fill=ink)
    hmask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(hmask).polygon(hexagon(cx, cy, 388), fill=255)
    clipped = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    clipped.paste(fig, (0, 0), Image.composite(fig.getchannel("A"), Image.new("L", (S, S), 0), hmask))
    img.alpha_composite(clipped)

    # four-point spark (the "create" accent), top right
    sx, sy, r = cx + 300, cy - 300, 95
    spark = [(sx, sy - r), (sx + r * 0.22, sy - r * 0.22), (sx + r, sy), (sx + r * 0.22, sy + r * 0.22),
             (sx, sy + r), (sx - r * 0.22, sy + r * 0.22), (sx - r, sy), (sx - r * 0.22, sy - r * 0.22)]
    sg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(sg).polygon(spark, fill=ORANGE + (200,))
    img.alpha_composite(sg.filter(ImageFilter.GaussianBlur(14)))
    d.polygon(spark, fill=ORANGE + (255,))
    return img


def main():
    img = draw()
    img.resize((256, 256), Image.LANCZOS).save(os.path.join(HERE, "web", "logo.png"))
    img.resize((256, 256), Image.LANCZOS).save(os.path.join(HERE, "tools", "app.ico"),
                                               sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote web/logo.png and app.ico")


if __name__ == "__main__":
    main()
