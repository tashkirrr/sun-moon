#!/usr/bin/env python3
"""Sun & Moon: an ASCII animation for Ramisha.

    python sunmoon.py                    play in your terminal (Ctrl+C to stop)
    python sunmoon.py --gif out.gif      render one loop to a GIF
    python sunmoon.py --still out.png    save one frame (--at 0..1 picks the moment)
"""
import argparse
import math
import shutil
import sys
import time

import numpy as np

NAME = "Ramisha"
PERIOD = 16.0                      # seconds for one full loop
TAU = 2 * math.pi
SUN_RAMP = np.array(list(".:;ilt1fYJCT7OQ08@"))
SUN_STOPS = [0.0, 0.4, 0.7, 1.0]
SUN_RGB = np.array([(150, 30, 10), (235, 100, 25), (255, 185, 60), (255, 245, 200)], float)
MOON_STOPS = [0.0, 0.35, 0.7, 1.0]
MOON_RGB = np.array([(15, 25, 80), (40, 80, 210), (110, 160, 255), (215, 230, 255)], float)


def ramp(b, stops, rgb):
    return np.stack([np.interp(b, stops, rgb[:, i]) for i in range(3)], axis=-1)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Scene:
    def __init__(self, cols, rows, aspect=2.0, seed=7):
        self.cols, self.rows = cols, rows
        rng = np.random.default_rng(seed)
        self.X, self.Y = np.meshgrid(np.arange(cols) + 0.5, (np.arange(rows) + 0.5) * aspect)
        self.W, self.H = cols, rows * aspect
        self.R = min(self.H * 0.36, self.W * 0.2)
        self.d = (self.W / 2 - self.R) * 0.92
        self.rnd = rng.random((rows, cols))
        self.thr = rng.random((rows, cols))
        self.odd = (np.arange(rows) % 2 == 1)[:, None]
        n = max(20, cols * rows // 45)
        self.s_r = rng.integers(0, rows, n)
        self.s_c = rng.integers(0, cols, n)
        self.s_ph = rng.random(n) * TAU
        self.s_k = rng.integers(1, 5, n)
        self.s_big = rng.random(n) < 0.22
        self.craters = [(-.35, -.25, .28, .45), (.30, .20, .22, .40), (.05, .55, .18, .35),
                        (-.10, -.60, .15, .30), (.55, -.30, .14, .30), (-.55, .40, .16, .30)]

    def _uv(self, cx, cy, R):
        u, v = (self.X - cx) / R, (self.Y - cy) / R
        r2 = u * u + v * v
        return u, v, r2 <= 1, np.sqrt(np.clip(1 - r2, 0, 1))

    def sun(self, tau, cx, cy, R):
        u, v, inside, nz = self._uv(cx, cy, R)
        n = 0.5 + 0.25 * np.sin(5 * u + 2 * tau) * np.sin(4 * v - tau) + 0.25 * np.sin(7 * (u + v) + 3 * tau)
        b = np.clip(0.30 + 0.70 * nz ** 0.45 + 0.35 * (n - 0.5), 0, 1)
        idx = np.clip(((b * 0.98 + (self.rnd - 0.5) * 0.35) * (len(SUN_RAMP) - 1)).astype(int), 0, len(SUN_RAMP) - 1)
        return inside, SUN_RAMP[idx], ramp(b, SUN_STOPS, SUN_RGB)

    def moon(self, tau, cx, cy, R, lx, ly):
        u, v, inside, nz = self._uv(cx, cy, R)
        L = np.array([lx, ly, 1.6])
        L /= np.linalg.norm(L)
        lit = np.clip(u * L[0] + v * L[1] + nz * L[2], 0, 1)
        crat = sum(dep * np.exp(-1.5 * ((u - a) ** 2 + (v - b) ** 2) / (r * r)) for a, b, r, dep in self.craters)
        b = np.clip(0.22 + 0.78 * lit * (1 - np.clip(crat, 0, 0.7)), 0, 1)
        b = b * np.where(self.odd, 0.78, 1.0)
        ch = np.where(b < 0.12, ".", np.where(self.odd, "-", "="))
        return inside, ch, ramp(b, MOON_STOPS, MOON_RGB)

    def frame(self, tau):
        s, c = math.sin(tau), math.cos(tau)
        cx, cy = self.W / 2, self.H / 2
        bob = 0.05 * self.H * math.sin(2 * tau)
        sx, sy, mx, my = cx - self.d * c, cy + bob, cx + self.d * c, cy - bob
        Rs, Rm = self.R * (1 - 0.07 * s), self.R * (1 + 0.07 * s)
        inS, chS, rgbS = self.sun(tau, sx, sy, Rs)
        inM, chM, rgbM = self.moon(tau, mx, my, Rm, (sx - mx) / Rm, (sy - my) / Rm)

        # glow around each body
        flick = 0.9 + 0.1 * math.sin(3 * tau)
        gS = np.where(np.hypot(self.X - sx, self.Y - sy) / Rs > 1, np.exp(-np.clip(np.hypot(self.X - sx, self.Y - sy) / Rs - 1, 0, 9) * 3.2), 0) * 0.55 * flick
        gM = np.where(np.hypot(self.X - mx, self.Y - my) / Rm > 1, np.exp(-np.clip(np.hypot(self.X - mx, self.Y - my) / Rm - 1, 0, 9) * 3.2), 0) * 0.45
        g = np.maximum(gS, gM)
        ch = np.where(g > 0.28, ":", np.where(g > 0.07, ".", " ")).astype("<U1")
        rgb = (gS[..., None] * np.array([255, 150, 50]) + gM[..., None] * np.array([90, 130, 255])) * 0.9
        rgb = np.clip(rgb, 0, 255)

        # stars
        cover = inS | inM
        for r, cc, ph, k, big in zip(self.s_r, self.s_c, self.s_ph, self.s_k, self.s_big):
            b = 0.5 + 0.5 * math.sin(k * tau + ph)
            if b < 0.25 or cover[r, cc]:
                continue
            col = 60 + 195 * b
            tint = (col * 0.85, col * 0.9, col)
            ch[r, cc] = "+" if big and b > 0.8 else ("*" if b > 0.6 else ".")
            rgb[r, cc] = tint
            if big and b > 0.8:
                for dr, dc, glyph in ((-1, 0, "|"), (1, 0, "|"), (0, -1, "-"), (0, 1, "-"), (0, -2, "-"), (0, 2, "-")):
                    rr, c2 = r + dr, cc + dc
                    if 0 <= rr < self.rows and 0 <= c2 < self.cols and not cover[rr, c2]:
                        ch[rr, c2] = glyph
                        rgb[rr, c2] = np.array(tint) * 0.55

        # sun and moon overlap: the nearer one wins, with a dithered edge
        score = (np.hypot(self.X - sx, self.Y - sy) - np.hypot(self.X - mx, self.Y - my)) / self.R * 1.2 + 0.9 * s
        use_moon = np.where(inS & inM, self.thr < smoothstep(-0.8, 0.8, score), inM)
        ch = np.where(cover, np.where(use_moon, chM, chS), ch)
        rgb = np.where(cover[..., None], np.where(use_moon[..., None], rgbM, rgbS), rgb)
        return ch, np.clip(rgb, 0, 255).astype(np.uint8)


def to_ansi(ch, rgb):
    q = (rgb >> 3 << 3).tolist()
    lines = []
    for r, row in enumerate(ch.tolist()):
        out, last = [], None
        for c, g in enumerate(row):
            if g == " ":
                out.append(" ")
                continue
            if q[r][c] != last:
                last = q[r][c]
                out.append("\x1b[38;2;%d;%d;%dm" % tuple(last))
            out.append(g)
        lines.append("".join(out))
    return "\x1b[H" + "\x1b[0m\n".join(lines) + "\x1b[0m"


def play(fps=24):
    sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J")
    scene, size = None, None
    caption = "for " + NAME
    try:
        start = time.time()
        while True:
            t0 = time.time()
            cols, rows = shutil.get_terminal_size((100, 32))
            rows -= 1
            if (cols, rows) != size:
                size, scene = (cols, rows), Scene(cols, rows)
                sys.stdout.write("\x1b[2J")
            ch, rgb = scene.frame(TAU * ((t0 - start) % PERIOD) / PERIOD)
            pad = " " * max(0, (cols - len(caption)) // 2)
            sys.stdout.write(to_ansi(ch, rgb) + "\n\x1b[38;2;150;150;170m" + pad + caption + "\x1b[0m")
            sys.stdout.flush()
            time.sleep(max(0, 1 / fps - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\x1b[?1049l")


def _font():
    from PIL import ImageFont
    for p in ("DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "Menlo.ttc", "Consolas.ttf", "Courier New.ttf"):
        try:
            return ImageFont.truetype(p, 14)
        except OSError:
            continue
    return ImageFont.load_default()


def to_image(ch, rgb, font):
    from PIL import Image, ImageDraw
    adv = font.getlength("M")
    rows, cols = ch.shape
    pad = 12
    img = Image.new("RGB", (int(cols * adv + 2 * pad), int(rows * adv * 2 + 2 * pad)), (4, 5, 10))
    d = ImageDraw.Draw(img)
    for r, row in enumerate(ch.tolist()):
        for c, g in enumerate(row):
            if g != " ":
                d.text((pad + c * adv, pad + r * adv * 2 - 3), g, font=font, fill=tuple(int(x) for x in rgb[r, c]))
    return img


def make_gif(path, cols=80, rows=30, fps=15):
    from PIL import Image
    scene, font = Scene(cols, rows), _font()
    n = int(PERIOD * fps)
    frames = [to_image(*scene.frame(TAU * i / n), font) for i in range(n)]
    sample = Image.new("RGB", (frames[0].width, frames[0].height * 4))
    for k in range(4):
        sample.paste(frames[k * n // 4], (0, k * frames[0].height))
    pal = sample.quantize(256)
    frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0, optimize=True)
    print("wrote", path, "(%d frames)" % n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gif", metavar="FILE")
    ap.add_argument("--still", metavar="FILE")
    ap.add_argument("--at", type=float, default=0.0, help="loop position for --still, 0 to 1")
    a = ap.parse_args()
    if a.gif:
        make_gif(a.gif)
    elif a.still:
        ch, rgb = Scene(80, 30).frame(TAU * a.at)
        to_image(ch, rgb, _font()).save(a.still)
    else:
        if sys.platform == "win32":
            import os
            os.system("")          # turns on ANSI colors in the Windows console
        play()


if __name__ == "__main__":
    main()
