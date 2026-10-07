import math
import os
import sys
import time
import numpy as np
from PIL import Image

# SETTINGS  (edit these, save, run again)

# --- Matching ---------------------------------------------------------------
RESOLUTION     = 150      # image is reduced to RESOLUTION x RESOLUTION pixels
SOLVER         = "auto"  # "auto" | "exact" | "fast"
EXACT_MAX_RES  = 48      # "auto" uses the exact solver up to this resolution
TRAVEL_PENALTY = 0.10    # 0 = match colors only, higher = prefer short moves
COLOR_WEIGHTS  = (1.0, 1.0, 1.0)   # weight of R, G, B in the color error
FAST_ROUNDS    = 3000    # fast solver: more rounds = better match, slower
RANDOM_SEED    = 0       # fast solver is repeatable for the same seed

# --- Animation --------------------------------------------------------------
DURATION       = 5.0     # seconds for one morph
SWIRL          = 1.0     # strength of the swirling flow (0 = none)
SWIRL_SCALE    = 0.7     # swirl size as a fraction of the image (smaller = tighter eddies)
VISCOSITY      = 1.0     # how strongly neighbours move together (0 = independent)
STIFFNESS      = 1.0     # how hard pixels are pulled to their slot
AUTO_PLAY      = True    # start morphing as soon as matching is done
SUBSTEPS       = 2       # physics steps per frame (raise if motion looks jittery)

# --- Window -----------------------------------------------------------------
WINDOW_SIZE    = 800     # window width and height in screen pixels
FPS            = 60
BACKGROUND     = (0, 0, 0)

# --- Output -----------------------------------------------------------------
RECORD_GIF     = False   # save the first morph as an animated GIF
GIF_PATH       = "morph.gif"
GIF_EVERY      = 2       # keep every Nth frame
GIF_SIZE       = 360     # GIF width and height in pixels
SCREENSHOT_DIR = "."     # where the S key saves PNG screenshots

# Image loading

IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.gif *.tif *.tiff"),
               ("All files", "*.*")]


def pick_file(title):
    """Open a file dialog; fall back to typing a path if no GUI is available."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title=title, filetypes=IMAGE_TYPES)
        root.destroy()
        return path
    except Exception:
        return input(f"{title}\nType the image path: ").strip().strip('"').strip("'")


def load_grid(path, n):
    """Center-crop to a square, shrink to n x n, return (n*n, 3) float array."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((n, n), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64).reshape(n * n, 3)

# Matching: which source pixel goes to which target slot

def make_cost(A, B, n, lam, weights):
    """Return cost(i, j) for source pixel i -> target slot j (vectorised)."""
    w = np.asarray(weights, dtype=np.float64)
    norm = 255.0 ** 2 * w.sum()
    X = np.arange(n * n) % n
    Y = np.arange(n * n) // n

    def cost(i, j):
        d = (A[i] - B[j]) ** 2 @ w / norm
        dx = X[i] - X[j]
        dy = Y[i] - Y[j]
        return d + lam * (dx * dx + dy * dy) / (2.0 * n * n)

    return cost


def solve_exact(A, B, n, lam, weights):
    """Mathematically optimal assignment (Hungarian / Jonker-Volgenant)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    w = np.sqrt(np.asarray(weights, dtype=np.float64))
    norm = 255.0 ** 2 * np.sum(np.square(w))
    P = np.stack([np.arange(n * n) % n, np.arange(n * n) // n], axis=1).astype(np.float64)
    C = cdist(A * w, B * w, "sqeuclidean") / norm
    C += lam * cdist(P, P, "sqeuclidean") / (2.0 * n * n)
    rows, cols = linear_sum_assignment(C)
    asg = np.empty(n * n, dtype=np.int64)
    asg[rows] = cols
    return asg


def solve_fast(A, B, n, lam, weights, rounds, seed):
    """Near-optimal: brightness-sorted start, then many parallel pair swaps."""
    M = n * n
    cost = make_cost(A, B, n, lam, weights)
    lum = np.array([0.299, 0.587, 0.114])
    si = np.argsort(A @ lum, kind="stable")
    ti = np.argsort(B @ lum, kind="stable")
    asg = np.empty(M, dtype=np.int64)
    asg[si] = ti
    rng = np.random.default_rng(seed)
    half = M // 2
    step = max(1, rounds // 10)
    for r in range(rounds):
        perm = rng.permutation(M)
        i, k = perm[:half], perm[half:2 * half]      # disjoint pairs
        ji, jk = asg[i], asg[k]
        gain = cost(i, ji) + cost(k, jk) - cost(i, jk) - cost(k, ji)
        s = gain > 1e-12
        asg[i[s]] = jk[s]
        asg[k[s]] = ji[s]
        if (r + 1) % step == 0:
            print(f"  refining... {round(100 * (r + 1) / rounds)}%")
    return asg


def total_cost(A, B, n, lam, weights, asg):
    return float(make_cost(A, B, n, lam, weights)(np.arange(n * n), asg).sum())


def compute_assignment(A, B, n):
    method = SOLVER
    if method == "auto":
        method = "exact" if n <= EXACT_MAX_RES else "fast"
    if method == "exact":
        try:
            print(f"Solving exact optimum for {n}x{n} ({n * n} pixels)...")
            return solve_exact(A, B, n, TRAVEL_PENALTY, COLOR_WEIGHTS)
        except ImportError:
            print("scipy not installed, using the fast solver instead.")
    print(f"Running fast solver for {n}x{n} ({n * n} pixels)...")
    return solve_fast(A, B, n, TRAVEL_PENALTY, COLOR_WEIGHTS, FAST_ROUNDS, RANDOM_SEED)

# Fluid-like simulation

def box3(a, n):
    """Sum over each cell's 3x3 neighbourhood."""
    p = np.pad(a, 1)
    return sum(p[dy:dy + n, dx:dx + n] for dy in range(3) for dx in range(3))


class Sim:
    def __init__(self, n, asg):
        self.n, self.M = n, n * n
        idx = np.arange(self.M)
        self.sx = (idx % n).astype(np.float64)
        self.sy = (idx // n).astype(np.float64)
        self.gx = (asg % n).astype(np.float64)
        self.gy = (asg // n).astype(np.float64)
        self.px, self.py = self.sx.copy(), self.sy.copy()
        self.vx = np.zeros(self.M)
        self.vy = np.zeros(self.M)
        self.tx, self.ty = self.gx, self.gy
        self.t, self.T, self.running = 1.0, 0.0, False

    def start(self, forward):
        self.tx, self.ty = (self.gx, self.gy) if forward else (self.sx, self.sy)
        self.t, self.running = 0.0, True

    def reset(self):
        self.px, self.py = self.sx.copy(), self.sy.copy()
        self.vx[:] = 0
        self.vy[:] = 0
        self.running = False

    def step(self, dt):
        n, M = self.n, self.M
        self.t += dt / DURATION
        self.T += dt
        tc = min(max(self.t, 0.0), 1.0)
        e = tc * tc * (3 - 2 * tc)
        k = (3 + 90 * e * e) * STIFFNESS
        damp = 2 * math.sqrt(k) * 0.85
        env = math.sin(math.pi * tc)
        f = 2 * math.pi / (n * SWIRL_SCALE)
        S = n * 0.9 * env * SWIRL

        cx = np.clip(self.px.astype(np.int64), 0, n - 1)
        cy = np.clip(self.py.astype(np.int64), 0, n - 1)
        cell = cy * n + cx
        gvx = np.bincount(cell, weights=self.vx, minlength=M).reshape(n, n)
        gvy = np.bincount(cell, weights=self.vy, minlength=M).reshape(n, n)
        gw = np.bincount(cell, minlength=M).reshape(n, n).astype(np.float64)
        bw = box3(gw, n)
        safe = np.where(bw > 0, bw, 1.0)
        avx = (box3(gvx, n) / safe)[cy, cx]
        avy = (box3(gvy, n) / safe)[cy, cx]

        a = f * self.px + self.T * 1.1
        b = f * self.py - self.T * 0.8
        ux = S * f * np.sin(a) * np.cos(b)
        uy = -S * f * np.cos(a) * np.sin(b)
        ax = (self.tx - self.px) * k - damp * self.vx + ux * 3
        ay = (self.ty - self.py) * k - damp * self.vy + uy * 3
        self.vx += ax * dt
        self.vy += ay * dt
        visc = min(0.9, (0.12 + 0.25 * env) * VISCOSITY)
        self.vx += (avx - self.vx) * visc
        self.vy += (avy - self.vy) * visc
        self.px += self.vx * dt
        self.py += self.vy * dt

        if self.t > 1.0:
            close = (np.abs(self.tx - self.px).max() < 0.02 and np.abs(self.ty - self.py).max() < 0.02
                     and np.abs(self.vx).max() < 0.05 and np.abs(self.vy).max() < 0.05)
            if close or self.t > 2.5:
                self.px, self.py = self.tx.copy(), self.ty.copy()
                self.vx[:] = 0
                self.vy[:] = 0
                self.running = False


_REPEAT = {}


def render(sim, colors, size, out=None):
    """Draw all pixels into a (size, size, 3) uint8 array indexed [x, y]."""
    n = sim.n
    if out is None:
        out = np.empty((size, size, 3), dtype=np.uint8)
    out[:] = BACKGROUND
    cs = size / n
    block = int(math.ceil(cs)) + 1
    xs = np.clip((sim.px * cs).astype(np.int64), 0, size - block)
    ys = np.clip((sim.py * cs).astype(np.int64), 0, size - block)
    off = np.arange(block)
    offs = (off[:, None] * size + off[None, :]).ravel()
    idx = ((xs * size + ys)[:, None] + offs[None, :]).ravel()
    key = (id(colors), block)
    if _REPEAT.get("key") != key:
        _REPEAT.update(key=key, val=np.repeat(colors, block * block, axis=0))
    out.reshape(-1, 3)[idx] = _REPEAT["val"]
    return out

# Window

def run_viewer(sim, colors):
    import pygame
    pygame.init()
    size = WINDOW_SIZE
    screen = pygame.display.set_mode((size, size))
    clock = pygame.time.Clock()
    buf = np.zeros((size, size, 3), dtype=np.uint8)
    forward = True            # direction of the current / last morph
    frames, recording = [], RECORD_GIF
    dt = 1.0 / FPS / SUBSTEPS
    if AUTO_PLAY:
        sim.start(True)

    print("Controls: SPACE morph/reverse | R reset | S screenshot | ESC quit")
    shot = 0
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return
                if ev.key == pygame.K_SPACE:
                    if not sim.running:
                        forward = not forward
                        sim.start(forward)
                elif ev.key == pygame.K_r:
                    sim.reset()
                    forward = True
                    sim.start(True)
                elif ev.key == pygame.K_s:
                    path = os.path.join(SCREENSHOT_DIR, f"screenshot_{shot:03d}.png")
                    Image.fromarray(buf.transpose(1, 0, 2)).save(path)
                    print("Saved", path)
                    shot += 1

        if sim.running:
            for _ in range(SUBSTEPS):
                sim.step(dt)
        render(sim, colors, size, buf)
        pygame.surfarray.blit_array(screen, buf)
        pygame.display.flip()

        if recording:
            frames.append(buf.copy())
            if not sim.running and len(frames) > 1:
                save_gif(frames)
                recording = False
                frames = []

        state = "playing" if sim.running else "finished (SPACE to reverse)"
        pygame.display.set_caption(f"Pixel Fluid Morph - {state} - {clock.get_fps():.0f} fps")
        clock.tick(FPS)


def save_gif(frames):
    keep = frames[::max(1, GIF_EVERY)]
    imgs = [Image.fromarray(f.transpose(1, 0, 2)).resize((GIF_SIZE, GIF_SIZE), Image.LANCZOS) for f in keep]
    ms = int(1000 * max(1, GIF_EVERY) / FPS)
    imgs[0].save(GIF_PATH, save_all=True, append_images=imgs[1:], duration=ms, loop=0)
    print("Saved", GIF_PATH)

# Main

def main():
    if len(sys.argv) >= 3:
        src_path, dst_path = sys.argv[1], sys.argv[2]
    else:
        print("Step 1: choose the image whose pixels will be rearranged (image A).")
        src_path = pick_file("Choose image A (pixels to rearrange)")
        print("Step 2: choose the image to build (image B).")
        dst_path = pick_file("Choose image B (image to build)")
    for p in (src_path, dst_path):
        if not p or not os.path.isfile(p):
            sys.exit(f"Image not found: {p!r}")

    n = int(RESOLUTION)
    A = load_grid(src_path, n)
    B = load_grid(dst_path, n)

    t0 = time.time()
    asg = compute_assignment(A, B, n)
    print(f"Matching done in {time.time() - t0:.1f}s. Total cost: "
          f"{total_cost(A, B, n, TRAVEL_PENALTY, COLOR_WEIGHTS, asg):.3f}")

    colors = A.astype(np.uint8)
    sim = Sim(n, asg)
    run_viewer(sim, colors)


if __name__ == "__main__":
    main()
