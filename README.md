# Pixel Fluid Morph

Rearranges every pixel of **image A** into the shape of **image B**, then plays the move as a fluid-like animation in a window.

Every pixel of A is used exactly once. Nothing is recolored. Only the positions change.

## Install

Python 3.9 or newer.

```bash
pip install numpy pillow pygame scipy
```

`scipy` is optional. Without it the program uses the fast solver at every resolution.

On Linux, the file picker needs Tk: `sudo apt install python3-tk`. If Tk is missing, the program asks you to type the image paths instead.

## Run

```bash
python main.py
```

1. A file dialog asks for **image A** (the pixels to rearrange).
2. A second dialog asks for **image B** (the picture to build).
3. The program matches pixels, then opens the animation window.

To skip the dialogs:

```bash
python main.py path/to/a.jpg path/to/b.jpg
```

Both images are center-cropped to a square, so use square images for best results.

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Morph again, in the opposite direction (when the animation has finished) |
| `R` | Reset to image A's layout and play again |
| `S` | Save a screenshot (`screenshot_000.png`, ...) |
| `ESC` or `Q` | Quit |

## Customizing

Open `main.py` and edit the **SETTINGS** block at the top. Save, then run again.

### Matching

| Setting | Default | What it does |
|---------|---------|--------------|
| `RESOLUTION` | `64` | Both images are shrunk to this many pixels per side. 64 means 4,096 moving pixels. Higher is sharper and slower. |
| `SOLVER` | `"auto"` | `"exact"` gives the mathematically best match. `"fast"` gives a very good match much sooner. `"auto"` picks exact up to `EXACT_MAX_RES`. |
| `EXACT_MAX_RES` | `48` | Largest resolution `"auto"` solves exactly. Exact time and memory grow steeply: 48 takes seconds, 64 can take minutes and over 1 GB of RAM. |
| `TRAVEL_PENALTY` | `0.10` | `0` matches colors only, so pixels may fly across the picture. Higher values keep pixels closer to where they started, at the cost of a less accurate picture. |
| `COLOR_WEIGHTS` | `(1, 1, 1)` | How much red, green and blue errors count. Try `(0.3, 0.59, 0.11)` to weight by perceived brightness, or `(0, 0, 1)` to match blue only. |
| `FAST_ROUNDS` | `3000` | Fast solver only. More rounds improve the match. Each round tests thousands of swaps. |
| `RANDOM_SEED` | `0` | Fast solver only. Change it to get a different but equally good arrangement. |

### Animation

| Setting | Default | What it does |
|---------|---------|--------------|
| `DURATION` | `5.0` | Seconds per morph. |
| `SWIRL` | `1.0` | Strength of the swirling flow. `0` gives straight, calm motion. `2` is turbulent. |
| `SWIRL_SCALE` | `0.7` | Size of the swirls as a fraction of the image. Smaller numbers give tight eddies. Larger numbers give one big current. |
| `VISCOSITY` | `1.0` | How much neighbouring pixels move together. `0` makes pixels independent, like sand. Higher values behave like thick liquid. |
| `STIFFNESS` | `1.0` | How hard pixels are pulled to their destination. Lower values give a looser, floatier arrival. |
| `AUTO_PLAY` | `True` | Start morphing immediately. Set `False` to start on image A and press `SPACE`. |
| `SUBSTEPS` | `2` | Physics steps per frame. Raise to `3` or `4` if motion looks jittery at high `STIFFNESS`. |

### Window and output

| Setting | Default | What it does |
|---------|---------|--------------|
| `WINDOW_SIZE` | `720` | Window width and height in screen pixels. |
| `FPS` | `60` | Target frame rate. |
| `BACKGROUND` | `(0, 0, 0)` | Background color as RGB. |
| `RECORD_GIF` | `False` | Set `True` to save the first morph as an animated GIF. |
| `GIF_PATH` | `"morph.gif"` | Output file for the GIF. |
| `GIF_EVERY` | `2` | Keep every Nth frame. Larger values give smaller, choppier files. |
| `GIF_SIZE` | `360` | GIF width and height in pixels. |
| `SCREENSHOT_DIR` | `"."` | Folder for `S` screenshots. |

### Recipes

**Calm, accurate result**
```python
TRAVEL_PENALTY = 0.0
SWIRL = 0.0
VISCOSITY = 0.5
```

**Pixels barely travel (looks like a local shuffle)**
```python
TRAVEL_PENALTY = 0.8
```

**Wild liquid**
```python
SWIRL = 2.0
SWIRL_SCALE = 0.35
VISCOSITY = 1.5
DURATION = 8.0
```

**Highest quality (patient)**
```python
RESOLUTION = 96
SOLVER = "fast"
FAST_ROUNDS = 10000
```

**Make a shareable GIF**
```python
RECORD_GIF = True
GIF_EVERY = 3
GIF_SIZE = 480
```

## How it works

**1. Cost.** Sending source pixel *i* to target slot *j* costs the weighted squared color difference, plus `TRAVEL_PENALTY` times the squared distance between where pixel *i* starts and where slot *j* is. Both parts are scaled to roughly 0 to 1.

**2. Assignment.** The goal is to give every pixel exactly one slot while minimizing the total cost. This is the classic assignment problem.
- *Exact solver:* the Hungarian method (via SciPy) finds the true minimum.
- *Fast solver:* both images are sorted by brightness and matched by rank, which is already a strong start. Then many disjoint random pairs of pixels are tested in parallel, and any pair that lowers total cost swaps destinations. This repeats `FAST_ROUNDS` times. The result is very close to optimal but not guaranteed to be.

**3. Fluid motion.** Each pixel is a particle. Every frame it gets:
- a damped spring pulling it to its slot, which strengthens over time so pixels scatter first and settle last
- a divergence-free swirl field, so the flow circulates instead of bunching up
- a blend toward the average velocity of its neighbours (found on a grid), which makes groups move together like a liquid

When everything is close enough, pixels snap exactly into place.

## Troubleshooting

- **The matching step takes too long.** Lower `RESOLUTION`, or set `SOLVER = "fast"`.
- **The animation is slow.** Lower `RESOLUTION` or `WINDOW_SIZE`. Rendering cost grows with both.
- **No file dialog appears.** Install Tk (see Install), or pass the two image paths on the command line.
- **The result looks muddy.** Image A needs the colors that image B needs. Pairing a photo with one that has a very different palette, such as a night scene and a bright daytime one, limits what is possible. Try `TRAVEL_PENALTY = 0` and a higher `RESOLUTION`.
- **`ModuleNotFoundError`.** Run the `pip install` line from the Install section.
