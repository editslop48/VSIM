-# Note: basically everything in this repository is vibecoded, so uh.. if you don't like that then.. I don't know, use other stuff.
# Variational Structural Image Morpher

A single-file desktop app (**Tkinter + PyTorch + Pillow**) that trains locally on
**1–4 of your own images** and lets you fluidly **morph, melt, and hallucinate
between them** in real time with 4 sliders — at 60 FPS on an ordinary CPU.

No GPU required. No internet required after installation. No datasets to download.
Peak RAM usage stays well under **200 MB**, so it runs comfortably even on a
4 GB machine.

The app uses **two separate windows**: a responsive **Control window** (all
buttons, sliders, and graphs — scrollable, so it fits small screens) and a
pure **Output window** showing only the dream image, auto-fitted to whatever
size you make it.

```text
  Your 1–4 images              WINDOW 1: Controls         WINDOW 2: Output
 ┌────┐ ┌────┐ ┌────┐ ┌────┐    ┌───────────────────┐      ┌─────────────────┐
 │img1│ │img2│ │img3│ │img4│    │ ① Dataset         │      │                 │
 └────┘ └────┘ └────┘ └────┘    │ ② Training        │      │  dream image    │
       │ train                 │ ③ 4 morph sliders │ ───▶ │  ONLY — always  │
  32-dim latent ──▶ Generator  │ ④ Output controls │      │  fitted to the  │
  vector        (~160k params)  │ ⑤ Loss graph + log│      │  window size    │
                               └───────────────────┘      └─────────────────┘
```

---

## Table of contents

1. [How it works](#how-it-works)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Running the app](#running-the-app)
5. [Quick-start tutorial](#quick-start-tutorial)
6. [UI guide: the Control window](#ui-guide-the-control-window)
7. [UI guide: the Output window](#ui-guide-the-output-window)
8. [Small screens & resizing](#small-screens--resizing)
9. [Reading the loss graph](#reading-the-loss-graph)
10. [Training tips](#training-tips)
11. [Performance notes](#performance-notes)
12. [Customization (tweakable constants)](#customization-tweakable-constants)
13. [Code map](#code-map)
14. [Troubleshooting](#troubleshooting)
15. [FAQ](#faq)
16. [Changelog](#changelog)

---

## How it works

### The core idea

Each of your images is assigned a **fixed random "anchor" point** in a
32-dimensional latent space. A tiny neural network (the *Generator*) learns to
map each anchor → its image. Because the network is deliberately handicapped
(see below), it can only learn the **broad color fields, contours, and
composition flows** — never pixel-perfect copies.

Once trained, dragging the sliders moves the input vector **between** the
anchors. The network has no memorized answer for in-between points, so it
*hallucinates* — structures melt and bleed into one another like a dream.

### Anti-overfitting design (why it can't just copy your images)

| Mechanism | Where in code | Effect |
|---|---|---|
| Tiny bottleneck (32-dim latent, ~160k params) | `DreamGenerator`, `LATENT_DIM` | Not enough capacity to memorize pixels |
| Gaussian noise injected in every block during training | `GaussianNoise`, `NOISE_STD = 0.12` | Activations are never exact; exact copies impossible |
| Strong weight decay | `Adam(..., weight_decay=1e-3)` | Penalizes the complex weights needed for memorization |
| Latent jitter (anchor + noise each batch) | `ANCHOR_JITTER = 0.08` | Network never sees the exact same code twice |
| Downsampled 16×16 blurred loss (no raw-pixel MSE) | `dream_loss()` | Only broad composition matters, not fine detail |
| Total Variation loss | `TV_WEIGHT = 0.25` | Rewards smooth, fluid, continuous color bleeding |

### Network architecture (v1.1+, checkerboard-free)

```text
 input (32,) ──▶ Linear(32 → 4096) ──▶ reshape to (64, 8, 8)
      │ ReLU
      ▼
 ┌─ Upsample ×2 (bilinear) ──▶ Conv2d(64→32, 3×3) ──▶ ReLU ──▶ +Noise ─┐ 16×16
 ├─ Upsample ×2 (bilinear) ──▶ Conv2d(32→16, 3×3) ──▶ ReLU ──▶ +Noise ─┤ 32×32
 ├─ Upsample ×2 (bilinear) ──▶ Conv2d(16→8,  3×3) ──▶ ReLU ──▶ +Noise ─┤ 64×64
 └─ Upsample ×2 (bilinear) ──▶ Conv2d(8→3,   3×3) ──▶ Tanh ─────────────┘ 128×128 RGB
```

> **Why no `ConvTranspose2d`?** Strided transposed convolutions overlap pixels
> unevenly, producing the classic "checkerboard artifact" grid. Plain
> interpolation followed by a regular convolution (*resize-convolution*)
> upsamples uniformly, so no grid can ever form — with identical RAM usage.

---

## Requirements

| Requirement | Details |
|---|---|
| **OS** | Windows 10/11, macOS, or Linux |
| **Python** | 3.10 or newer (`python --version` to check) |
| **RAM** | 4 GB is plenty (app uses < 200 MB) |
| **CPU** | Any 64-bit CPU; GPU is **not** used or needed |
| **Screen** | Any size — both windows are resizable; Controls scroll on short screens |
| **Disk** | ~250 MB for PyTorch (CPU build) + Pillow |

Python packages (installed below): `torch` (CPU-only build is fine and much
smaller), `pillow`. Tkinter ships with most Python installs (see Linux note in
[Troubleshooting](#troubleshooting)).

---

## Installation

### 1. Install Python 3.10+

- **Windows/macOS:** download from [python.org](https://www.python.org/downloads/).
  On Windows, tick **"Add python.exe to PATH"** during setup.
- **Linux (Debian/Ubuntu):**
  ```bash
  sudo apt update
  sudo apt install python3 python3-pip python3-tk python3-venv
  ```
  (`python3-tk` provides Tkinter, which some distros ship separately.)

### 2. (Recommended) Create a virtual environment

```bash
# Windows
py -m venv morpher-env
morpher-env\Scripts\activate

# macOS / Linux
python3 -m venv morpher-env
source morpher-env/bin/activate
```

### 3. Install dependencies

```bash
# CPU-only PyTorch (small download, all this app needs) + Pillow:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pillow
```

> Tip: plain `pip install torch` also works but downloads the larger default
> build (CUDA libraries you will never use). The CPU-only index above keeps
> the install around ~200 MB.

### 4. Get the script

Place `variational_structural_image_morpher.py` anywhere you like — it's fully
self-contained. No other project files are needed.

Verify everything once:

```bash
python variational_structural_image_morpher.py
```

Two windows should open immediately (Controls + a random "untrained dream" in
the Output window). If they do, you're ready.

---

## Running the app

```bash
# Windows
py variational_structural_image_morpher.py
# (or: python variational_structural_image_morpher.py)

# macOS / Linux
python3 variational_structural_image_morpher.py
```

No command-line arguments are needed — everything is controlled from the GUI.
The terminal stays quiet during training and morphing (hardware-backend
warnings are suppressed by design; see [Changelog](#changelog)).

---

## Quick-start tutorial

### Step 1 — Prepare 1 to 4 images

- Create a folder and drop in **1–4 images** (`.png`, `.jpg`, `.jpeg`,
  `.webp`, `.bmp`). If there are more than 4, only the first 4 (sorted by
  name) are used.
- Any size or aspect ratio works — images are **resized to 128×128**, so
  square-ish sources look best.
- Good starters: portraits, characters, landscapes with distinct color moods.
  High-contrast pairs (e.g. a face + a landscape) produce the most dramatic
  melts.

### Step 2 — Select the folder

In the **Control window**, click **📁 Select Folder…** in panel ①. Thumbnails
appear to confirm what was loaded, and the status bar shows
`Loaded N image(s)`.

### Step 3 — Train

1. Set **Epochs** (default **600** is a good starting point).
2. Click **✨ Train Morpher**.
3. Training runs on a background thread — the UI stays responsive, the
   progress bar advances, and the loss graph draws live. A fresh preview
   renders in the Output window every 25 epochs.
4. Use **⏹ Stop** any time; the current state is kept and remains morphable.

Typical training time on a modern laptop CPU: **2–6 minutes for 600 epochs**.

### Step 4 — Morph!

Drag the 4 sliders in panel ③ and watch the **Output window**. It re-renders
in real time (~60 FPS). Shapes, colors, and figures fluidly melt into one
another.

- **🎲 Random** — jump to a random point in latent space.
- **◉ Midpoint** — return to the balanced blend of all images.
- **▶ Animate** — auto-plays a smooth sine-wave morph loop (click again to pause).
- **💾 Save Dream…** (panel ④) — export the current dream as a crisp
  **512×512** PNG/JPEG, whatever size the Output window is.

---

## UI guide: the Control window

The big window. Everything stretches/shrinks with it, and it scrolls
vertically on short screens.

```text
┌─ CONTROLS (resizable + scrollable) ─────────┐
│ ① Dataset                                   │
│   [Select Folder…]  thumbnails              │
│ ② Training Panel                            │
│   Epochs spinner, [Train] [Stop], progress  │
│ ③ Latent Space Morphing                     │
│   4 sliders (auto-stretch) + Random/Mid/Anim│
│ ④ Output Window                             │
│   FPS meter, [Save Dream…] [Show/Hide]      │
│ ⑤ Loss (16×16 spatial + TV)                 │
│   live log-scale curve (auto-width) + log   │
│ status bar                                  │
└─────────────────────────────────────────────┘
```

### What the 4 sliders do (adaptive mapping)

The sliders always control the 32-dim input vector, but their meaning adapts
so every slider stays useful regardless of image count:

| Images loaded | Slider 1 | Slider 2 | Slider 3 | Slider 4 |
|---|---|---|---|---|
| **4** | Img 1 mix | Img 2 mix | Img 3 mix | Img 4 mix |
| **3** | Img 1 mix | Img 2 mix | Img 3 mix | Dream drift (fixed random direction) |
| **2** | Img 1 mix | Img 2 mix | Latent dim-1 nudge | Latent dim-2 nudge |
| **1** | Latent dim-1 | Latent dim-2 | Latent dim-3 | Latent dim-4 |
| **0** | Dream explore (random neighbourhood — works before training!) |

*Mix* sliders act as blend weights between the images' anchor points
(normalized, so only their *relative* positions matter). With a single image,
the sliders instead explore the latent neighbourhood around its anchor —
you still get endless hallucinating variations of that one image.

---

## UI guide: the Output window

The second window contains **nothing but the dream image** on a black
background — no buttons, no text (Save / FPS live in Control panel ④).

- **Resize it freely** (down to 160×160): the image always refits to the
  window, staying square and centered. Fresh renders use high-quality
  bicubic scaling; live drag-resizing uses fast bilinear so it stays smooth.
- **Closing it (✕) only hides it** — click **👁 Show Output** in panel ④ to
  bring it back. Rendering continues in the background while hidden.
- On startup it parks itself just right of the Control window; move it
  wherever you like (second monitor works great).

---

## Small screens & resizing

- Both windows remember nothing about size — set them however you like each
  run. Minimums: Controls 400×480, Output 160×160.
- If the Control window is shorter than its content, **scroll** (mouse wheel
  or the scrollbar) — every section stays reachable on e.g. 768px-tall
  displays.
- Sliders, buttons, the progress bar, and the loss graph all **follow the
  window width** automatically.
- Tight on space? Shrink the Output window to a small preview while training,
  then blow it up to full-screen-ish for the morphing show.

---

## Reading the loss graph

The graph plots total loss on a **log scale** (loss falls fast at first, so a
linear plot would look flat after the first seconds). The readout shows the
decomposition:

```text
epoch 250/600  loss: 0.14123  (spatial 0.13901 + TV 0.00887)
```

- **Falling curve** = the network is learning the broad structures. Good.
- **Flat curve after ~100 epochs** = normal. The anti-overfitting bottleneck
  prevents it from ever reaching zero — that floor *is* the dreaminess.
- **Spatial** dominates early (composition learning); **TV** keeps colors
  fluid throughout.

---

## Training tips

- **Sweet spot: 300–800 epochs.** Under ~150 the output is vague soup;
  past ~1500 you get diminishing returns (the bottleneck caps fidelity).
- **More distinct inputs = more dramatic morphs.** Four near-identical photos
  blend subtly; wildly different images melt spectacularly.
- **Retrain after changing the dataset.** Selecting a new folder resets the
  anchors — previous training does not carry over (by design).
- **The Output window is live during training.** Every 25 epochs you get a
  preview; you can also drag sliders mid-training to peek at the current state.
- **Want sharper dreams?** Lower `TV_WEIGHT` slightly (e.g. 0.15) — see
  [Customization](#customization-tweakable-constants). Want soupier dreams?
  Raise it (e.g. 0.4).
- **Batch size is fixed at 1** and the model is ~0.6 MB — training will not
  exhaust RAM on any machine from the last 15 years.

---

## Performance notes

Measured on CPU (PyTorch 2.x, typical laptop):

| Metric | Value |
|---|---|
| Parameters | 159,635 (~0.61 MB fp32) |
| RAM footprint | < 200 MB total (model + images + UI) |
| Inference | ~1–3 ms/frame → hundreds of FPS raw; UI-capped smooth 60 FPS feel |
| Window refit | Cached 128px master rescaled (bilinear while dragging, bicubic on render) |
| Training (600 epochs, 4 images) | a few minutes on CPU |
| Threads | Training on 1 background thread; UI never blocks |

---

## Customization (tweakable constants)

All knobs live near the top of the script — no need to touch the network code:

| Constant | Default | What it does |
|---|---|---|
| `LATENT_DIM` | `32` | Latent vector size. Larger = more capacity/detail but weaker anti-copying |
| `TV_WEIGHT` | `0.25` | Fluidity of color bleeding. ↑ smoother/soupier, ↓ sharper/blockier |
| `SPATIAL_WEIGHT` | `1.0` | How hard the 16×16 composition match is enforced |
| `NOISE_STD` | `0.12` | Training-time Gaussian noise. ↑ dreamier/vaguer, ↓ more literal |
| `ANCHOR_JITTER` | `0.08` | Per-batch latent jitter. Extra memorization resistance |
| `LEARNING_RATE` | `2e-3` | Adam step size |
| `WEIGHT_DECAY` | `1e-3` | The strong L2 anti-copying penalty — keep high for dreams |
| `DISPLAY_SIZE` | `512` | Default Output-window size + save-export size (net always renders 128px) |
| `MAX_IMAGES` | `4` | Image cap (raising it needs matching slider/UI work) |

> After editing constants, just re-run the script and retrain.

---

## Code map

Single file: `variational_structural_image_morpher.py` (~950 lines)

| Piece | Description |
|---|---|
| Quiet-terminal block (top) | `TORCH_CPP_MIN_LOG_LEVEL`, warning filters, torch log level — runs *before* `import torch` |
| `_quiet_noisy_backends()` | Disables NNPACK/oneDNN spam paths (fully guarded no-op if APIs missing) |
| `GaussianNoise` | Training-only noise injection module |
| `DreamGenerator` | Latent → 128×128 network (FC + 4× resize-convolution blocks) |
| `dream_loss()` | 16×16 downsampled MSE + Total Variation |
| `load_images_from_folder()` / `tensor_to_pil()` | PIL ↔ tensor conversion (128px net space) |
| `MorpherApp._build_control_ui()` | Responsive scrollable Control window (①–⑤ + status) |
| `MorpherApp._build_output_window()` | Pure image-only Output window with auto-refit |
| `MorpherApp.compute_latent()` | The 4-sliders → 32-dim vector adaptive mapping |
| `MorpherApp._display_base()` | Draws the cached 128px master fitted to the Output window |
| `main()` | Entry point |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**
PyTorch isn't installed in the Python you're running. Re-run the install
step, making sure the same interpreter/virtualenv is active:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pillow
```

**`No module named 'tkinter'` (Linux)**
```bash
sudo apt install python3-tk
```

**I closed the Output window — where did my image go?**
Click **👁 Show Output** in Control panel ④. Closing the Output window never
destroys anything; it just hides until you bring it back.

**The Control window doesn't fit my short screen**
Scroll it — mouse wheel or the scrollbar on the right. All five sections stay
reachable at any height (minimum 480px).

**"No images found" when selecting a folder**
Only `.png .jpg .jpeg .webp .bmp` count, and the app reads the top level of
the folder (not subfolders). Check extensions and spelling.

**Training loss stops falling but output looks vague**
Normal — the bottleneck caps fidelity by design. Try: more epochs (up to
~800), slightly lower `NOISE_STD` (e.g. 0.08), or higher-contrast source
images.

**UI feels sluggish while training**
Training and rendering share the CPU. Close the **Animate** loop while
training, or shrink the Output window for cheaper refits.

**`pip install torch` is huge / slow**
Use the CPU-only index (see Installation step 3) — roughly 5× smaller than
the default CUDA build.

**macOS: app opens but sliders don't respond**
Rare Tk scaling quirk — resize the window once; event bindings refresh.

---

## FAQ

**Does it need a GPU?**
No. It is CPU-only by design (`torch.device("cpu")`).

**Does it upload my images anywhere?**
No. Everything runs locally; there is no networking code in the script at all.

**Can I train on more than 4 images?**
Not without editing the UI (4 sliders = 4 anchors). Conceptually you could
raise `MAX_IMAGES` and add sliders, but the 4-anchor blend is the designed
experience.

**Can I save/load a trained model?**
The script doesn't include checkpointing (training takes only minutes).

**Why is the output blurry / dreamy instead of sharp?**
That *is* the aesthetic goal: the 16×16 loss and TV term deliberately discard
fine detail so interpolation hallucinates instead of cross-fading. For
sharper results, lower `TV_WEIGHT` and `NOISE_STD` modestly.

**Which Python versions work?**
3.10+ (the code uses `X | None` type syntax). Tested with PyTorch 2.x CPU
builds and Pillow 10+.

---

## Changelog

- **v1.3 — Two-window UI.** Responsive scrollable Control window (sliders,
  buttons, loss graph all follow the window size) + pure auto-fitting Output
  window showing only the image. Small-screen friendly; exports still 512px.
- **v1.2 — Quiet terminal.** `TORCH_CPP_MIN_LOG_LEVEL=3`, oneDNN verbose flags
  off, Python warnings ignored, torch logger to ERROR, and guarded
  NNPACK/oneDNN backend disabling. No more hardware-warning spam.
- **v1.1 — Checkerboard fix.** Replaced all `ConvTranspose2d` with
  bilinear `Upsample` + regular `Conv2d` (resize-convolution); TV weight
  0.12 → 0.25 for fluid color bleeding. Params: 178,619 → 159,635.
- **v1.0 — Initial release.** Tiny generator, noise + weight-decay
  anti-copying bottleneck, 16×16 + TV loss, threaded training, 4-slider
  morphing UI with live loss graph.

---

*Free to use and modify for any purpose. If you make something beautiful with
it, the sliders did most of the work — but take the credit anyway.*
