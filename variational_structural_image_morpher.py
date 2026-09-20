#!/usr/bin/env python3
"""
Variational Structural Image Morpher
====================================
A complete, single-file desktop app (Tkinter + PyTorch + Pillow).

What it does:
  - Trains LOCALLY on 1-4 user images from a folder.
  - Learns only abstract color fields / contours / composition flows,
    NOT pixel-perfect copies (anti-overfitting bottleneck).
  - Lets you morph / melt / hallucinate BETWEEN the learned structures
    in real-time with 4 sliders (60 FPS capable on CPU).
  - Two-window UI: a responsive Control window (all buttons/sliders/graphs,
    scrollable for small screens) + a pure Output window that shows ONLY
    the dream image and always fits it to the window size.

Anti-copying design:
  1. Tiny generator (~160k params): 32-dim latent -> 128x128 RGB.
     Uses Upsample + Conv2d (resize-convolution) instead of ConvTranspose2d,
     so no checkerboard artifacts can form.
  2. Gaussian noise injected into intermediate layers during training.
  3. Strong weight decay (Adam weight_decay=1e-3).
  4. Loss = downsampled 16x16 blurred spatial loss + Total Variation loss
     (NO raw-pixel MSE). This forces fluid, dreamy color bleeding.

Performance / safety:
  - CPU-only, batch-size 1, ~few MB model, <200MB RAM total.
  - Training runs on a background thread; UI stays responsive.
  - Explicit gc.collect() inside the training loop.
  - Thread-safe model access via a lock.

Requirements:
    pip install torch pillow

Run:
    python variational_structural_image_morpher.py

Author: Arena.ai Agent Mode — 2026

Changelog:
  - v1.3: two-window UI — responsive scrollable Control window + pure
    auto-fitting Output window. Small-screen friendly. Exports still 512px.
  - v1.2: quiet-terminal mode — TORCH_CPP_MIN_LOG_LEVEL + NNPACK/oneDNN
    backends disabled, so no "couldnt initialize nnpack" spam.
  - v1.1: resize-convolution backbone (Upsample + Conv2d), TV 0.12 -> 0.25.
"""

import gc
import logging
import os
import math
import time
import queue
import random
import threading
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# ---------------------------------------------------------------------------
# Silence PyTorch / backend hardware warnings (NNPACK, oneDNN, glog).
# NOTE: this block intentionally runs BEFORE `import torch`, because the
# TORCH_CPP_MIN_LOG_LEVEL / *_VERBOSE environment variables are only read
# once, while PyTorch's C++ libraries initialize. Placing them after the
# torch import would be too late to stop the "couldnt initialize nnpack"
# spam (10+ lines/sec) seen on CPUs without the required SIMD extensions.
# ---------------------------------------------------------------------------
# Force PyTorch and system libraries to suppress non-critical hardware warnings
os.environ["TORCH_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["ONEDNN_VERBOSE"] = "0"
os.environ["MKLDNN_VERBOSE"] = "0"
os.environ["DNNL_VERBOSE"] = "0"
warnings.filterwarnings("ignore")

# Set logging levels to error only
logging.getLogger("torch").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Dependency imports with friendly error if missing
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    _root = tk.Tk()
    _root.withdraw()
    messagebox.showerror(
        "Missing dependency",
        "PyTorch is not installed.\n\nInstall it with:\n\n    pip install torch pillow\n\n"
        "Then re-run this script.",
    )
    raise SystemExit(1)

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    _root = tk.Tk()
    _root.withdraw()
    messagebox.showerror(
        "Missing dependency",
        "Pillow is not installed.\n\nInstall it with:\n\n    pip install pillow\n\n"
        "Then re-run this script.",
    )
    raise SystemExit(1)

# Keep CPU usage predictable on weak machines. Still fast enough for 60 FPS
# inference because the generator is tiny (~160k params).
try:
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2))))
except Exception:
    pass
torch.set_num_interop_threads(1)

# ---------------------------------------------------------------------------
# Disable NNPACK / oneDNN backends that spam "unsupported hardware" warnings.
# Fully guarded: on builds where an API is missing this is a silent no-op.
# ---------------------------------------------------------------------------
def _quiet_noisy_backends():
    """Stop PyTorch from invoking unsupported NNPACK/oneDNN code paths.

    Called once at startup AND right before model creation (both calls are
    cheap and idempotent). Conv ops then fall back to plain CPU kernels —
    zero terminal spam, identical numerics for a net this tiny.
    """
    try:  # NNPACK — the "couldnt initialize nnpack" spammer.
        nnpack = getattr(torch.backends, "nnpack", None)
        if nnpack is not None and hasattr(nnpack, "set_flags"):
            nnpack.set_flags(False)
    except Exception:
        pass
    try:  # oneDNN (MKLDNN) — some builds expose set_flags/enabled knobs.
        mkldnn = getattr(torch.backends, "mkldnn", None)
        if mkldnn is not None:
            if hasattr(mkldnn, "set_flags"):
                try:
                    mkldnn.set_flags(False)
                except TypeError:
                    pass
            if hasattr(mkldnn, "enabled"):
                mkldnn.enabled = False
    except Exception:
        pass


_quiet_noisy_backends()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LATENT_DIM = 32
IMG_SIZE = 128          # network output resolution
SMALL_SIZE = 16         # downsampled loss resolution
DISPLAY_SIZE = 512      # default output-window size + PNG/JPEG export size
MAX_IMAGES = 4
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

SPATIAL_WEIGHT = 1.0
TV_WEIGHT = 0.25        # raised from 0.12: softens harsh/blocky boundaries so
                        # colors bleed and morph seamlessly like a fluid sim
NOISE_STD = 0.12        # Gaussian noise injected during training
ANCHOR_JITTER = 0.08    # latent jitter per batch (extra anti-memorization)
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-3     # STRONG L2 — the key anti-copying penalty


# ===========================================================================
# 1. NEURAL ARCHITECTURE
# ===========================================================================

class GaussianNoise(nn.Module):
    """Inject continuous Gaussian noise — ONLY while training.

    This is half of the anti-copying bottleneck: the network can never rely
    on exact activations, so it cannot form precise pixel-perfect copies.
    At inference (eval mode) it is a transparent pass-through.
    """

    def __init__(self, std: float = 0.12):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class DreamGenerator(nn.Module):
    """Lightweight latent -> image generator (resize-convolution backbone).

    32-dim vector -> FC -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 RGB.
    Total params: ~160k (~0.6 MB). Inference: a few ms on CPU.

    NOTE: this uses Upsample + Conv2d (NOT ConvTranspose2d). Transposed
    convolutions with stride 2 cause uneven pixel overlap -> the classic
    "checkerboard artifact" grid. Plain interpolation followed by a regular
    3x3 convolution upsamples uniformly, so no grid can form — with
    identical (actually slightly lower) RAM usage.
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Single FC projection keeps the param count tiny (32 * 4096 = 131k).
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(inplace=True),
        )

        def up_block(in_ch: int, out_ch: int) -> nn.Sequential:
            # Resize-convolution: smooth bilinear x2 resize, then a regular
            # 3x3 convolution. No transposed convolution => no checkerboard.
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear",
                            align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                GaussianNoise(NOISE_STD),
            )

        self.up1 = up_block(64, 32)   # 8x8 -> 16x16
        self.up2 = up_block(32, 16)   # 16x16 -> 32x32
        self.up3 = up_block(16, 8)    # 32x32 -> 64x64
        self.final = nn.Sequential(   # 64x64 -> 128x128 RGB in [-1, 1]
            nn.Upsample(scale_factor=2, mode="bilinear",
                        align_corners=False),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 64, 8, 8)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.final(x)
        return x

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def dream_loss(pred: torch.Tensor, target: torch.Tensor):
    """Weighted combination loss — deliberately NOT raw-pixel MSE.

    1. Spatial term: similarity on a tiny 16x16 BLURRED (bilinear
       downsampled) version -> only broad color fields / composition matter.
    2. Total Variation term on full-res output -> rewards smooth, continuous,
       fluid bleeding of colors (the 'dreamy' baseline).
    """
    # Heavily downsampled + blurred spatial comparison (128 -> 16).
    pred_small = F.interpolate(pred, size=(SMALL_SIZE, SMALL_SIZE),
                               mode="bilinear", align_corners=False)
    target_small = F.interpolate(target, size=(SMALL_SIZE, SMALL_SIZE),
                                 mode="bilinear", align_corners=False)
    spatial = F.mse_loss(pred_small, target_small)

    # Total Variation (anisotropic) on [0, 1] image.
    p01 = (pred + 1.0) * 0.5
    diff_h = torch.abs(p01[:, :, 1:, :] - p01[:, :, :-1, :]).mean()
    diff_w = torch.abs(p01[:, :, :, 1:] - p01[:, :, :, :-1]).mean()
    tv = diff_h + diff_w

    total = SPATIAL_WEIGHT * spatial + TV_WEIGHT * tv
    return total, spatial.detach(), tv.detach()


# ===========================================================================
# 2. IMAGE HELPERS
# ===========================================================================

def load_images_from_folder(folder: str):
    """Load up to MAX_IMAGES images as a tensor (N,3,128,128) in [-1,1].

    Returns (tensor, pil_thumbs, filenames). Raises ValueError if none found.
    """
    paths = sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not paths:
        raise ValueError("No images found.\nSupported: " + ", ".join(SUPPORTED_EXTS))
    if len(paths) > MAX_IMAGES:
        paths = paths[:MAX_IMAGES]

    tensors, thumbs, names = [], [], []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        thumbs.append(img.copy())
        names.append(p.name)
        # PIL -> tensor in [-1, 1] without requiring numpy directly.
        px = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))  # type: ignore
        px = px.view(IMG_SIZE, IMG_SIZE, 3).permute(2, 0, 1).float() / 255.0
        tensors.append(px * 2.0 - 1.0)
        del img, px
    gc.collect()
    return torch.stack(tensors), thumbs, names


def tensor_to_pil(img_tensor: torch.Tensor, out_size: int = DISPLAY_SIZE) -> Image.Image:
    """Convert a (3,128,128) tensor in [-1,1] to an upscaled PIL image."""
    with torch.no_grad():
        x = ((img_tensor.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).byte()
        x = x.permute(1, 2, 0).contiguous()  # HWC
        pil = Image.frombytes("RGB", (IMG_SIZE, IMG_SIZE), bytes(x.view(-1).tolist()))
    if out_size != IMG_SIZE:
        pil = pil.resize((out_size, out_size), Image.BICUBIC)
    return pil


# ===========================================================================
# 3. TKINTER APPLICATION (two windows: Controls + Output)
# ===========================================================================

class MorpherApp:
    BG = "#16182a"
    PANEL = "#1f2240"
    ACCENT = "#8b7bff"
    ACCENT2 = "#4fd1c5"
    TEXT = "#e8e8f2"
    DIM = "#9a9ab5"

    def __init__(self, root: tk.Tk):
        self.root = root  # ---- CONTROL window (the big, responsive one) ----
        self.root.title("Variational Structural Image Morpher — Controls")
        self.root.configure(bg=self.BG)
        self.root.geometry("500x780")
        self.root.minsize(400, 480)

        # ---- Model state (CPU only) ----
        _quiet_noisy_backends()  # re-assert quiet backends before model init
        self.device = torch.device("cpu")
        self.model = DreamGenerator(LATENT_DIM).to(self.device)
        self.model_lock = threading.Lock()
        self.targets = None          # (N,3,128,128) in [-1,1]
        self.anchors = None          # (N,32) fixed latent codes, one per image
        self.thumbs = []
        self.filenames = []
        self.num_images = 0
        # Fixed random "dream direction" for adaptive slider mapping.
        torch.manual_seed(7)
        self.dream_dir = torch.randn(1, LATENT_DIM)
        self.dream_dir = self.dream_dir / self.dream_dir.norm()
        torch.seed()

        # ---- Training state ----
        self.train_thread = None
        self.stop_event = threading.Event()
        self.training = False
        self.trained_once = False
        self.train_queue: queue.Queue = queue.Queue()
        self.loss_history: list[float] = []
        self.total_epochs = 600

        # ---- Render state ----
        self.base_render: Image.Image | None = None  # 128px master, scaled to window
        self.photo = None                            # PhotoImage ref for output label
        self._render_after_id = None
        self._output_refit_after_id = None
        self._loss_draw_after_id = None
        self._fps_smooth = 0.0
        self.animating = False
        self.thumb_photos = []

        self._build_style()
        self._build_control_ui()
        self._build_output_window()
        self._place_output_window()
        self._poll_train_queue()       # start queue polling loop
        self.schedule_render(300)      # show an initial random dream
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.log(f"Generator: {self.model.count_params():,} params "
                 f"({self.model.count_params()*4/1024/1024:.2f} MB fp32). CPU-only, RAM-safe.")

    # ---------------------------------------------------------- style
    def _build_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.PANEL, foreground=self.TEXT,
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"),
                        foreground=self.ACCENT2)
        style.configure("Dim.TLabel", foreground=self.DIM, font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("Accent.TButton", background=self.ACCENT, foreground="white")
        style.map("Accent.TButton", background=[("active", "#a394ff")])
        style.configure("TScale", background=self.PANEL)
        style.configure("TProgressbar", thickness=10)
        style.configure("TLabelframe", background=self.PANEL, foreground=self.TEXT,
                        font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background=self.PANEL,
                        foreground=self.ACCENT2)
        try:
            style.configure("Vertical.TScrollbar", background="#2c2f55",
                            troughcolor=self.BG, bordercolor=self.BG,
                            arrowcolor=self.DIM)
        except Exception:
            pass

    # ============================================ WINDOW 1 — CONTROLS
    def _build_control_ui(self):
        """Responsive, scrollable control panel.

        Everything stretches horizontally with the window (grid weights +
        dynamic slider lengths + loss-graph redraw), and a scrollbar covers
        short screens, so this fits small resolutions.
        """
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._scroll_canvas = tk.Canvas(self.root, bg=self.BG,
                                        highlightthickness=0, bd=0)
        self._scrollbar = ttk.Scrollbar(self.root, orient="vertical",
                                        command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=self._scrollbar.set)
        self._scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self._scrollbar.grid(row=0, column=1, sticky="ns")

        inner = ttk.Frame(self._scroll_canvas, padding=10)
        self._inner_id = self._scroll_canvas.create_window((0, 0), window=inner,
                                                           anchor="nw")
        inner.columnconfigure(0, weight=1)
        inner.bind("<Configure>", self._on_inner_configure)
        self._scroll_canvas.bind("<Configure>", self._on_scroll_canvas_configure)
        # Mouse wheel scrolls the panel (all platforms). Guarded so it never
        # hijacks the Output window or the log box's own scrolling.
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self._scroll_canvas.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self._scroll_canvas.bind_all("<Button-5>", self._on_mousewheel, add="+")

        # ---- ① Dataset section ----
        ds = ttk.LabelFrame(inner, text="  ①  Dataset (1–4 images)  ", padding=10)
        ds.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ds.columnconfigure(0, weight=1)
        ttk.Button(ds, text="📁  Select Folder…", command=self.select_folder)\
            .grid(row=0, column=0, sticky="ew")
        self.folder_label = ttk.Label(ds, text="No folder selected",
                                      style="Dim.TLabel", wraplength=340)
        self.folder_label.grid(row=1, column=0, sticky="w", pady=(6, 2))
        self.count_label = ttk.Label(ds, text="Images: 0 / 4", style="Dim.TLabel")
        self.count_label.grid(row=2, column=0, sticky="w")
        self.thumb_frame = ttk.Frame(ds)
        self.thumb_frame.grid(row=3, column=0, sticky="w", pady=(6, 0))

        # ---- ② Training panel ----
        tr = ttk.LabelFrame(inner, text="  ②  Training Panel  ", padding=10)
        tr.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        tr.columnconfigure(1, weight=1)
        ttk.Label(tr, text="Epochs:").grid(row=0, column=0, sticky="w")
        self.epochs_var = tk.IntVar(value=600)
        ttk.Spinbox(tr, from_=50, to=3000, increment=50, width=8,
                    textvariable=self.epochs_var).grid(row=0, column=1, sticky="w")
        self.train_btn = ttk.Button(tr, text="✨  Train Morpher",
                                    style="Accent.TButton", command=self.start_training)
        self.train_btn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        self.stop_btn = ttk.Button(tr, text="⏹  Stop", command=self.stop_training,
                                   state="disabled")
        self.stop_btn.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.progress = ttk.Progressbar(tr, mode="determinate", maximum=100)
        self.progress.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        self.loss_label = ttk.Label(tr, text="loss: —", style="Dim.TLabel",
                                    font=("Consolas", 9), wraplength=340)
        self.loss_label.grid(row=4, column=0, columnspan=2, sticky="w")

        # ---- ③ Latent morphing sliders (responsive lengths) ----
        mo = ttk.LabelFrame(inner, text="  ③  Latent Space Morphing  ", padding=10)
        mo.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        mo.columnconfigure(1, weight=1)
        self._morph_frame = mo
        self.sliders: list[tk.Scale] = []
        self.slider_labels: list[ttk.Label] = []
        for i in range(MAX_IMAGES):
            lab = ttk.Label(mo, text=f"Morph {i+1}")
            lab.grid(row=i, column=0, sticky="w", padx=(0, 6))
            self.slider_labels.append(lab)
            s = tk.Scale(mo, from_=0, to=100, orient="horizontal",
                         length=220, showvalue=True, resolution=1,
                         bg=self.PANEL, fg=self.TEXT, troughcolor="#2c2f55",
                         highlightthickness=0, activebackground=self.ACCENT,
                         command=lambda _v, idx=i: self.on_slider(idx))
            s.set(50 if i == 0 else 0)
            s.grid(row=i, column=1, sticky="ew", pady=1)
            self.sliders.append(s)
        mo.bind("<Configure>", self._on_morph_configure)
        btn_row = ttk.Frame(mo)
        btn_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        btn_row.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(btn_row, text="🎲 Random", command=self.randomize_sliders)\
            .grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(btn_row, text="◉ Midpoint", command=self.midpoint_sliders)\
            .grid(row=0, column=1, sticky="ew", padx=(0, 4))
        self.animate_btn = ttk.Button(btn_row, text="▶ Animate",
                                      command=self.toggle_animate)
        self.animate_btn.grid(row=0, column=2, sticky="ew")
        ttk.Label(mo, text="Tip: train 300–800 epochs, then drag sliders to melt\n"
                           "structures into each other.",
                  style="Dim.TLabel").grid(row=5, column=0, columnspan=2,
                                           sticky="w", pady=(8, 0))

        # ---- ④ Output window controls (the image itself lives in window 2) ----
        out = ttk.LabelFrame(inner, text="  ④  Output Window  ", padding=10)
        out.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        out.columnconfigure((0, 1), weight=1)
        self.fps_label = ttk.Label(out, text="— FPS", style="Dim.TLabel",
                                   font=("Consolas", 9))
        self.fps_label.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(out, text="💾 Save Dream…",
                   command=self.save_dream).grid(row=1, column=0, sticky="ew",
                                                 padx=(0, 4), pady=(6, 0))
        self.show_output_btn = ttk.Button(out, text="👁 Hide Output",
                                          command=self.toggle_output_window)
        self.show_output_btn.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(out, text="The dream renders in its own window — resize it freely,\n"
                            "the image always fits. Exports save at 512×512.",
                  style="Dim.TLabel").grid(row=2, column=0, columnspan=2,
                                           sticky="w", pady=(8, 0))

        # ---- ⑤ Loss graph + log (width follows the window) ----
        loss_box = ttk.LabelFrame(inner, text="  Loss (16×16 spatial + TV)  ",
                                  padding=10)
        loss_box.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        loss_box.columnconfigure(0, weight=1)
        self.loss_canvas = tk.Canvas(loss_box, height=110, bg="#0d0e1a",
                                     highlightthickness=0)
        self.loss_canvas.grid(row=0, column=0, sticky="ew")
        self.loss_canvas.bind("<Configure>", self._on_loss_canvas_configure)
        self.log_text = tk.Text(loss_box, height=4, bg="#0d0e1a", fg=self.DIM,
                                font=("Consolas", 8), relief="flat",
                                highlightthickness=0, state="disabled")
        self.log_text.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        # ---- status ----
        self.status_label = ttk.Label(inner, text="Ready. Select a folder to begin.",
                                      style="Dim.TLabel", wraplength=340)
        self.status_label.grid(row=5, column=0, sticky="ew")

    # ---- responsive-layout event handlers ----
    def _on_inner_configure(self, _event=None):
        try:
            self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all"))
        except Exception:
            pass

    def _on_scroll_canvas_configure(self, event):
        try:
            self._scroll_canvas.itemconfigure(self._inner_id, width=event.width)
        except Exception:
            pass

    def _on_mousewheel(self, event):
        try:
            if str(event.widget).startswith(str(self.output_win)):
                return  # never hijack scrolling over the Output window
            if event.widget is getattr(self, "log_text", None):
                return  # let the log box keep its own scrolling
        except Exception:
            pass
        try:
            num = getattr(event, "num", None)
            if num == 4:
                self._scroll_canvas.yview_scroll(-1, "units")
            elif num == 5:
                self._scroll_canvas.yview_scroll(1, "units")
            else:
                delta = getattr(event, "delta", 0)
                if delta:
                    self._scroll_canvas.yview_scroll(int(-delta / 120), "units")
        except Exception:
            pass

    def _on_morph_configure(self, event):
        """Stretch/shrink the 4 sliders to fill the available width.

        Stability: the trough length is the frame width MINUS the scale's
        own *measured* internal overhead (value readout, slider, borders)
        and the fixed grid padding — so the scale always fits its cell and
        can never push the window wider. (Tk toplevels grow to satisfy
        over-demanding slaves even with an explicit geometry, so an
        underestimated margin here would be an infinite growth loop.)
        """
        try:
            if event.widget is not self._morph_frame:
                return
            if event.width < 50:
                return  # not laid out yet — wait for a real size
            label_w = 0
            for lab in self.slider_labels:
                try:
                    label_w = max(label_w, lab.winfo_width())
                except Exception:
                    pass
            if label_w < 10:
                return  # labels not laid out yet
            s0 = self.sliders[0]
            try:
                internal = max(0, s0.winfo_reqwidth() - int(s0.cget("length")))
            except Exception:
                internal = 60
            grid_pad = 40  # mo padding 2x10 + label padx + borders + slack
            length = max(120, event.width - label_w - grid_pad - internal)
            for s in self.sliders:
                try:
                    if abs(int(s.cget("length")) - length) >= 2:
                        s.configure(length=length)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_loss_canvas_configure(self, event):
        if event.widget is not self.loss_canvas:
            return
        if self._loss_draw_after_id is not None:
            try:
                self.root.after_cancel(self._loss_draw_after_id)
            except Exception:
                pass
        self._loss_draw_after_id = self.root.after(120, self._loss_draw_debounced)

    def _loss_draw_debounced(self):
        self._loss_draw_after_id = None
        self._draw_loss_graph()

    # ============================================ WINDOW 2 — PURE OUTPUT
    def _build_output_window(self):
        """A second, fully resizable window showing ONLY the dream image.

        No buttons, no labels, no chrome besides the OS title bar: one black
        label that fills the window, with the image refit on every resize.
        """
        self.output_win = tk.Toplevel(self.root)
        self.output_win.title("✦ Dream Output")
        self.output_win.configure(bg="#000000")
        self.output_win.geometry(f"{DISPLAY_SIZE}x{DISPLAY_SIZE}")
        self.output_win.minsize(160, 160)
        self.output_label = tk.Label(self.output_win, bg="#000000",
                                     bd=0, highlightthickness=0)
        self.output_label.pack(fill="both", expand=True)
        self.output_win.bind("<Configure>", self._on_output_configure)
        # Closing the X button only hides the window (reopen from Controls).
        self.output_win.protocol("WM_DELETE_WINDOW", self._hide_output)

    def _place_output_window(self):
        """Park the Output window just right of the Controls on startup."""
        try:
            self.root.update_idletasks()
            ox = self.root.winfo_rootx() + self.root.winfo_width() + 12
            oy = max(0, self.root.winfo_rooty())
            self.output_win.geometry(f"{DISPLAY_SIZE}x{DISPLAY_SIZE}+{ox}+{oy}")
        except Exception:
            pass

    def _on_output_configure(self, event):
        if event.widget is not self.output_win:
            return
        if self.base_render is None:
            return
        if self._output_refit_after_id is not None:
            try:
                self.root.after_cancel(self._output_refit_after_id)
            except Exception:
                pass
        self._output_refit_after_id = self.root.after(90, self._refit_output_debounced)

    def _refit_output_debounced(self):
        self._output_refit_after_id = None
        self._display_base(high_quality=False)  # fast BILINEAR while resizing

    def _output_target_size(self) -> int:
        """Square size the image should be drawn at (fits the window)."""
        try:
            size = min(self.output_win.winfo_width(),
                       self.output_win.winfo_height())
        except Exception:
            return DISPLAY_SIZE
        if size < 64:
            return DISPLAY_SIZE  # not mapped yet — <Configure> refits next
        return min(size, 2048)

    def _display_base(self, high_quality: bool):
        """Draw the cached 128px master into the Output window, fitted."""
        if self.base_render is None:
            return
        try:
            if not self.output_win.winfo_viewable():
                return  # hidden — stays cached, drawn again on unhide
        except Exception:
            return
        try:
            size = self._output_target_size()
            interp = Image.BICUBIC if high_quality else Image.BILINEAR
            img = (self.base_render.resize((size, size), interp)
                   if size != IMG_SIZE else self.base_render)
            self.photo = ImageTk.PhotoImage(img)
            self.output_label.configure(image=self.photo)
        except Exception:
            pass

    def toggle_output_window(self):
        try:
            if self.output_win.winfo_viewable():
                self._hide_output()
            else:
                self._show_output()
        except Exception:
            pass

    def _hide_output(self):
        try:
            self.output_win.withdraw()
            self.show_output_btn.configure(text="👁 Show Output")
        except Exception:
            pass

    def _show_output(self):
        try:
            self.output_win.deiconify()
            self.output_win.lift()
            self.show_output_btn.configure(text="👁 Hide Output")
            self._display_base(high_quality=True)
        except Exception:
            pass

    # ------------------------------------------------------------ logging
    def log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see("end")
        # Keep the widget light (RAM-safe): cap lines.
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 200:
            self.log_text.delete("1.0", f"{lines-200}.0")
        self.log_text.configure(state="disabled")
        gc.collect()

    def set_status(self, msg: str):
        self.status_label.configure(text=msg)

    # ------------------------------------------------------------ dataset
    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder with 1–4 images")
        if not folder:
            return
        try:
            targets, thumbs, names = load_images_from_folder(folder)
        except ValueError as e:
            messagebox.showwarning("No images", str(e))
            return
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Load error", f"Could not load images:\n{e}")
            return

        # Stop any running training before swapping the dataset.
        self.stop_training()
        if self.train_thread and self.train_thread.is_alive():
            self.train_thread.join(timeout=2.0)

        self.targets = targets
        self.thumbs = thumbs
        self.filenames = names
        self.num_images = targets.size(0)
        self.trained_once = False
        self.loss_history.clear()
        self._draw_loss_graph()

        # Fresh anchors: one fixed random latent code per image.
        torch.manual_seed(int(time.time()) % 100000)
        anchors = torch.randn(self.num_images, LATENT_DIM) * 1.5
        torch.seed()
        self.anchors = anchors

        self.folder_label.configure(text=folder)
        self.count_label.configure(
            text=f"Images: {self.num_images} / {MAX_IMAGES}  ({', '.join(names)})")
        self._refresh_thumbnails()
        self._refresh_slider_labels()
        self.midpoint_sliders()
        self.log(f"Loaded {self.num_images} image(s) from {folder}. "
                 f"Anchors fixed in latent space.")
        self.set_status(f"Loaded {self.num_images} image(s). Press 'Train Morpher'.")
        gc.collect()

    def _refresh_thumbnails(self):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self.thumb_photos.clear()
        for img in self.thumbs:
            small = img.resize((64, 64), Image.BILINEAR)
            ph = ImageTk.PhotoImage(small)
            self.thumb_photos.append(ph)  # keep reference
            tk.Label(self.thumb_frame, image=ph, bg=self.PANEL,
                     highlightthickness=1, highlightbackground="#2c2f55")\
                .pack(side="left", padx=3)
        gc.collect()

    def _refresh_slider_labels(self):
        """Adaptive mapping so sliders are ALWAYS useful.

        n=4: all four blend the four anchors.
        n=3: sliders 1-3 blend; slider 4 = dream drift along a fixed dir.
        n=2: sliders 1-2 blend; sliders 3-4 nudge latent dims 0-1.
        n=1: sliders explore latent dims 0-3 around the single anchor.
        n=0: sliders explore a random latent neighbourhood.
        """
        n = self.num_images
        names = [f"Img{i+1}" for i in range(MAX_IMAGES)]
        if n == 4:
            labels = [f"{names[i]} mix" for i in range(4)]
        elif n == 3:
            labels = [f"{names[i]} mix" for i in range(3)] + ["Dream drift"]
        elif n == 2:
            labels = [f"{names[0]} mix", f"{names[1]} mix",
                      "Latent dim-1", "Latent dim-2"]
        elif n == 1:
            labels = [f"Latent dim-{i+1}" for i in range(4)]
        else:
            labels = [f"Dream {i+1}" for i in range(4)]
        for lab, txt in zip(self.slider_labels, labels):
            lab.configure(text=txt)

    # ------------------------------------------------------------ training
    def start_training(self):
        if self.training:
            return
        if self.targets is None or self.num_images == 0:
            messagebox.showinfo("No dataset",
                                "Select a folder with 1–4 images first.")
            return
        try:
            epochs = int(self.epochs_var.get())
            epochs = max(10, min(5000, epochs))
        except Exception:
            epochs = 600
        self.total_epochs = epochs
        self.stop_event.clear()
        self.training = True
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.configure(value=0)
        self.loss_history.clear()
        self.set_status(f"Training… 0/{epochs} epochs")
        self.log(f"Training started: {epochs} epochs, "
                 f"lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, "
                 f"noise_std={NOISE_STD}.")

        self.train_thread = threading.Thread(
            target=self._train_worker, args=(epochs,), daemon=True,
            name="morpher-train")
        self.train_thread.start()

    def stop_training(self):
        if self.training:
            self.stop_event.set()
            self.set_status("Stopping training…")

    def _train_worker(self, epochs: int):
        """Background backprop loop. NEVER touches Tk directly — uses queue."""
        import gc as _gc  # explicit, per stability constraints
        try:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,  # <-- strong L2 anti-copying penalty
            )
            targets = self.targets
            anchors = self.anchors
            n = self.num_images
            assert targets is not None and anchors is not None

            for epoch in range(epochs):
                if self.stop_event.is_set():
                    self.train_queue.put(("stopped", epoch))
                    break
                # One pass over images in random order, batch-size 1 (RAM-safe).
                perm = torch.randperm(n)
                epoch_loss, epoch_sp, epoch_tv = 0.0, 0.0, 0.0
                for j in perm:
                    idx = int(j.item())
                    with self.model_lock:
                        self.model.train()
                        optimizer.zero_grad(set_to_none=True)
                        # Jittered anchor: never the exact same code twice,
                        # so exact memorization is impossible.
                        z = anchors[idx:idx + 1] + \
                            torch.randn(1, LATENT_DIM) * ANCHOR_JITTER
                        pred = self.model(z)
                        tgt = targets[idx:idx + 1]
                        loss, s_loss, tv = dream_loss(pred, tgt)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 1.0)
                        optimizer.step()
                        epoch_loss += float(loss.item())
                        epoch_sp += float(s_loss.item())
                        epoch_tv += float(tv.item())
                        del z, pred, tgt, loss, s_loss, tv
                # ---- explicit cleanup every epoch (stability constraint) ----
                _gc.collect()
                import gc; gc.collect()

                avg = epoch_loss / max(1, n)
                self.train_queue.put(("epoch", epoch + 1, epochs,
                                      avg, epoch_sp / n, epoch_tv / n))
            else:
                self.train_queue.put(("done", epochs))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.train_queue.put(("error", str(e)))
        finally:
            import gc; gc.collect()

    def _poll_train_queue(self):
        """Main-thread poller: applies worker updates to the UI safely."""
        try:
            while True:
                msg = self.train_queue.get_nowait()
                kind = msg[0]
                if kind == "epoch":
                    _, ep, total, avg, sp, tv = msg
                    self.loss_history.append(avg)
                    if len(self.loss_history) > 3000:  # RAM cap
                        self.loss_history = self.loss_history[-3000:]
                    self.progress.configure(value=100.0 * ep / total)
                    self.loss_label.configure(
                        text=f"epoch {ep}/{total}  loss: {avg:.5f}  "
                             f"(spatial {sp:.5f} + TV {tv:.5f})")
                    self.set_status(f"Training… {ep}/{total}  loss={avg:.5f}")
                    self._draw_loss_graph()
                    # Live preview of the current dream every ~25 epochs.
                    if ep % 25 == 0 or ep == total:
                        self.schedule_render(1)
                elif kind == "done":
                    self._finish_training(
                        f"Training complete ({msg[1]} epochs). "
                        "Drag the sliders to morph!")
                elif kind == "stopped":
                    self._finish_training(
                        f"Training stopped at epoch {msg[1]}. "
                        "Sliders still morph the current state.")
                elif kind == "error":
                    self._finish_training(f"Training error: {msg[1]}")
                    messagebox.showerror("Training error", msg[1])
        except queue.Empty:
            pass
        self.root.after(120, self._poll_train_queue)

    def _finish_training(self, msg: str):
        self.training = False
        self.trained_once = True
        self.train_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.set_status(msg)
        self.log(msg)
        self.schedule_render(1)
        gc.collect()

    # ------------------------------------------------------------ morphing
    def on_slider(self, _idx: int):
        # Debounced re-render -> smooth 60 FPS feel while dragging on CPU.
        self.schedule_render(16)

    def randomize_sliders(self):
        for s in self.sliders:
            s.set(random.randint(0, 100))
        self.schedule_render(1)

    def midpoint_sliders(self):
        n = max(1, self.num_images)
        for i, s in enumerate(self.sliders):
            if self.num_images <= 1:
                s.set(50)
            else:
                s.set(100 // n if i < n else 0)
        self.schedule_render(1)

    def toggle_animate(self):
        self.animating = not self.animating
        self.animate_btn.configure(
            text="⏸ Pause" if self.animating else "▶ Animate")
        if self.animating:
            self._animate_step()

    def _animate_step(self):
        if not self.animating:
            return
        t = time.time() * 0.9
        for i, s in enumerate(self.sliders):
            v = 50 + 46 * math.sin(t + i * (math.pi / 2) + i * 0.6)
            s.set(int(max(0, min(100, v))))
        self._render_now()
        self.root.after(33, self._animate_step)  # ~30 FPS auto-loop (CPU-safe)

    def compute_latent(self) -> torch.Tensor:
        """Map the 4 slider values (0-100) to a 32-dim latent vector."""
        vals = [s.get() / 100.0 for s in self.sliders]  # 0..1
        n = self.num_images
        if n >= 1 and self.anchors is not None:
            A = self.anchors
            if n == 4:
                w = torch.tensor(vals, dtype=torch.float32)
                w = w / w.sum() if w.sum() > 1e-6 else torch.full((4,), 0.25)
                z = (w.view(4, 1) * A).sum(dim=0, keepdim=True)
            elif n == 3:
                w = torch.tensor(vals[:3], dtype=torch.float32)
                w = w / w.sum() if w.sum() > 1e-6 else torch.full((3,), 1 / 3)
                z = (w.view(3, 1) * A).sum(dim=0, keepdim=True)
                z = z + (vals[3] - 0.5) * 2.0 * 1.2 * self.dream_dir
            elif n == 2:
                w0, w1 = vals[0], vals[1]
                tot = w0 + w1 if (w0 + w1) > 1e-6 else 1.0
                z = ((w0 * A[0] + w1 * A[1]) / tot).unsqueeze(0).clone()
                z[0, 0] = z[0, 0] + (vals[2] - 0.5) * 4.0
                z[0, 1] = z[0, 1] + (vals[3] - 0.5) * 4.0
            else:  # n == 1
                z = A[0:1].clone()
                for d in range(4):
                    z[0, d] = z[0, d] + (vals[d] - 0.5) * 4.0
            return z
        # No dataset yet: explore a random neighbourhood so the canvas is alive.
        z = torch.zeros(1, LATENT_DIM)
        for d in range(4):
            z[0, d] = (vals[d] - 0.5) * 4.0
        z[0, 4:] = torch.randn(LATENT_DIM - 4) * 0.3
        return z

    # ------------------------------------------------------------ rendering
    def schedule_render(self, delay_ms: int = 16):
        if self._render_after_id is not None:
            try:
                self.root.after_cancel(self._render_after_id)
            except Exception:
                pass
        self._render_after_id = self.root.after(
            max(1, delay_ms), self._render_now)

    def _render_now(self):
        self._render_after_id = None
        t0 = time.perf_counter()
        try:
            z = self.compute_latent()
            with self.model_lock:
                was_training = self.model.training
                self.model.eval()
                with torch.no_grad():
                    out = self.model(z).squeeze(0)
                if was_training:
                    self.model.train()
            # Cache the 128px master; the Output window scales it to fit.
            base = tensor_to_pil(out, IMG_SIZE)
            del out, z
            self.base_render = base
            self._display_base(high_quality=True)
            dt = max(1e-4, time.perf_counter() - t0)
            fps = 1.0 / dt
            self._fps_smooth = (0.85 * self._fps_smooth + 0.15 * fps
                                if self._fps_smooth > 0 else fps)
            state = "trained" if self.trained_once else (
                "training…" if self.training else "untrained dream")
            self.fps_label.configure(
                text=f"{self._fps_smooth:5.1f} FPS  ·  "
                     f"128→{self._output_target_size()}px  ·  {state}")
        except Exception as e:  # noqa: BLE001
            try:
                self.fps_label.configure(text=f"render error: {e}")
            except Exception:
                pass

    def save_dream(self):
        if self.base_render is None:
            messagebox.showinfo("Nothing yet", "No dream rendered yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save dream image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")])
        if path:
            try:
                # Always export a crisp 512×512, whatever the window size.
                export = self.base_render.resize((DISPLAY_SIZE, DISPLAY_SIZE),
                                                 Image.BICUBIC)
                export.save(path)
                self.log(f"Dream saved to {path}")
                self.set_status(f"Saved {path}")
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("Save error", str(e))

    # ------------------------------------------------------------ loss graph
    def _draw_loss_graph(self):
        c = self.loss_canvas
        try:
            c.delete("all")
        except Exception:
            return
        W = c.winfo_width() or 400
        H = c.winfo_height() or 110
        # Border + grid.
        c.create_rectangle(1, 1, W - 1, H - 1, outline="#2c2f55")
        for f in (0.25, 0.5, 0.75):
            c.create_line(0, H * f, W, H * f, fill="#23264a")
        hist = self.loss_history
        if len(hist) < 2:
            c.create_text(W // 2, H // 2, fill=self.DIM,
                          font=("Segoe UI", 9),
                          text="loss curve appears here during training…")
            return
        # Log-scale y for readability (loss decays fast early on).
        import math as _m
        logs = [_m.log10(max(1e-6, v)) for v in hist]
        lo, hi = min(logs), max(logs)
        span = max(1e-6, hi - lo)
        pts = []
        for i, v in enumerate(logs):
            x = 4 + (W - 8) * i / (len(logs) - 1)
            y = 6 + (H - 12) * (1.0 - (v - lo) / span)
            pts += [x, y]
        c.create_line(*pts, fill=self.ACCENT2, width=2, smooth=True)
        c.create_text(W - 6, 10, anchor="ne", fill=self.DIM,
                      font=("Consolas", 8), text=f"{hist[-1]:.5f}")

    # ------------------------------------------------------------ closing
    def on_closing(self):
        self.animating = False
        self.stop_event.set()
        if self.train_thread and self.train_thread.is_alive():
            self.train_thread.join(timeout=2.5)
        gc.collect()
        try:
            self.root.destroy()  # also destroys the child Output window
        except Exception:
            pass


# ===========================================================================
# 4. ENTRY POINT
# ===========================================================================

def main():
    root = tk.Tk()
    # Hi-DPI nicety on Windows.
    try:
        from ctypes import windll  # type: ignore
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = MorpherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
