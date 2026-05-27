import threading
import queue
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image, ImageTk

from physics import RS
from renderer2d_nb import compute_trajectories_nb, make_ray_fan
from renderer3d_nb  import render_3d, generate_starfield


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
BG = "#0d0d1a"
PANEL = "#1a1a2e"
ACCENT = "#4a8cff"
ACCENT2 = "#ff7744"
ACCENT3 = "#44ddaa"
TEXT = "#d0d0e8"
TEXT2 = "#8080a0"
ENTRY_BG = "#222235"
BTN_ACT = "#2255bb"
DARK = "#080814"


# ─────────────────────────────────────────────────────────────────────────────
# LabeledSlider  — slider + live label + text entry (bidirectional sync)
# ─────────────────────────────────────────────────────────────────────────────
class LabeledSlider(tk.Frame):
    """
    Slider with a numeric Entry field for exact value input.

    Layout:

    The Entry accepts keyboard input; pressing Enter or losing focus
    clamps the value to [from_, to] and updates the slider.
    The slider updates the Entry in real time as it is dragged.
    """

    def __init__(self, parent, label: str, from_: float, to: float,
                 initial: float, resolution: float = 0.1,
                 fmt: str = "{:.1f}", callback=None, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self._fmt = fmt
        self._from = from_
        self._to = to
        self._resolution = resolution
        self._callback = callback

        # ── label row ────────────────────────────────────────────────────────
        tk.Label(self, text=label, bg=PANEL, fg=TEXT2,
                 font=("Consolas", 9)).pack(anchor="w", padx=6, pady=(6, 0))

        # ── slider + entry row ───────────────────────────────────────────────
        row = tk.Frame(self, bg=PANEL)
        row.pack(fill="x", padx=6, pady=(1, 4))

        self.var = tk.DoubleVar(value=initial)

        # Slider
        style_name = f"LS{id(self)}.Horizontal.TScale"
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(style_name, background=PANEL,
                    troughcolor=ENTRY_BG, sliderlength=13, sliderrelief="flat")
        self.slider = ttk.Scale(row, from_=from_, to=to, orient="horizontal",
                                variable=self.var, style=style_name,
                                command=self._slider_moved)
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 4))

        # Entry
        vcmd = (self.register(self._validate_entry), "%P")
        self._entry_var = tk.StringVar(value=fmt.format(initial))
        self.entry = tk.Entry(row, textvariable=self._entry_var,
                              width=7, bg=ENTRY_BG, fg=ACCENT,
                              insertbackground=ACCENT, relief="flat",
                              font=("Consolas", 9, "bold"),
                              highlightthickness=1,
                              highlightcolor=ACCENT,
                              highlightbackground="#333355",
                              validate="key", validatecommand=vcmd)
        self.entry.pack(side="right")
        self.entry.bind("<Return>",   self._entry_committed)
        self.entry.bind("<FocusOut>", self._entry_committed)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _validate_entry(self, new_val: str) -> bool:
        """Allow digits, minus sign, and decimal point only."""
        if new_val in ("", "-", "."):
            return True
        try:
            float(new_val)
            return True
        except ValueError:
            return False

    def _slider_moved(self, _=None):
        """Slider dragged → update entry text."""
        v = self.var.get()
        self._entry_var.set(self._fmt.format(v))
        if self._callback:
            self._callback(v)

    def _entry_committed(self, _=None):
        """Enter pressed or focus lost → parse entry, clamp, push to slider."""
        try:
            v = float(self._entry_var.get())
        except ValueError:
            # Restore last good value
            self._entry_var.set(self._fmt.format(self.var.get()))
            return
        v = max(self._from, min(self._to, v))
        self.var.set(v)
        self._entry_var.set(self._fmt.format(v))
        if self._callback:
            self._callback(v)

    def get(self) -> float:
        return self.var.get()

    def set(self, v: float):
        v = max(self._from, min(self._to, float(v)))
        self.var.set(v)
        self._entry_var.set(self._fmt.format(v))


# ─────────────────────────────────────────────────────────────────────────────
# ToggleButton
# ─────────────────────────────────────────────────────────────────────────────
class ToggleButton(tk.Button):
    def __init__(self, parent, text_on, text_off, initial=True, **kw):
        self._on = text_on; self._off = text_off; self._state = initial
        super().__init__(parent,
                         text=text_on if initial else text_off,
                         command=self._toggle,
                         relief="flat", bd=0, cursor="hand2",
                         font=("Consolas", 9, "bold"), **kw)
        self._update_look()

    def _toggle(self):
        self._state = not self._state
        self.config(text=self._on if self._state else self._off)
        self._update_look()

    def _update_look(self):
        bg = ACCENT if self._state else ENTRY_BG
        fg = "white" if self._state else TEXT2
        self.config(bg=bg, fg=fg,
                    activebackground=BTN_ACT, activeforeground="white")

    def get(self) -> bool:
        return self._state


# ─────────────────────────────────────────────────────────────────────────────
# Small icon button helper
# ─────────────────────────────────────────────────────────────────────────────
def icon_btn(parent, text, cmd, bg=PANEL, fg=TEXT, width=4, tip=""):
    b = tk.Button(parent, text=text, command=cmd,
                  bg=bg, fg=fg, activebackground=BTN_ACT,
                  activeforeground="white", relief="flat",
                  font=("Consolas", 11, "bold"), cursor="hand2",
                  width=width, bd=0)
    return b


# ─────────────────────────────────────────────────────────────────────────────
# 2D Panel - real-time photon animation
# ─────────────────────────────────────────────────────────────────────────────
class Panel2D(tk.Frame):
    """
    Real-time 2D photon trajectory visualisation.

    Workflow

    1. User clicks ▶ COMPUTE — Numba traces all N rays in one call and stores
       the full path buffers  (paths, lengths, statuses).
    2. The animation engine replays those buffers frame by frame using
       matplotlib Line2D.set_data() — no re-computation, just array slicing.
    3. Controls: Play/Pause, Step (single frame), Reset (go to frame 0),
       plus a speed slider that sets the millisecond delay between frames.

    Frame-rate note:
       Each animation tick draws all N ray segments up to the current step,
       then calls canvas.draw_idle()
    """

    # Number of position samples to advance per animation tick
    STEP_SIZE = 8

    def __init__(self, parent, status_cb, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._status_cb = status_cb
        self._render_queue = queue.Queue()
        self._computing = False

        # Animation state
        self._paths = None # (N, MAX_PTS, 2) float64
        self._lengths = None # (N,) int64
        self._statuses = None # (N,) int64
        self._n_rays = 0
        self._frame = 0 # current animation step index
        self._max_frame = 0 # max(lengths)
        self._playing = False
        self._after_id = None # tkinter after() handle
        self._lines = [] # list of matplotlib Line2D objects (trails)
        self._heads = None # scatter of photon head dots
        self._cmap = None

        self._build_controls()
        self._build_canvas()
        self._draw_placeholder()

    # ─────────────────────────────────────────────────────────────────────────
    # Controls
    # ─────────────────────────────────────────────────────────────────────────
    def _build_controls(self):
        ctrl = tk.Frame(self, bg=PANEL, width=250)
        ctrl.pack(side="left", fill="y", padx=(8, 4), pady=8)
        ctrl.pack_propagate(False)

        # ── Title ──────────────────────────────────────────────────────────
        tk.Label(ctrl, text="2D  TRAJECTORIES", bg=PANEL, fg=ACCENT,
                 font=("Consolas", 11, "bold")).pack(pady=(12, 2))
        tk.Label(ctrl, text="real-time photon animation", bg=PANEL, fg=TEXT2,
                 font=("Consolas", 8)).pack()
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=8, pady=6)

        # ── Ray parameters ────────────────────────────────────────────────
        tk.Label(ctrl, text="RAY PARAMETERS", bg=PANEL, fg=TEXT2,
                 font=("Consolas", 8)).pack(anchor="w", padx=8, pady=(4, 0))

        self.n_rays  = LabeledSlider(ctrl, "Number of rays",   10, 200,  60,
                                     resolution=1, fmt="{:.0f}")
        self.b_min = LabeledSlider(ctrl, "Impact param min", -12,  0, -10)
        self.b_max = LabeledSlider(ctrl, "Impact param max",   0, 12,  10)
        self.start_x = LabeledSlider(ctrl, "Ray origin X",    -40, -5, -22)
        for w in (self.n_rays, self.b_min, self.b_max, self.start_x):
            w.pack(fill="x", padx=8)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=8, pady=6)

        # ── Display options ───────────────────────────────────────────────
        tk.Label(ctrl, text="DISPLAY", bg=PANEL, fg=TEXT2,
                 font=("Consolas", 8)).pack(anchor="w", padx=8)

        self.show_circles = ToggleButton(ctrl, "● Circles ON",
                                         "● Circles OFF", initial=True,
                                         bg=ACCENT, fg="white")
        self.show_circles.pack(fill="x", padx=8, pady=3)

        self.show_trails = ToggleButton(ctrl, "〜 Trails ON",
                                        "〜 Trails OFF", initial=True,
                                        bg=ENTRY_BG, fg=TEXT2)
        self.show_trails.pack(fill="x", padx=8, pady=3)

        cm_row = tk.Frame(ctrl, bg=PANEL)
        cm_row.pack(fill="x", padx=8, pady=2)
        tk.Label(cm_row, text="Colormap", bg=PANEL, fg=TEXT2,
                 font=("Consolas", 9)).pack(side="left")
        self.colormap_var = tk.StringVar(value="plasma")
        ttk.Combobox(cm_row, textvariable=self.colormap_var,
                     values=["plasma","viridis","inferno","cool","spring","turbo"],
                     width=9, state="readonly").pack(side="right")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=8, pady=6)

        # ── Animation speed ───────────────────────────────────────────────
        tk.Label(ctrl, text="ANIMATION", bg=PANEL, fg=TEXT2,
                 font=("Consolas", 8)).pack(anchor="w", padx=8)

        self.speed = LabeledSlider(ctrl, "Frame delay (ms)", 10, 200, 40,
                                   resolution=5, fmt="{:.0f}")
        self.speed.pack(fill="x", padx=8)

        self.step_size = LabeledSlider(ctrl, "Steps per frame", 1, 40, 8,
                                       resolution=1, fmt="{:.0f}")
        self.step_size.pack(fill="x", padx=8)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=8, pady=6)

        # ── Compute button ────────────────────────────────────────────────
        self._compute_btn = tk.Button(
            ctrl, text="⚡  COMPUTE RAYS",
            command=self._start_compute,
            bg=ACCENT, fg="white",
            font=("Consolas", 10, "bold"),
            relief="flat", cursor="hand2",
            activebackground=BTN_ACT, activeforeground="white")
        self._compute_btn.pack(fill="x", padx=8, pady=3, ipady=7)

        # ── Playback controls ─────────────────────────────────────────────
        pb = tk.Frame(ctrl, bg=PANEL)
        pb.pack(fill="x", padx=8, pady=3)

        self._play_btn = icon_btn(pb, "▶", self._play,
                                   bg=ACCENT3, fg=DARK, width=3)
        self._pause_btn = icon_btn(pb, "⏸", self._pause,
                                   bg=ENTRY_BG, fg=TEXT, width=3)
        self._step_btn = icon_btn(pb, "▷|", self._step_once,
                                   bg=ENTRY_BG, fg=TEXT, width=3)
        self._reset_btn = icon_btn(pb, "⏮", self._reset_anim,
                                   bg=ENTRY_BG, fg=TEXT, width=3)
        for b in (self._play_btn, self._pause_btn,
                  self._step_btn, self._reset_btn):
            b.pack(side="left", expand=True, fill="x", padx=2, ipady=5)

        # ── Frame counter display ─────────────────────────────────────────
        self._frame_var = tk.StringVar(value="frame —  /  —")
        tk.Label(ctrl, textvariable=self._frame_var,
                 bg=PANEL, fg=TEXT2, font=("Consolas", 8)).pack(pady=2)

        # Progress bar for animation
        self._anim_progress = tk.IntVar(value=0)
        ttk.Progressbar(ctrl, variable=self._anim_progress, maximum=100,
                        length=200,
                        style="Horizontal.TProgressbar").pack(
            padx=8, pady=(0, 4), fill="x")

        # ── Physics info ──────────────────────────────────────────────────
        info = tk.Frame(ctrl, bg=ENTRY_BG)
        info.pack(fill="x", padx=8, pady=(6, 4))
        b_c = 1.5 * np.sqrt(3) * RS
        tk.Label(info,
                 text=(f"b_c = 3√3/2·Rs ≈ {b_c:.3f}\n"
                       f"r_photon = 1.5·Rs = {1.5*RS:.1f}\n"
                       f"r_ISCO   = 3·Rs   = {3*RS:.1f}"),
                 bg=ENTRY_BG, fg=TEXT2,
                 font=("Consolas", 8), justify="left").pack(padx=6, pady=5)

    # ─────────────────────────────────────────────────────────────────────────
    # Matplotlib canvas
    # ─────────────────────────────────────────────────────────────────────────
    def _build_canvas(self):
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)

        self._fig = Figure(facecolor=BG, tight_layout=True)
        self._ax  = self._fig.add_subplot(111)
        self._style_ax()

        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _style_ax(self):
        ax = self._ax
        ax.set_facecolor(DARK)
        ax.tick_params(colors=TEXT2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#222244")
        ax.set_xlabel("x  [GM/c²]", color=TEXT2, fontsize=9)
        ax.set_ylabel("y  [GM/c²]", color=TEXT2, fontsize=9)

    def _draw_placeholder(self):
        ax = self._ax
        ax.cla();  self._style_ax()
        ax.set_title("Click  ⚡ COMPUTE RAYS  ->  then  ▶  to animate",
                     color=TEXT2, fontsize=10)
        ax.set_xlim(-25, 25);  ax.set_ylim(-15, 15);  ax.set_aspect("equal")
        self._add_reference_circles()
        self._mpl_canvas.draw()

    def _add_reference_circles(self):
        ax = self._ax
        if not self.show_circles.get():
            return
        b_c = 1.5 * np.sqrt(3) * RS
        ax.add_patch(mpatches.Circle((0,0), RS, color=DARK, zorder=10))
        ax.add_patch(mpatches.Circle((0,0), RS, color="white",
                                     fill=False, lw=1.8, zorder=11))
        ax.add_patch(mpatches.Circle((0,0), 1.5*RS, color="#ffcc44",
                                     fill=False, lw=1.0, ls="--",
                                     alpha=0.5, zorder=9))
        ax.add_patch(mpatches.Circle((0,0), 3*RS, color="#ff7733",
                                     fill=False, lw=0.7, ls=":",
                                     alpha=0.35, zorder=9))
        # Critical impact parameter guide lines
        for sign in [1, -1]:
            ax.axhline(sign * b_c, color="#ffcc44", lw=0.5, ls="--", alpha=0.25)

    # ─────────────────────────────────────────────────────────────────────────
    # Compute (background thread)
    # ─────────────────────────────────────────────────────────────────────────
    def _start_compute(self):
        if self._computing:
            return
        self._pause()
        self._computing = True
        self._compute_btn.config(state="disabled", text="⏳ Computing…")
        self._status_cb("Computing trajectories with Numba…", 0)
        threading.Thread(target=self._compute_worker, daemon=True).start()
        self.after(40, self._poll_compute_queue)

    def _compute_worker(self):
        n = max(2, int(self.n_rays.get()))
        bmin = self.b_min.get()
        bmax = self.b_max.get()
        sx = self.start_x.get()
        t0 = time.time()
        starts, dirs = make_ray_fan(n, bmin, bmax, sx)
        paths, lengths, statuses = compute_trajectories_nb(starts, dirs)
        elapsed = time.time() - t0
        self._render_queue.put((paths, lengths, statuses, n, elapsed))

    def _poll_compute_queue(self):
        try:
            paths, lengths, statuses, n, elapsed = \
                self._render_queue.get_nowait()
        except queue.Empty:
            self.after(40, self._poll_compute_queue)
            return

        self._computing = False
        self._compute_btn.config(state="normal", text="⚡  COMPUTE RAYS")
        self._paths = paths
        self._lengths = lengths
        self._statuses = statuses
        self._n_rays = n
        self._max_frame = int(lengths.max()) if n > 0 else 0
        self._frame = 0

        n_cap = int((statuses == 1).sum())
        self._status_cb(
            f"Ready — {n} rays, {n_cap} captured, {n-n_cap} escaped  "
            f"({elapsed*1000:.0f} ms Numba)",
            100)
        self._init_animation_objects()
        # Auto-play
        self._play()

    # ─────────────────────────────────────────────────────────────────────────
    # Animation objects — created ONCE after compute, reused every frame
    # ─────────────────────────────────────────────────────────────────────────

    def _init_animation_objects(self):
        """
        Set up the matplotlib scene once:
          - One Line2D per ray (trail)
          - One scatter collection for photon head dots
          - Reference circles and labels
        """
        ax = self._ax
        ax.cla();  self._style_ax()
        ax.set_xlim(-25, 25);  ax.set_ylim(-15, 15);  ax.set_aspect("equal")

        self._cmap = matplotlib.colormaps.get_cmap(self.colormap_var.get())
        b_c = 1.5 * np.sqrt(3) * RS

        # ── Trail lines ───────────────────────────────────────────────────
        self._lines = []
        for i in range(self._n_rays):
            b = abs(self._paths[i, 0, 1])
            t = min(b / 12.0, 1.0)
            color = self._cmap(t)
            near = abs(b - b_c) < 0.3
            lw = 1.8 if near else 0.7
            alpha = 1.0 if near else 0.80
            line, = ax.plot([], [], color=color, lw=lw,
                            alpha=alpha, zorder=5, rasterized=True)
            self._lines.append(line)

        # ── Photon head dots ──────────────────────────────────────────────
        # One scatter collection - update offsets every frame
        head_colors = [self._cmap(min(abs(self._paths[i,0,1])/12.0, 1.0))
                       for i in range(self._n_rays)]
        self._heads = ax.scatter(
            [], [], s=18, zorder=15,
            c=head_colors[:1], # placeholder; updated each frame
            edgecolors="white", linewidths=0.4)
        self._head_colors = head_colors # store for per-frame use

        # ── Reference circles ─────────────────────────────────────────────
        self._add_reference_circles()

        # ── Title placeholder (updated each frame) ────────────────────────
        self._title_obj = ax.set_title("", color=TEXT, fontsize=9, pad=4)

        self._mpl_canvas.draw()
        self._frame_var.set(f"frame  0  /  {self._max_frame}")
        self._anim_progress.set(0)

    # ─────────────────────────────────────────────────────────────────────────
    # Animation playback controls
    # ─────────────────────────────────────────────────────────────────────────
    def _play(self):
        if self._paths is None or self._playing:
            return
        if self._frame >= self._max_frame:
            self._frame = 0   # auto-rewind
        self._playing = True
        self._play_btn.config(bg=ACCENT3, fg=DARK)
        self._pause_btn.config(bg=ENTRY_BG, fg=TEXT)
        self._tick()

    def _pause(self):
        self._playing = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._play_btn.config(bg=ENTRY_BG, fg=TEXT)
        self._pause_btn.config(bg=ACCENT2, fg="white")

    def _step_once(self):
        """Advance exactly one tick without starting continuous playback."""
        if self._paths is None:
            return
        self._pause()
        self._advance_frame()
        self._draw_frame()

    def _reset_anim(self):
        self._pause()
        if self._paths is None:
            return
        self._frame = 0
        self._draw_frame()

    # ─────────────────────────────────────────────────────────────────────────
    # Core animation tick
    # ─────────────────────────────────────────────────────────────────────────
    def _tick(self):
        if not self._playing:
            return
        self._advance_frame()
        self._draw_frame()
        if self._frame < self._max_frame:
            delay = max(10, int(self.speed.get()))
            self._after_id = self.after(delay, self._tick)
        else:
            # Reached the end — pause, leave final frame visible
            self._playing = False
            self._play_btn.config(bg=ENTRY_BG, fg=TEXT)
            self._status_cb(
                f"Animation complete — {self._n_rays} rays, "
                f"frame {self._frame}/{self._max_frame}",
                100)

    def _advance_frame(self):
        step = max(1, int(self.step_size.get()))
        self._frame = min(self._frame + step, self._max_frame)

    # ─────────────────────────────────────────────────────────────────────────
    # Draw one frame
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_frame(self):
        f = self._frame
        show_tr = self.show_trails.get()
        head_xy = []

        for i, line in enumerate(self._lines):
            L = int(self._lengths[i])
            # how far this ray has travelled so far
            end = min(f, L)

            if show_tr and end >= 2:
                line.set_data(self._paths[i, :end, 0],
                              self._paths[i, :end, 1])
            elif not show_tr:
                line.set_data([], [])

            # Head position: last computed point
            if end >= 1 and self._statuses[i] != 1 or end < L:
                # Show head only while ray is still "in flight"
                if end < L or self._statuses[i] == 0:
                    px = float(self._paths[i, end - 1, 0])
                    py = float(self._paths[i, end - 1, 1])
                    # Don't show head inside event horizon
                    if px*px + py*py > RS*RS * 0.9:
                        head_xy.append((px, py))

        # Update head scatter
        if head_xy:
            xy_arr = np.array(head_xy)
            self._heads.set_offsets(xy_arr)
            # colour each head by its ray's impact parameter
            colors_now = []
            for i in range(self._n_rays):
                L   = int(self._lengths[i])
                end = min(f, L)
                if end >= 1 and end < L or self._statuses[i] == 0:
                    px = float(self._paths[i, end-1, 0])
                    py = float(self._paths[i, end-1, 1])
                    if px*px + py*py > RS*RS * 0.9:
                        colors_now.append(self._head_colors[i])
            if colors_now:
                self._heads.set_color(colors_now)
        else:
            self._heads.set_offsets(np.empty((0, 2)))

        # Title
        pct = 100 * f / max(self._max_frame, 1)
        n_alive = sum(
            1 for i in range(self._n_rays)
            if min(f, int(self._lengths[i])) < int(self._lengths[i])
        )
        self._title_obj.set_text(
            f"Step {f} / {self._max_frame}   "
            f"({self._n_rays} rays,  {n_alive} in flight,  "
            f"{int((self._statuses==1).sum())} captured)")

        self._frame_var.set(f"frame  {f}  /  {self._max_frame}")
        self._anim_progress.set(int(pct))
        self._status_cb(
            f"Animating…  step {f}/{self._max_frame}  "
            f"({n_alive} photons in flight)", int(pct))

        self._mpl_canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# 3D Panel
# ─────────────────────────────────────────────────────────────────────────────
class Panel3D(tk.Frame):
    PRESETS = {
        "Tiny  (160×90)": (160,  90),
        "Small (320×180)": (320, 180),
        "Medium (640×360)": (640, 360),
        "HD (1280×720)": (1280, 720),
    }

    def __init__(self, parent, status_cb, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._status_cb = status_cb
        self._render_queue = queue.Queue()
        self._rendering = False
        self._photo = None
        self._starfield = None

        self._build_controls()
        self._build_canvas()

    def _build_controls(self):
        ctrl = tk.Frame(self, bg=PANEL, width=260)
        ctrl.pack(side="left", fill="y", padx=(8, 4), pady=8)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="3D  RAY TRACER", bg=PANEL, fg=ACCENT2,
                 font=("Consolas", 11, "bold")).pack(pady=(12, 4))
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=8, pady=4)

        tk.Label(ctrl, text="CAMERA", bg=PANEL, fg=TEXT2,
                 font=("Consolas", 8)).pack(anchor="w", padx=8, pady=(8, 0))

        self.cam_dist = LabeledSlider(ctrl, "Distance from BH",  5, 50,  45)
        self.cam_theta = LabeledSlider(ctrl, "Polar angle (°)",   0, 89,  85)
        self.cam_phi = LabeledSlider(ctrl, "Azimuth angle (°)", 0, 360, 180)
        self.fov = LabeledSlider(ctrl, "Field of view (°)", 15, 90,  85)
        for w in (self.cam_dist, self.cam_theta, self.cam_phi, self.fov):
            w.pack(fill="x", padx=8)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=8, pady=6)

        tk.Label(ctrl, text="ACCRETION DISK", bg=PANEL, fg=TEXT2,
                 font=("Consolas", 8)).pack(anchor="w", padx=8)

        self.disk_toggle = ToggleButton(ctrl, "◉ Disk  ON",
                                        "◎ Disk  OFF", initial=True,
                                        bg=ACCENT2, fg="white")
        self.disk_toggle.pack(fill="x", padx=8, pady=4)

        self.disk_rin  = LabeledSlider(ctrl, "Inner radius (× Rs)",
                                       1.5, 6.0, 2.0, resolution=0.1)
        self.disk_rout = LabeledSlider(ctrl, "Outer radius (× Rs)",
                                       4.0, 20.0, 12.0, resolution=0.5)
        for w in (self.disk_rin, self.disk_rout):
            w.pack(fill="x", padx=8)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=8, pady=6)

        tk.Label(ctrl, text="PHYSICS", bg=PANEL, fg=TEXT2,
                 font=("Consolas", 8)).pack(anchor="w", padx=8)

        self.dt = LabeledSlider(ctrl, "Step size dt",
                                0.05, 2.0, 0.8, resolution=0.05, fmt="{:.2f}")
        self.dt.pack(fill="x", padx=8)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=8, pady=6)

        tk.Label(ctrl, text="RESOLUTION", bg=PANEL, fg=TEXT2,
                 font=("Consolas", 8)).pack(anchor="w", padx=8)

        self.res_var = tk.StringVar(value="Small (320×180)")
        ttk.Combobox(ctrl, textvariable=self.res_var,
                     values=list(self.PRESETS.keys()),
                     state="readonly", width=22).pack(fill="x", padx=8, pady=4)

        self.regen_stars = ToggleButton(ctrl, "★ Regen stars",
                                        "★ Keep stars", initial=False,
                                        bg=ENTRY_BG, fg=TEXT2)
        self.regen_stars.pack(fill="x", padx=8, pady=4)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=8, pady=4)

        self._btn = tk.Button(ctrl, text="▶  RENDER 3D",
                              command=self._start_render,
                              bg=ACCENT2, fg="white",
                              font=("Consolas", 11, "bold"),
                              relief="flat", cursor="hand2",
                              activebackground="#cc5522",
                              activeforeground="white")
        self._btn.pack(fill="x", padx=8, pady=4, ipady=8)

        self._eta_var = tk.StringVar(value="")
        tk.Label(ctrl, textvariable=self._eta_var, bg=PANEL,
                 fg=TEXT2, font=("Consolas", 8)).pack(pady=4)

    def _build_canvas(self):
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)
        self._canvas = tk.Canvas(right, bg=DARK, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._show_placeholder()

    def _show_placeholder(self):
        self._canvas.delete("all")
        w = self._canvas.winfo_width()  or 600
        h = self._canvas.winfo_height() or 400
        self._canvas.create_text(
            w//2, h//2,
            text="Press  ▶ RENDER 3D  to start\n\n"
                 "Small (320×180)\n"
                 "HD (1280×720) with Numba",
            fill=TEXT2, font=("Consolas", 12), justify="center")

    def _on_canvas_resize(self, _=None):
        if self._photo is not None:
            self._display_stored_image()

    def _cam_pos(self):
        d   = self.cam_dist.get()
        th  = np.radians(self.cam_theta.get())
        phi = np.radians(self.cam_phi.get())
        return np.array([
            d * np.sin(th) * np.cos(phi),
            d * np.sin(th) * np.sin(phi),
            d * np.cos(th),
        ])

    def _start_render(self):
        if self._rendering:
            return
        self._rendering = True
        self._btn.config(state="disabled", text="⏳ Rendering…")
        self._eta_var.set("")
        self._status_cb("Starting 3D render (Numba parallel)…", 0)
        params = dict(
            width = self.PRESETS[self.res_var.get()][0],
            height = self.PRESETS[self.res_var.get()][1],
            cam_pos = self._cam_pos(),
            fov_deg = self.fov.get(),
            disk_on = self.disk_toggle.get(),
            r_in = self.disk_rin.get()  * RS,
            r_out = self.disk_rout.get() * RS,
            dt_base = self.dt.get(),
            regen = self.regen_stars.get(),
        )
        threading.Thread(target=self._render_worker,
                         args=(params,), daemon=True).start()
        self.after(200, self._poll_render_queue)

    def _render_worker(self, p):
        t0 = time.time()
        if self._starfield is None or p["regen"]:
            self._status_cb("Generating star texture…", 5)
            self._starfield = generate_starfield(
                n_bright=5000, width=2048, height=1024)
        self._status_cb(f"Rendering {p['width']}×{p['height']}…", 10)
        img_arr = render_3d(
            width=p["width"], height=p["height"],
            cam_pos=p["cam_pos"], look_at=np.zeros(3),
            fov_deg=p["fov_deg"], starfield=self._starfield,
            disk_on=p["disk_on"], r_in=p["r_in"], r_out=p["r_out"],
            dt_base=p["dt_base"])
        self._render_queue.put((img_arr, time.time()-t0, p["width"], p["height"]))

    def _poll_render_queue(self):
        try:
            img_arr, elapsed, w, h = self._render_queue.get_nowait()
        except queue.Empty:
            self.after(200, self._poll_render_queue)
            return
        self._last_img_arr = img_arr
        self._display_stored_image()
        self._btn.config(state="normal", text="▶  RENDER 3D")
        self._eta_var.set(f"Rendered in {elapsed:.1f}s")
        self._rendering = False
        self._status_cb(
            f"3D done — {w}×{h}, {elapsed:.1f}s "
            f"({w*h/elapsed:,.0f} px/s, Numba parallel)", 100)

    def _display_stored_image(self):
        img_arr = getattr(self, "_last_img_arr", None)
        if img_arr is None:
            return
        cw = self._canvas.winfo_width()  or 640
        ch = self._canvas.winfo_height() or 360
        pil_img = Image.fromarray(
            (np.clip(img_arr, 0, 1) * 255).astype(np.uint8))
        ih, iw = img_arr.shape[:2]
        scale   = min(cw / iw, ch / ih)
        pil_img = pil_img.resize(
            (int(iw*scale), int(ih*scale)), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        self._canvas.create_image(cw//2, ch//2,
                                  anchor="center", image=self._photo)


# ─────────────────────────────────────────────────────────────────────────────
# Application root
# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Black Hole Simulation — Schwarzschild Ray Tracer")
        self.configure(bg=BG)
        self.minsize(960, 620)
        self.geometry("1300x780")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",     background=BG,    borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=TEXT2,
                        padding=[14, 6], font=("Consolas", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", ENTRY_BG)],
                  foreground=[("selected", ACCENT)])
        style.configure("TCombobox",
                        fieldbackground=ENTRY_BG, background=ENTRY_BG,
                        foreground=TEXT, arrowcolor=ACCENT,
                        selectbackground=ENTRY_BG, selectforeground=TEXT)
        style.configure("Horizontal.TProgressbar",
                        troughcolor=ENTRY_BG, background=ACCENT,
                        borderwidth=0, thickness=4)

        self._build_ui()

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=PANEL, height=44)
        hdr.pack(fill="x");  hdr.pack_propagate(False)
        tk.Label(hdr, text="⚫  BLACK HOLE SIMULATION",
                 bg=PANEL, fg=TEXT,
                 font=("Consolas", 14, "bold")).pack(side="left", padx=16, pady=8)
        tk.Label(hdr,
                 text="G=c=M=1  ·  Rs=2  ·  Schwarzschild  ·  RK4 geodesics",
                 bg=PANEL, fg=TEXT2,
                 font=("Consolas", 9)).pack(side="right", padx=16)

        # Notebook
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=(6, 0))

        self._p2d = Panel2D(nb, self._status)
        self._p3d = Panel3D(nb, self._status)
        nb.add(self._p2d, text="  2D  Trajectories  ")
        nb.add(self._p3d, text="  3D  Ray Tracer  ")

        # Status bar
        bar = tk.Frame(self, bg=PANEL, height=28)
        bar.pack(fill="x", side="bottom");  bar.pack_propagate(False)
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self._status_var,
                 bg=PANEL, fg=TEXT2,
                 font=("Consolas", 9)).pack(side="left", padx=12, pady=4)
        self._prog_var = tk.IntVar(value=0)
        ttk.Progressbar(bar, variable=self._prog_var, maximum=100, length=180,
                        style="Horizontal.TProgressbar").pack(
            side="right", padx=12, pady=8)

    def _status(self, msg: str, progress: int):
        self._status_var.set(msg)
        self._prog_var.set(progress)
        self.update_idletasks()


if __name__ == "__main__":
    print("Black Hole Simulation GUI starting…")
    print("Numba JIT compiles on first compute")
    App().mainloop()