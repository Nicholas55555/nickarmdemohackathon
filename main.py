"""
MARS — Two-Hand Control, Mouse Drag, Per-Joint Home Offsets

Features:
  - Drag grabbed blocks in 3D → arm follows via IK
  - Claw drag mode → mouse controls end-effector position
  - Per-joint home offsets: J1/J5 dials, J2/J3/J4 text boxes
  - Arrows in 3D showing each joint's home direction
  - View cube, auto-track, throttled rendering
"""
import sys, time, traceback, math
import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import (SERVO, HOME, CAM_INDEX, CAM_W, CAM_H,
    J1_J2, J2_J3, J3_J4, J4_J5, J5_J6, CLAW,
    PLATFORM_H, PLATFORM_H_MIN, PLATFORM_H_MAX, PLATFORM_W, PLATFORM_D,
    BG, FG, ACCENT, ACCENT2, ACCENT3, PANEL)
from arm_kinematics import ArmKinematics
from body_tracker import HandPairTracker, detect_camera_fov
from block_physics import BlockPhysics

def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def _box_faces(p0, p1, width):
    d = p1 - p0; l = np.linalg.norm(d)
    if l < 0.1: return []
    dn = d / l; up = np.array([0, 1, 0])
    if abs(np.dot(dn, up)) > 0.99: up = np.array([1, 0, 0])
    a = np.cross(dn, up); a /= np.linalg.norm(a) + 1e-9
    b = np.cross(dn, a); b /= np.linalg.norm(b) + 1e-9
    hw = width / 2.0
    c = [base + a * s1 + b * s2
         for base in [p0, p1] for s1 in [-hw, hw] for s2 in [-hw, hw]]
    return [[c[0],c[1],c[3],c[2]], [c[4],c[5],c[7],c[6]],
            [c[0],c[1],c[5],c[4]], [c[2],c[3],c[7],c[6]],
            [c[0],c[2],c[6],c[4]], [c[1],c[3],c[7],c[5]]]


# ══════════════════════════════════════════════════════════════════════
class DegreeDial(tk.Canvas):
    """Circular dial for -180 to +180°."""
    def __init__(self, parent, size=80, value=0, command=None, label="", **kw):
        super().__init__(parent, width=size, height=size + 16,
                         bg=BG, highlightthickness=0, **kw)
        self._sz = size; self._val = value; self._cmd = command; self._label = label
        self._cx = size // 2; self._cy = size // 2; self._r = size // 2 - 6
        self.bind("<Button-1>", self._click)
        self.bind("<B1-Motion>", self._click)
        self._draw()

    def _draw(self):
        self.delete("all")
        cx, cy, r = self._cx, self._cy, self._r
        self.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#475569", width=2)
        for d in range(0, 360, 45):
            rad = math.radians(d - 90)
            self.create_line(cx+(r-4)*math.cos(rad), cy+(r-4)*math.sin(rad),
                             cx+r*math.cos(rad), cy+r*math.sin(rad),
                             fill="#475569", width=1)
        # Labels: show signed values at cardinal points
        for d, lbl in [(0,"0"),(90,"+90"),(180,"±180"),(270,"-90")]:
            rad = math.radians(d - 90)
            self.create_text(cx+(r-14)*math.cos(rad), cy+(r-14)*math.sin(rad),
                             text=lbl, fill="#555", font=("Consolas", 5))
        # Needle — map value (-180..180) to visual angle (0°=top)
        vis_angle = self._val  # -180..180 → visual: 0=top, +90=right, -90=left
        rad = math.radians(vis_angle - 90)
        self.create_line(cx, cy, cx+(r-12)*math.cos(rad), cy+(r-12)*math.sin(rad),
                         fill=ACCENT, width=2, arrow=tk.LAST)
        self.create_oval(cx-2, cy-2, cx+2, cy+2, fill=ACCENT, outline="")
        # Value + label text below
        txt = f"{self._label} {self._val:+.0f}°" if self._label else f"{self._val:+.0f}°"
        self.create_text(cx, self._sz + 8, text=txt,
                         fill=ACCENT, font=("Consolas", 7, "bold"))

    def _click(self, e):
        raw = math.degrees(math.atan2(e.y - self._cy, e.x - self._cx)) + 90
        # Convert 0..360 to -180..180
        raw = raw % 360
        if raw > 180:
            raw -= 360
        self._val = round(max(-180, min(180, raw)))
        self._draw()
        if self._cmd:
            self._cmd(self._val)

    def get(self): return self._val
    def set(self, v): self._val = max(-180, min(180, round(v))); self._draw()


# ══════════════════════════════════════════════════════════════════════
class MacroEngine:
    def __init__(self, arm, physics):
        self.arm = arm; self.phys = physics
        self.active = False; self._steps = []; self._si = 0
        self._tgt = {}; self._st = {}; self._t = 0.; self._dur = 0.

    def start_pickup(self, color):
        b = self.phys.find_by_color(color)
        if not b: return f"No '{color}' block"
        bp = b.pos.copy()
        self._steps = [("open",None,.3),("move",bp+[0,80,0],.8),
                       ("move",bp+[0,10,0],.5),("close",None,.3),
                       ("move",bp+[0,120,0],.6)]
        self._si = 0; self.active = True; self._t = 0.; self._begin()
        return f"Picking up {b.label}..."

    def _begin(self):
        if self._si >= len(self._steps): self.active = False; return
        k, p, d = self._steps[self._si]
        self._dur = d; self._t = 0.
        self._st = {j: self.arm.angles[j] for j in self.arm.JOINTS}
        if k == "open": self._tgt = dict(self._st); self._tgt["J6"] = 73.
        elif k == "close": self._tgt = dict(self._st); self._tgt["J6"] = 10.
        elif k == "move" and p is not None:
            ik = self.arm.solve_angles_for_position(p)
            self._tgt = ik if ik else dict(self._st)
        else: self._tgt = dict(self._st)

    def update(self, dt):
        if not self.active: return False
        self._t += dt; f = min(1., self._t / self._dur)
        f = f * f * (3 - 2 * f)
        for j in self.arm.JOINTS:
            if j in self._st and j in self._tgt:
                self.arm.set_angle(j, self._st[j]*(1-f) + self._tgt[j]*f)
        self.arm.enforce_ground_constraint()
        if f >= 1.: self._si += 1; self._begin()
        return self.active

    def cancel(self): self.active = False


VIEW_PRESETS = {"Front":(20,-90),"Back":(20,90),"Left":(20,0),
                "Right":(20,-180),"Top":(85,-90),"Iso":(30,-60)}


# ══════════════════════════════════════════════════════════════════════
class App:
    _3D_THROTTLE = 3

    def __init__(self, root):
        self.root = root
        root.title("MARS — Two-Hand Robot Arm Sim")
        root.configure(bg=BG); root.geometry("1600x920"); root.minsize(1100,650)
        root.report_callback_exception = self._on_err

        self.arm = ArmKinematics()
        self.tracker = None; self._tracker_err = None
        try: self.tracker = HandPairTracker()
        except Exception as e: self._tracker_err = str(e)

        self.cap = None; self.running = False; self.tracking_on = False
        self._last = None; self._ftimes = []; self._last_time = time.time()
        self._frame_count = 0

        self.physics = BlockPhysics(); self.physics.spawn_default_set()
        self.macro = MacroEngine(self.arm, self.physics)

        # View state
        self._view_elev = 25.; self._view_azim = -60.
        self._auto_track = True; self._drag_active = False
        self._view_locked = False  # locks perspective completely

        # Mouse interaction modes
        self._claw_drag = False     # mouse drags claw via IK
        self._block_drag = False    # mouse drags a grabbed block
        self._mouse_press_pos = None
        self._drag_start_ee = None
        self._mpl_rotation_disabled = False  # track mpl rotation state

        # Per-joint home offsets (degrees)
        self._homes = {j: SERVO[j][2] for j in [f"J{i}" for i in range(1,7)]}

        self._style(); self._build(); self._init_3d(); self._draw_arm()
        if self._tracker_err:
            self._log("Tracker FAILED:")
            for l in self._tracker_err.split("\n"):
                if l.strip(): self._log(f"  {l.strip()}")
        else: self._log("Ready — two-hand control")
        self._physics_tick()

    def _style(self):
        s = ttk.Style(); s.theme_use("clam")
        for n, bg_c in [("D.TFrame",BG),("D.TLabel",BG),("T.TLabel",BG),
                         ("S.TLabel",PANEL),("D.TLabelframe",BG),
                         ("D.TLabelframe.Label",BG)]:
            kw = dict(background=bg_c)
            if "Label" in n and "frame" not in n.lower():
                kw["foreground"] = ACCENT if "T." in n else (ACCENT3 if "S." in n else FG)
                kw["font"] = ("Consolas",12 if "T." in n else 9 if "S." in n else 10,
                              "bold" if "T." in n else "")
            if "frame" in n.lower() and "Label" not in n.split(".")[-1]:
                kw["foreground"] = FG; kw["font"] = ("Consolas",10,"bold")
            if n == "D.TLabelframe.Label":
                kw["foreground"] = ACCENT; kw["font"] = ("Consolas",10,"bold")
            s.configure(n, **kw)

    def _build(self):
        top = ttk.Frame(self.root, style="D.TFrame")
        top.pack(fill=tk.X, padx=10, pady=(3,1))
        ttk.Label(top, text="MARS", style="T.TLabel").pack(side=tk.LEFT)
        self.fps_lbl = ttk.Label(top, text="", style="D.TLabel")
        self.fps_lbl.pack(side=tk.RIGHT)

        body = ttk.Frame(self.root, style="D.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)
        body.columnconfigure(0, weight=3); body.columnconfigure(1, weight=4)
        body.columnconfigure(2, weight=2); body.rowconfigure(0, weight=1)

        # LEFT — camera
        lf = ttk.LabelFrame(body, text=" Camera ", style="D.TLabelframe")
        lf.grid(row=0, column=0, sticky="nsew", padx=(0,3))
        self.cam_lbl = tk.Label(lf, bg="#0d1117")
        self.cam_lbl.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        ir = ttk.Frame(lf, style="D.TFrame"); ir.pack(fill=tk.X, padx=3, pady=(0,3))
        self.info_lbl = ttk.Label(ir, text="Waiting...", style="D.TLabel")
        self.info_lbl.pack(side=tk.LEFT)
        self.grip_lbl = ttk.Label(ir, text="", style="D.TLabel")
        self.grip_lbl.pack(side=tk.RIGHT)

        # CENTRE — 3D
        cf = ttk.LabelFrame(body, text=" 3D View ", style="D.TLabelframe")
        cf.grid(row=0, column=1, sticky="nsew", padx=3)
        self.fig = Figure(figsize=(5,4), dpi=100, facecolor=BG)
        self.canvas3d = FigureCanvasTkAgg(self.fig, master=cf)
        self.canvas3d.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.ee_lbl = ttk.Label(cf, text="EE: --", style="S.TLabel")
        self.ee_lbl.pack(fill=tk.X, padx=3, pady=(0,1))
        # View cube row
        vcf = ttk.Frame(cf, style="D.TFrame"); vcf.pack(fill=tk.X, padx=3, pady=(0,2))
        for nm in VIEW_PRESETS:
            tk.Button(vcf, text=nm, font=("Consolas",7,"bold"), bg="#1e293b",
                      fg="#94a3b8", activebackground="#334155", relief=tk.FLAT,
                      padx=3, pady=1, width=5,
                      command=lambda n=nm: self._set_view_preset(n)).pack(side=tk.LEFT, padx=1)
        self.btn_view = tk.Button(vcf, text="AUTO", font=("Consolas",7,"bold"),
                                   bg="#2d3a2d", fg="white", relief=tk.FLAT,
                                   padx=3, pady=1, command=self._toggle_auto_view)
        self.btn_view.pack(side=tk.RIGHT, padx=1)
        self.btn_lock = tk.Button(vcf, text="LOCK", font=("Consolas",7,"bold"),
                                   bg="#2d2d2d", fg="#888", relief=tk.FLAT,
                                   padx=3, pady=1, command=self._toggle_lock_view)
        self.btn_lock.pack(side=tk.RIGHT, padx=1)

        # RIGHT — controls (scrollable)
        scroll_container = ttk.Frame(body, style="D.TFrame")
        scroll_container.grid(row=0, column=2, sticky="nsew", padx=(3, 0))
        scroll_container.rowconfigure(0, weight=1)
        scroll_container.columnconfigure(0, weight=1)

        rf_canvas = tk.Canvas(scroll_container, bg=BG, highlightthickness=0)
        rf_canvas.grid(row=0, column=0, sticky="nsew")

        rf_scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=rf_canvas.yview)
        rf_scrollbar.grid(row=0, column=1, sticky="ns")
        rf_canvas.configure(yscrollcommand=rf_scrollbar.set)

        # The actual frame that will hold all your controls
        rf = ttk.Frame(rf_canvas, style="D.TFrame")
        rf_window = rf_canvas.create_window((0, 0), window=rf, anchor="nw")

        # 1. Update the canvas scroll region whenever the inner frame's height changes
        rf.bind("<Configure>", lambda e: rf_canvas.configure(scrollregion=rf_canvas.bbox("all")))

        # 2. Force the inner frame to expand and match the canvas's width
        rf_canvas.bind("<Configure>", lambda e: rf_canvas.itemconfigure(rf_window, width=e.width))

        # 3. Cross-platform mouse wheel scrolling (only active when hovering over the panel)
        def _on_mousewheel(event):
            if sys.platform == "win32":
                rf_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            else:  # macOS / generic
                rf_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        def _bind_scroll(e):
            rf_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            rf_canvas.bind_all("<Button-4>", lambda e: rf_canvas.yview_scroll(-1, "units"))  # Linux up
            rf_canvas.bind_all("<Button-5>", lambda e: rf_canvas.yview_scroll(1, "units"))  # Linux down

        def _unbind_scroll(e):
            rf_canvas.unbind_all("<MouseWheel>")
            rf_canvas.unbind_all("<Button-4>")
            rf_canvas.unbind_all("<Button-5>")

        scroll_container.bind("<Enter>", _bind_scroll)
        scroll_container.bind("<Leave>", _unbind_scroll)

        def btn(p, t, bg_c, cmd, **kw):
            b = tk.Button(p, text=t, font=("Consolas", 8, "bold"), bg=bg_c,
                          fg="white", activebackground="#444", relief=tk.FLAT,
                          padx=4, pady=1, command=cmd, **kw)
            b.pack(fill=tk.X, padx=3, pady=1);
            return b
        bf = ttk.LabelFrame(rf, text=" Controls ", style="D.TLabelframe")
        bf.pack(fill=tk.X, pady=(0,2))
        self.btn_cam = btn(bf, "Start Camera", "#1b4332", self._toggle_cam)
        self.btn_track = btn(bf, "Track (M)", "#1a3a5c", self._toggle_track, state=tk.DISABLED)
        self.btn_cal = btn(bf, "Calibrate (Space)", "#5c3a1a", self._calibrate, state=tk.DISABLED)
        self.btn_home = btn(bf, "Home (H)", "#333", self._go_home)
        self.btn_claw_drag = btn(bf, "Claw Drag: OFF", "#2d2d3a", self._toggle_claw_drag)

        # ── Home Offsets — all joints get dials ─────────────────────────
        hf = ttk.LabelFrame(rf, text=" Home Offsets ", style="D.TLabelframe")
        hf.pack(fill=tk.X, pady=2)

        self._home_dials = {}
        # Row 1: J1, J5 (rotation joints)
        dr1 = ttk.Frame(hf, style="D.TFrame")
        dr1.pack(fill=tk.X, padx=2, pady=1)
        for jn, lbl in [("J1", "J1"), ("J5", "J5")]:
            box = ttk.Frame(dr1, style="D.TFrame")
            box.pack(side=tk.LEFT, expand=True)
            d = DegreeDial(box, size=56, value=0, label=lbl,
                           command=lambda v, j=jn: self._on_dial(j, v))
            d.pack(padx=1)
            self._home_dials[jn] = d

        # Row 2: J2, J3, J4 (pitch joints)
        dr2 = ttk.Frame(hf, style="D.TFrame")
        dr2.pack(fill=tk.X, padx=2, pady=1)
        for jn, lbl in [("J2", "J2"), ("J3", "J3"), ("J4", "J4")]:
            box = ttk.Frame(dr2, style="D.TFrame")
            box.pack(side=tk.LEFT, expand=True)
            d = DegreeDial(box, size=56, value=0, label=lbl,
                           command=lambda v, j=jn: self._on_dial(j, v))
            d.pack(padx=1)
            self._home_dials[jn] = d

        # ── Forearm Sensitivity ──────────────────────────────────────
        fsf = ttk.LabelFrame(rf, text=" Forearm Sens. ", style="D.TLabelframe")
        fsf.pack(fill=tk.X, pady=2)
        self._sens_sliders = {}
        for jn, lbl, default in [("J1", "J1 Base", 20), ("J2", "J2 Shldr", 20),
                                  ("J4", "J4 Wrist", 20), ("J5", "J5 Roll", 20)]:
            r = ttk.Frame(fsf, style="D.TFrame")
            r.pack(fill=tk.X, padx=4, pady=1)
            ttk.Label(r, text=f"{lbl}:", width=8, style="D.TLabel",
                      font=("Consolas", 7)).pack(side=tk.LEFT)
            sl = ttk.Label(r, text=f"{default}%", width=4, style="D.TLabel",
                           font=("Consolas", 7))
            sl.pack(side=tk.RIGHT)
            sc = tk.Scale(r, from_=0, to=100, orient=tk.HORIZONTAL, bg=BG,
                          fg=FG, troughcolor=PANEL, highlightbackground=BG,
                          activebackground="#4a7c59", length=70, showvalue=False,
                          command=lambda v, j=jn, l=sl: self._on_sens(j, int(float(v)), l))
            sc.set(default)
            sc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
            self._sens_sliders[jn] = sc

        # ── Finger Sensitivity ───────────────────────────────────────
        fgf = ttk.LabelFrame(rf, text=" Finger Sens. ", style="D.TLabelframe")
        fgf.pack(fill=tk.X, pady=2)
        for key, lbl, default in [("left", "L-Hand", 50), ("right", "R-Hand", 50)]:
            r = ttk.Frame(fgf, style="D.TFrame")
            r.pack(fill=tk.X, padx=4, pady=1)
            ttk.Label(r, text=f"{lbl}:", width=8, style="D.TLabel",
                      font=("Consolas", 7)).pack(side=tk.LEFT)
            sl = ttk.Label(r, text=f"{default}%", width=4, style="D.TLabel",
                           font=("Consolas", 7))
            sl.pack(side=tk.RIGHT)
            sc = tk.Scale(r, from_=0, to=100, orient=tk.HORIZONTAL, bg=BG,
                          fg=FG, troughcolor=PANEL, highlightbackground=BG,
                          activebackground="#59794a", length=70, showvalue=False,
                          command=lambda v, k=key, l=sl: self._on_finger_sens(k, int(float(v)), l))
            sc.set(default)
            sc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # ── Platform ─────────────────────────────────────────────────
        pf = ttk.LabelFrame(rf, text=" Platform ", style="D.TLabelframe")
        pf.pack(fill=tk.X, pady=2)
        pr = ttk.Frame(pf, style="D.TFrame"); pr.pack(fill=tk.X, padx=4, pady=2)
        self.plat_lbl = ttk.Label(pr, text=f"{PLATFORM_H:.0f}", width=4,
                                   style="D.TLabel", font=("Consolas",8))
        self.plat_lbl.pack(side=tk.RIGHT)
        self.plat_slider = tk.Scale(pr, from_=PLATFORM_H_MIN, to=PLATFORM_H_MAX,
            orient=tk.HORIZONTAL, bg=BG, fg=FG, troughcolor=PANEL,
            highlightbackground=BG, activebackground="#4a7c59", length=80,
            showvalue=False, resolution=5, command=self._on_plat)
        self.plat_slider.set(PLATFORM_H)
        self.plat_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # ── Servo sliders ────────────────────────────────────────────
        sf = ttk.LabelFrame(rf, text=" Servos ", style="D.TLabelframe")
        sf.pack(fill=tk.X, pady=2)
        self.sliders = {}; self.slbl = {}
        jl = {"J1":"J1","J2":"J2","J3":"J3","J4":"J4","J5":"J5","J6":"J6"}
        for jn in ArmKinematics.JOINTS:
            lo,hi,hm = SERVO[jn]
            r = ttk.Frame(sf, style="D.TFrame"); r.pack(fill=tk.X, padx=3, pady=0)
            ttk.Label(r, text=f"{jl[jn]}:", width=3, style="D.TLabel",
                      font=("Consolas",7)).pack(side=tk.LEFT)
            vl = ttk.Label(r, text=f"{hm:.0f}", width=4, style="D.TLabel",
                           font=("Consolas",7))
            vl.pack(side=tk.RIGHT); self.slbl[jn] = vl
            sc = tk.Scale(r, from_=lo, to=hi, orient=tk.HORIZONTAL, bg=BG,
                          fg=FG, troughcolor=PANEL, highlightbackground=BG,
                          activebackground=ACCENT, length=70, showvalue=False,
                          command=lambda v,j=jn: self._slider(j,float(v)))
            sc.set(hm); sc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
            self.sliders[jn] = sc

        # ── Macros + blocks ──────────────────────────────────────────
        mf = ttk.LabelFrame(rf, text=" Macros ", style="D.TLabelframe")
        mf.pack(fill=tk.X, pady=2)
        mr = ttk.Frame(mf, style="D.TFrame"); mr.pack(fill=tk.X, padx=3, pady=1)
        for i,(col,nm) in enumerate([("#ef4444","Red"),("#3b82f6","Blue"),
                                      ("#22c55e","Grn"),("#eab308","Yel")]):
            tk.Button(mr, text=nm, font=("Consolas",7,"bold"), bg=col, fg="white",
                      relief=tk.FLAT, padx=3, pady=0, width=3,
                      command=lambda n=nm.replace("Grn","Green").replace("Yel","Yellow"):
                      self._run_macro(n)).grid(row=0, column=i, sticky="ew", padx=1)
        mr.columnconfigure(0,weight=1); mr.columnconfigure(1,weight=1)
        mr.columnconfigure(2,weight=1); mr.columnconfigure(3,weight=1)
        btn(mf, "Spawn (B)", "#3a2d3a", self._spawn_block)
        btn(mf, "Clear", "#3a2d2d", self._clear_blocks)

        # ── Log ──────────────────────────────────────────────────────
        logf = ttk.LabelFrame(rf, text=" Log ", style="D.TLabelframe")
        logf.pack(fill=tk.BOTH, expand=True, pady=(2,0))
        self.log_w = tk.Text(logf, bg="#0d1117", fg="#58a6ff",
                             font=("Consolas",7), height=3, width=20,
                             wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT)
        self.log_w.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

    # ══════════════════════════════════════════════════════════════════
    # 3D INIT & MOUSE DRAG
    # ══════════════════════════════════════════════════════════════════

    def _init_3d(self):
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#0d1117")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas3d.mpl_connect('button_press_event', self._on_3d_press)
        self.canvas3d.mpl_connect('button_release_event', self._on_3d_release)
        self.canvas3d.mpl_connect('motion_notify_event', self._on_3d_motion)

    def _disable_mpl_rotation(self):
        """Disable matplotlib's built-in 3D drag rotation."""
        if not self._mpl_rotation_disabled:
            try: self.ax.disable_mouse_rotation()
            except: pass
            self._mpl_rotation_disabled = True

    def _enable_mpl_rotation(self):
        """Re-enable matplotlib's built-in 3D drag rotation."""
        if self._mpl_rotation_disabled:
            try: self.ax.mouse_init()
            except: pass
            self._mpl_rotation_disabled = False

    def _on_3d_press(self, event):
        if event.button != 1:
            return
        if self._claw_drag:
            self._block_drag = True
            self._drag_start_ee = self.arm.get_end_effector().copy()
            self._mouse_press_pos = (event.x, event.y)
            self._disable_mpl_rotation()
            return
        grabbed = [b for b in self.physics.blocks if b.grabbed]
        if grabbed:
            self._block_drag = True
            self._drag_start_ee = grabbed[0].pos.copy()
            self._mouse_press_pos = (event.x, event.y)
            self._disable_mpl_rotation()
            return
        if not self._view_locked:
            self._drag_active = True
            self._enable_mpl_rotation()
        else:
            self._disable_mpl_rotation()

    def _on_3d_release(self, event):
        was_ik = self._block_drag
        self._drag_active = False
        self._block_drag = False
        self._mouse_press_pos = None
        if was_ik and not self._claw_drag and not self._view_locked:
            self._enable_mpl_rotation()

    def _on_3d_motion(self, event):
        if not self._block_drag or self._mouse_press_pos is None:
            return
        if event.x is None or event.y is None:
            return
        dx = (event.x - self._mouse_press_pos[0])
        dy = (event.y - self._mouse_press_pos[1])

        # Camera-relative movement:
        # Horizontal drag → move perpendicular to camera view (XZ plane)
        # Vertical drag → move up/down (Y axis)
        azim_rad = math.radians(self._view_azim)

        # View-right direction in plot space → arm XZ plane
        right_x = -math.sin(azim_rad)
        right_z = math.cos(azim_rad)

        scale = 1.2  # pixels to mm
        move_x = dx * right_x * scale
        move_z = dx * right_z * scale
        move_y = -dy * scale  # up = negative screen Y

        target = self._drag_start_ee + np.array([move_x, move_y, move_z])
        target[1] = max(5, target[1])
        ik = self.arm.solve_angles_for_position(target)
        if ik:
            for j, v in ik.items():
                self.arm.set_angle(j, v)
            self.arm.enforce_ground_constraint()
            self._sync_sliders()
            self._draw_arm()

    def _toggle_claw_drag(self):
        self._claw_drag = not self._claw_drag
        if self._claw_drag:
            self._disable_mpl_rotation()
        else:
            if not self._view_locked:
                self._enable_mpl_rotation()
        self.btn_claw_drag.config(
            text=f"Claw Drag: {'ON' if self._claw_drag else 'OFF'}",
            bg="#3a5c1a" if self._claw_drag else "#2d2d3a")
        self._log(f"Claw drag {'ON' if self._claw_drag else 'OFF'}")

    # ══════════════════════════════════════════════════════════════════
    # VIEW
    # ══════════════════════════════════════════════════════════════════

    def _set_view_preset(self, name):
        if name in VIEW_PRESETS:
            self._view_elev, self._view_azim = VIEW_PRESETS[name]
            self._auto_track = False
            self._view_locked = True
            self._disable_mpl_rotation()
            self.btn_view.config(text="MANUAL", bg="#3a2d2d")
            self.btn_lock.config(text="LOCKED", bg="#5c3a1a", fg="white")
            self._draw_arm()

    def _toggle_auto_view(self):
        self._auto_track = not self._auto_track
        if self._auto_track:
            self._view_locked = False
            if not self._claw_drag:
                self._enable_mpl_rotation()
            self.btn_lock.config(text="LOCK", bg="#2d2d2d", fg="#888")
        self.btn_view.config(text="AUTO" if self._auto_track else "MANUAL",
                             bg="#2d3a2d" if self._auto_track else "#3a2d2d")

    def _toggle_lock_view(self):
        self._view_locked = not self._view_locked
        if self._view_locked:
            self._auto_track = False
            self._disable_mpl_rotation()
            self.btn_lock.config(text="LOCKED", bg="#5c3a1a", fg="white")
            self.btn_view.config(text="MANUAL", bg="#3a2d2d")
            self._log("View LOCKED")
        else:
            if not self._claw_drag:
                self._enable_mpl_rotation()
            self.btn_lock.config(text="LOCK", bg="#2d2d2d", fg="#888")
            self._log("View unlocked")

    def _compute_best_view(self, pts):
        pp = np.array([[p[0],p[2],p[1]] for p in pts])
        rg = np.array([pp[-1][0]-pp[0][0], pp[-1][1]-pp[0][1]])
        rl = np.linalg.norm(rg)
        azim = (math.degrees(math.atan2(rg[1],rg[0]))-90) if rl>1e-3 else self._view_azim
        hs = [p[2] for p in pp]; hr = max(hs)-min(hs)
        v = hr/(hr+max(rl,1.)); elev = 15+v*35
        cn = pp - np.mean(pp,axis=0)
        try:
            _,S,_ = np.linalg.svd(cn, full_matrices=False)
            if len(S)>1 and S[1]/(S[0]+1e-9)<.15: elev = max(elev,35)
        except: pass
        return max(8,min(65,elev)), azim

    # ══════════════════════════════════════════════════════════════════
    # 3D DRAW
    # ══════════════════════════════════════════════════════════════════

    def _draw_arm(self):
        try:
            ax = self.ax
            # Capture view before clear to prevent jump-back
            if hasattr(ax,'elev') and hasattr(ax,'azim'):
                if (not self._auto_track or self._drag_active) and not self._view_locked:
                    self._view_elev = ax.elev
                    self._view_azim = ax.azim

            ax.clear(); ax.set_facecolor("#0d1117")
            pts = self.arm.forward_kinematics(); ph = self.arm.platform_h
            def p(v): return (v[0], v[2], v[1])

            # ── Platform ──────────────────────────────────────────────
            if ph > 0.5:
                hw,hd = PLATFORM_W/2., PLATFORM_D/2.
                c = [np.array([dx,dy,dz])
                     for dx in[-hw,hw] for dy in[-hd,hd] for dz in[0,ph]]
                pf = [[c[0],c[1],c[3],c[2]],[c[4],c[5],c[7],c[6]],
                      [c[0],c[1],c[5],c[4]],[c[2],c[3],c[7],c[6]],
                      [c[0],c[2],c[6],c[4]],[c[1],c[3],c[7],c[5]]]
                ax.add_collection3d(Poly3DCollection(pf, alpha=.5,
                    facecolor=(.12,.16,.22), edgecolor=(.2,.25,.34), linewidths=.7))

            # Base disc
            th = np.linspace(0,2*np.pi,30)
            bx,by = 40*np.cos(th), 40*np.sin(th)
            bz = np.full(30, ph)
            ax.plot(bx,by,bz, color="#475569", lw=1.5)
            ax.add_collection3d(Poly3DCollection([list(zip(bx,by,bz))],
                alpha=.3, facecolor="#1e3a5f", edgecolor="#475569"))

            # ── Home direction arrows ─────────────────────────────────
            yaw = np.radians(self.arm.angles["J1"])

            # J1 arrow — FK: forward = (cos(yaw), 0, sin(yaw)) → plot (cos,sin,0)
            j1_home = self._homes.get("J1", 0)
            j1_rad = math.radians(j1_home)
            alen = 65
            arrow_px = alen * math.cos(j1_rad)
            arrow_py = alen * math.sin(j1_rad)
            a0_px = arrow_px * 0.25; a0_py = arrow_py * 0.25
            ax.plot([a0_px, arrow_px], [a0_py, arrow_py], [ph, ph],
                    color="#facc15", lw=2.5, alpha=.8)
            back_px = arrow_px - 12*math.cos(j1_rad)
            back_py = arrow_py - 12*math.sin(j1_rad)
            pp_x = -8*math.sin(j1_rad); pp_y = 8*math.cos(j1_rad)
            ax.plot([arrow_px, back_px+pp_x], [arrow_py, back_py+pp_y],
                    [ph, ph], color="#facc15", lw=2, alpha=.6)
            ax.plot([arrow_px, back_px-pp_x], [arrow_py, back_py-pp_y],
                    [ph, ph], color="#facc15", lw=2, alpha=.6)
            ax.text(arrow_px*1.15, arrow_py*1.15, ph+6,
                    "J1H", color="#facc15", fontsize=5, ha="center", alpha=.5)

            # J2-J4 arrows in arm's swing plane
            arm_fwd = np.array([math.cos(yaw), 0, math.sin(yaw)])
            arm_up = np.array([0, 1, 0])
            arrow_joints = [("J2",1,"#22d3ee",25),("J3",2,"#22d3ee",20),("J4",3,"#a78bfa",18)]
            for jn, idx, col, al in arrow_joints:
                offset = self._homes.get(jn, SERVO[jn][2]) - SERVO[jn][2]
                if abs(offset) > 0.5:
                    jpos = pts[idx]
                    off_rad = math.radians(offset)
                    direction = arm_fwd*math.sin(off_rad) + arm_up*math.cos(off_rad)
                    tip = jpos + direction * al
                    s = p(jpos); e = p(tip)
                    ax.plot([s[0],e[0]],[s[1],e[1]],[s[2],e[2]],color=col,lw=1.8,alpha=.6)
                    ax.text(e[0],e[1],e[2]+5,f"{jn}H",color=col,fontsize=4,ha="center",alpha=.5)

            # J5 arrow (roll direction at J5 position)
            j5_home = self._homes.get("J5", 0)
            if abs(j5_home) > 0.5:
                j5_pos = pts[4]
                roll_rad = math.radians(j5_home)
                roll_dir = np.array([-math.sin(yaw)*math.cos(roll_rad),
                                      math.sin(roll_rad),
                                      math.cos(yaw)*math.cos(roll_rad)])
                tip5 = j5_pos + roll_dir * 22
                s5 = p(j5_pos); e5 = p(tip5)
                ax.plot([s5[0],e5[0]],[s5[1],e5[1]],[s5[2],e5[2]],
                        color="#fb923c", lw=1.8, alpha=.6)
                ax.text(e5[0],e5[1],e5[2]+5, "J5H", color="#fb923c", fontsize=4,
                        ha="center", alpha=.5)

            # ── Link boxes ────────────────────────────────────────────
            for lk in self.arm.link_boxes():
                fs = _box_faces(lk["start"], lk["end"], lk["width"])
                if fs:
                    rgb = _hex_rgb(lk["color"])
                    ax.add_collection3d(Poly3DCollection(
                        [[p(v) for v in f] for f in fs],
                        alpha=lk["alpha"], facecolor=rgb,
                        edgecolor=(*rgb,.6), linewidths=.4))

            # Skeleton + joints
            xs = [pt[0] for pt in pts]; ys = [pt[2] for pt in pts]; zs = [pt[1] for pt in pts]
            ax.plot(xs,ys,zs, color="white", lw=.8, alpha=.3, ls="--")
            jc = ["#475569","#0ea5e9","#0ea5e9","#6366f1","#a78bfa","#ec4899","#f472b6"]
            for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
                ax.scatter(x,y,z, color=jc[min(i,6)], s=40 if i<2 else 55-i*5,
                           edgecolors="white", linewidths=.2, zorder=5, depthshade=True)

            # ── Finger boxes ──────────────────────────────────────────
            for fb in self.arm.finger_boxes():
                fs = _box_faces(fb["start"], fb["end"], fb["width"])
                if fs:
                    rgb = _hex_rgb(fb["color"])
                    ax.add_collection3d(Poly3DCollection(
                        [[p(v) for v in f] for f in fs],
                        alpha=fb["alpha"], facecolor=rgb,
                        edgecolor=(*rgb,.7), linewidths=.5))

            fg = self.arm.get_finger_geometry()
            for tk_ in ["left_tip","right_tip"]:
                t = fg[tk_]; b = fg["claw_base"]
                ax.plot([p(b)[0],p(t)[0]],[p(b)[1],p(t)[1]],[p(b)[2],p(t)[2]],
                        color="#f472b6", lw=.8, alpha=.4, ls="--")
            gc = fg["grip_center"]
            gc_col = "#51cf66" if fg["is_closed"] else "#ff6b6b"
            ax.scatter(*p(gc), color=gc_col, s=25, marker="o", zorder=7,
                       depthshade=False, alpha=.6)

            # Joint axes
            for ja in self.arm.joint_axes():
                av = ja["axis"]*ja["length"]
                s = p(ja["pos"]-av*.5); e = p(ja["pos"]+av*.5)
                ax.plot([s[0],e[0]],[s[1],e[1]],[s[2],e[2]],
                        color=ja["color"], lw=1.2, alpha=.5)

            # Blocks
            for bs in self.physics.get_block_states():
                brgb = _hex_rgb(bs["color"]); ba = .7 if bs["grabbed"] else .5
                ax.add_collection3d(Poly3DCollection(
                    [[p(v) for v in f] for f in bs["faces"]],
                    alpha=ba, facecolor=brgb,
                    edgecolor=(*brgb,.4) if not bs["grabbed"] else (1,1,1,.3),
                    linewidths=.4))
                bp = bs["pos"]
                ax.text(p(bp)[0],p(bp)[1],p(bp)[2]+bs["size"]+6,
                        bs["label"], color=bs["color"], fontsize=4, ha="center", alpha=.7)

            # Ground grid
            for g in np.linspace(-200,200,9):
                ax.plot([g,g],[-200,200],[0,0], color="#1e293b", lw=.15, alpha=.3)
                ax.plot([-200,200],[g,g],[0,0], color="#1e293b", lw=.15, alpha=.3)

            # Workspace arc
            reach = J2_J3+J3_J4+J4_J5
            tw = np.linspace(-np.pi/2, np.pi/2, 50)
            ax.plot(reach*np.cos(tw), reach*np.sin(tw),
                    np.full(50,ph+J1_J2), color="#334155", lw=.5, ls=":", alpha=.2)

            L = 280; zt = max(L+80, ph+L)
            ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_zlim(-20,zt)
            ax.tick_params(colors="#475569", labelsize=4)
            ax.set_xlabel("X", color="#475569", fontsize=5)
            ax.set_ylabel("Z", color="#475569", fontsize=5)
            ax.set_zlabel("Y", color="#475569", fontsize=5)

            if self._auto_track and not self._drag_active and not self._view_locked:
                te,ta = self._compute_best_view(pts); bl = .12
                da = ta - self._view_azim
                if da > 180: da -= 360
                if da < -180: da += 360
                self._view_azim += da*bl
                self._view_elev += (te-self._view_elev)*bl

            ax.view_init(elev=self._view_elev, azim=self._view_azim)
            ax.set_box_aspect([1,1,1])
            for pn in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pn.fill = False; pn.set_edgecolor("#1e293b")
            ax.grid(True, alpha=.08, color="#334155")

            ee = pts[-1]
            self.ee_lbl.config(text=f"EE X={ee[0]:.0f} Y={ee[1]:.0f} Z={ee[2]:.0f}")
            self.canvas3d.draw_idle()
        except Exception as e:
            print(f"[3D] {e}", file=sys.stderr)

    # ══════════════════════════════════════════════════════════════════
    # CAMERA LOOP
    # ══════════════════════════════════════════════════════════════════

    def _toggle_cam(self):
        if self.running: self._stop_cam()
        else: self._start_cam()

    def _start_cam(self):
        if not self.tracker: self._log("No tracker"); return
        try: self.cap = cv2.VideoCapture(CAM_INDEX)
        except Exception as e: self._log(f"Err: {e}"); return
        if not self.cap or not self.cap.isOpened(): self._log("No camera"); return
        ok,f = self.cap.read()
        if not ok: self._log("Read failed"); self.cap.release(); self.cap=None; return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        fov = detect_camera_fov(self.cap); self.tracker.set_fov(fov)
        self.running = True; self._frame_count = 0
        self.btn_cam.config(text="Stop Camera", bg="#5c1a1a")
        self.btn_track.config(state=tk.NORMAL)
        self.btn_cal.config(state=tk.NORMAL)
        self._log("Camera started"); self._loop()

    def _stop_cam(self):
        self.running = False; self.tracking_on = False
        self.btn_cam.config(text="Start Camera", bg="#1b4332")
        self.btn_track.config(text="Track (M)", bg="#1a3a5c", state=tk.DISABLED)
        self.btn_cal.config(state=tk.DISABLED)
        for s in self.sliders.values(): s.config(state=tk.NORMAL)
        if self.cap:
            try: self.cap.release()
            except: pass
            self.cap = None

    def _loop(self):
        if not self.running or not self.cap: return
        try:
            t0 = time.time()
            ok,frame = self.cap.read()
            if not ok or frame is None: self.root.after(50,self._loop); return
            frame = cv2.flip(frame, 1)
            result = self.tracker.process(frame)
            self._last = result

            if result["detected"]:
                a = result["angles"]
                la = "L:OK" if result.get("left_hand_ok") else "L:--"
                ra = "R:OK" if result.get("right_hand_ok") else "R:--"
                wk = " WINK" if result.get("face_ok") else ""
                self.info_lbl.config(text=f"{la} {ra}{wk}")
                g = a.get("J6", 73)
                self.grip_lbl.config(
                    text=f"Claw:{'OPEN' if g>(SERVO['J6'][0]+SERVO['J6'][1])/2 else 'CLOSED'}")
                if self.tracking_on and not self.macro.active:
                    self.arm.apply_angles_smooth(a)
                    self._sync_sliders()
            else:
                self.info_lbl.config(text="No hands"); self.grip_lbl.config(text="")

            now = time.time(); dt = min(now-self._last_time, .1); self._last_time = now
            if self.macro.active: self.macro.update(dt); self._sync_sliders()
            fg = self.arm.get_finger_geometry()
            fg["grip_velocity"] = self.arm._grip_velocity.copy()
            self.physics.step(dt, fg)

            self._show(result["frame"])
            self._frame_count += 1
            if self._frame_count % self._3D_THROTTLE == 0: self._draw_arm()

            dt2 = time.time()-t0; self._ftimes.append(dt2)
            if len(self._ftimes)>30: self._ftimes = self._ftimes[-30:]
            self.fps_lbl.config(text=f"FPS:{1./(np.mean(self._ftimes)+1e-9):.0f}")
            self.root.after(max(1,int((1/30-dt2)*1000)), self._loop)
        except Exception as e:
            print(f"[loop] {e}\n{traceback.format_exc()}", file=sys.stderr)
            self.root.after(500, self._loop)

    def _show(self, bgr):
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lw,lh = self.cam_lbl.winfo_width(), self.cam_lbl.winfo_height()
            if lw>30 and lh>30: rgb = cv2.resize(rgb, (lw,lh))
            im = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.cam_lbl.imgtk = im; self.cam_lbl.config(image=im)
        except: pass

    def _physics_tick(self):
        if not self.running:
            now = time.time(); dt = min(now-self._last_time,.1); self._last_time = now
            if self.macro.active: self.macro.update(dt); self._sync_sliders()
            fg = self.arm.get_finger_geometry()
            fg["grip_velocity"] = self.arm._grip_velocity.copy()
            self.physics.step(dt, fg)
            if any(not b.resting or b.grabbed for b in self.physics.blocks) \
                    or self.macro.active:
                self._draw_arm()
        self.root.after(33, self._physics_tick)

    # ══════════════════════════════════════════════════════════════════
    # ACTIONS
    # ══════════════════════════════════════════════════════════════════

    def _toggle_track(self):
        self.tracking_on = not self.tracking_on
        if self.tracking_on:
            self.btn_track.config(text="Stop Track (M)", bg="#264b73")
            for s in self.sliders.values(): s.config(state=tk.DISABLED)
            self._log("Tracking ON")
        else:
            self.btn_track.config(text="Track (M)", bg="#1a3a5c")
            for s in self.sliders.values(): s.config(state=tk.NORMAL)
            self._log("Tracking OFF")

    def _calibrate(self):
        if not self.tracker: return
        if self._last and self._last["detected"]:
            left_lms = self._last.get("left_hand_lms")
            self.tracker.calibrate(self._last["raw_angles"], left_lms)
            for j in self.arm.JOINTS:
                self.arm.set_angle(j, self._homes.get(j, SERVO[j][2]))
            self.arm.enforce_ground_constraint()
            self._sync_sliders(); self._draw_arm()
            self._log("Calibrated")
        else: self._log("No hands")

    def _on_dial(self, joint, offset_deg):
        """Dial value is offset from default home."""
        absolute = SERVO[joint][2] + offset_deg
        absolute = max(SERVO[joint][0], min(SERVO[joint][1], absolute))
        self._homes[joint] = absolute
        if self.tracker:
            self.tracker.set_joint_home(joint, absolute)
        self._draw_arm()

    def _on_sens(self, joint, val, label_widget):
        label_widget.config(text=f"{val}%")
        if self.tracker:
            if joint == "J1": self.tracker.j1_forearm_sens = val
            elif joint == "J2": self.tracker.j2_forearm_sens = val
            elif joint == "J4": self.tracker.j4_forearm_sens = val
            elif joint == "J5": self.tracker.j5_forearm_sens = val

    def _on_finger_sens(self, key, val, label_widget):
        """Handle finger sensitivity slider (left or right hand)."""
        label_widget.config(text=f"{val}%")
        if self.tracker:
            if key == "left":
                self.tracker.left_finger_sens = val
            elif key == "right":
                self.tracker.right_finger_sens = val

    def _go_home(self):
        self.macro.cancel()
        # Apply user's home offsets instead of config defaults
        for j in self.arm.JOINTS:
            self.arm.set_angle(j, self._homes.get(j, SERVO[j][2]))
        self.arm.enforce_ground_constraint()
        self._sync_sliders(); self._draw_arm()

    def _slider(self, j, v):
        if self.tracking_on: return
        self.arm.set_angle(j, v)
        self.arm.enforce_ground_constraint()
        self.slbl[j].config(text=f"{self.arm.angles[j]:.0f}")
        fg = self.arm.get_finger_geometry()
        fg["grip_velocity"] = self.arm._grip_velocity.copy()
        self.physics.step(.001, fg)
        self._sync_sliders(); self._draw_arm()

    def _on_plat(self, v):
        self.arm.platform_h = float(v)
        self.plat_lbl.config(text=f"{float(v):.0f}")
        self._draw_arm()

    def _sync_sliders(self):
        for j,sc in self.sliders.items():
            a = self.arm.angles[j]; sc.set(a)
            self.slbl[j].config(text=f"{a:.0f}")

    def _spawn_block(self):
        b = self.physics.spawn_block(); self._log(f"Spawned {b.label}"); self._draw_arm()
    def _clear_blocks(self):
        self.physics.clear(); self._log("Cleared"); self._draw_arm()
    def _run_macro(self, color):
        if self.macro.active: self.macro.cancel()
        self.tracking_on = False
        self.btn_track.config(text="Track (M)", bg="#1a3a5c")
        for s in self.sliders.values(): s.config(state=tk.DISABLED)
        self._log(self.macro.start_pickup(color))

    def _log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line)
        try:
            self.log_w.config(state=tk.NORMAL)
            self.log_w.insert(tk.END, line+"\n")
            self.log_w.see(tk.END); self.log_w.config(state=tk.DISABLED)
        except: pass

    def _on_err(self, *a):
        print(f"[TK] {''.join(traceback.format_exception(*a))}", file=sys.stderr)

    def close(self):
        self.running = False
        if self.cap:
            try: self.cap.release()
            except: pass
        if self.tracker:
            try: self.tracker.release()
            except: pass
        self.root.destroy()



def main():
    root = tk.Tk(); app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.bind("<space>", lambda e: app._calibrate())
    root.bind("<h>", lambda e: app._go_home())
    root.bind("<m>", lambda e: app._toggle_track())
    root.bind("<v>", lambda e: app._toggle_auto_view())
    root.bind("<l>", lambda e: app._toggle_lock_view())
    root.bind("<b>", lambda e: app._spawn_block())
    root.bind("<Escape>", lambda e: app.close())
    root.bind("1", lambda e: app._run_macro("Red"))
    root.bind("2", lambda e: app._run_macro("Blue"))
    root.bind("3", lambda e: app._run_macro("Green"))
    root.bind("4", lambda e: app._run_macro("Yellow"))
    root.mainloop()

if __name__ == "__main__":
    main()