"""
MARS — Keyboard-Controlled Robot Arm Sim

Controls:
  A/D  → J1 base rotation
  W/S  → J2 shoulder
  Q/E  → J3 elbow
  R/F  → J4 wrist pitch
  T/G  → J5 wrist rotation
  Z/X  → J6 claw close/open
  H    → Home
  B    → Spawn block
  1-4  → Macro pickup (Red/Blue/Green/Yellow)
  V    → Auto/manual 3D view
  Shift+L → Lock/unlock view
  Esc  → Quit

Hold keys for continuous movement. Speed adjustable via slider.
"""
import sys, time, traceback, math
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

from config import (SERVO, HOME, J1_J2, J2_J3, J3_J4, J4_J5, J5_J6, CLAW,
    PLATFORM_H, PLATFORM_H_MIN, PLATFORM_H_MAX, PLATFORM_W, PLATFORM_D,
    BG, FG, ACCENT, ACCENT2, ACCENT3, PANEL)
from arm_kinematics import ArmKinematics
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
        for d, lbl in [(0,"0"),(90,"+90"),(180,"±180"),(270,"-90")]:
            rad = math.radians(d - 90)
            self.create_text(cx+(r-14)*math.cos(rad), cy+(r-14)*math.sin(rad),
                             text=lbl, fill="#555", font=("Consolas", 5))
        vis_angle = self._val
        rad = math.radians(vis_angle - 90)
        self.create_line(cx, cy, cx+(r-12)*math.cos(rad), cy+(r-12)*math.sin(rad),
                         fill=ACCENT, width=2, arrow=tk.LAST)
        self.create_oval(cx-2, cy-2, cx+2, cy+2, fill=ACCENT, outline="")
        txt = f"{self._label} {self._val:+.0f}°" if self._label else f"{self._val:+.0f}°"
        self.create_text(cx, self._sz + 8, text=txt,
                         fill=ACCENT, font=("Consolas", 7, "bold"))

    def _click(self, e):
        raw = math.degrees(math.atan2(e.y - self._cy, e.x - self._cx)) + 90
        raw = raw % 360
        if raw > 180: raw -= 360
        self._val = round(max(-180, min(180, raw)))
        self._draw()
        if self._cmd: self._cmd(self._val)

    def get(self): return self._val
    def set(self, v): self._val = max(-180, min(180, round(v))); self._draw()


# ══════════════════════════════════════════════════════════════════════
class MacroEngine:
    def __init__(self, arm, physics):
        self.arm = arm; self.phys = physics
        self.active = False; self._steps = []; self._si = 0
        self._tgt = {}; self._st = {}; self._t = 0.; self._dur = 0.
        self._loop_start = -1; self._loop_count = 0; self._loops_left = 0

    def _run(self, steps, msg="Macro"):
        """Start a sequence of steps. Each step is (kind, param, duration).
        Kinds: 'open', 'close', 'move' (IK pos), 'angles' (dict of angles),
               'loop_start' (begin loop, param=count), 'loop_end'."""
        self._steps = steps; self._si = 0
        self.active = True; self._t = 0.
        self._loop_start = -1; self._loops_left = 0
        self._begin()
        return msg

    def start_pickup(self, color):
        b = self.phys.find_by_color(color)
        if not b: return f"No '{color}' block"
        bp = b.pos.copy()
        return self._run([
            ("open",None,.3), ("move",bp+[0,80,0],.8),
            ("move",bp+[0,10,0],.5), ("close",None,.3),
            ("move",bp+[0,120,0],.6)
        ], f"Picking up {b.label}...")

    def start_stack(self, color1, color2):
        """Pick up color1 and stack it on top of color2."""
        b1 = self.phys.find_by_color(color1)
        b2 = self.phys.find_by_color(color2)
        if not b1: return f"No '{color1}' block"
        if not b2: return f"No '{color2}' block"
        p1 = b1.pos.copy(); p2 = b2.pos.copy()
        stack_h = b2.size * 2 + 15  # on top of b2
        return self._run([
            ("open",None,.2), ("move",p1+[0,80,0],.7),
            ("move",p1+[0,10,0],.4), ("close",None,.3),
            ("move",p1+[0,120,0],.5),
            ("move",p2+[0,stack_h+60,0],.7),
            ("move",p2+[0,stack_h,0],.4),
            ("open",None,.3), ("move",p2+[0,120,0],.5),
        ], f"Stacking {b1.label} on {b2.label}...")

    def start_wave(self):
        """Wave the arm back and forth."""
        home = {j: self.arm.angles[j] for j in self.arm.JOINTS}
        return self._run([
            ("angles", dict(home, J2=20, J3=40, J4=0, J6=73), .6),
            ("loop_start", 3, 0),
            ("angles", dict(home, J1=40, J2=20, J3=140, J4=30, J6=73), .35),
            ("angles", dict(home, J1=-40, J2=20, J3=140, J4=-30, J6=73), .35),
            ("loop_end", None, 0),
            ("angles", dict(home, J1=0, J2=20, J3=40, J4=0, J6=73), .3),
            ("angles", home, .5),
        ], "Waving...")

    def start_bow(self):
        """Polite bow gesture."""
        home = {j: self.arm.angles[j] for j in self.arm.JOINTS}
        return self._run([
            ("angles", dict(home, J2=30, J3=100), .5),
            ("angles", dict(home, J2=100, J3=130, J4=40), .7),
            ("angles", dict(home, J2=100, J3=130, J4=40), .6),  # hold
            ("angles", dict(home, J2=30, J3=100), .5),
            ("angles", home, .5),
        ], "Bowing...")

    def start_spin_show(self):
        """Spin the base 360° with arm extended, showing off reach."""
        home = {j: self.arm.angles[j] for j in self.arm.JOINTS}
        return self._run([
            ("angles", dict(home, J2=50, J3=80, J4=10, J6=10), .6),
            ("angles", dict(home, J1=90, J2=50, J3=80, J4=10, J6=10), .8),
            ("angles", dict(home, J1=-90, J2=50, J3=80, J4=10, J6=10), 1.6),
            ("angles", dict(home, J1=0, J2=50, J3=80, J4=10, J6=10), .8),
            ("open", None, .3),
            ("angles", home, .5),
        ], "Spin show...")

    def start_flex(self):
        """Curl and extend like flexing a muscle."""
        home = {j: self.arm.angles[j] for j in self.arm.JOINTS}
        return self._run([
            ("angles", dict(home, J2=10, J3=40, J6=10), .5),
            ("loop_start", 2, 0),
            ("angles", dict(home, J2=90, J3=140, J4=60, J6=10), .4),
            ("angles", dict(home, J2=10, J3=40, J4=-20, J6=10), .4),
            ("loop_end", None, 0),
            ("angles", dict(home, J2=10, J3=40, J6=73), .3),
            ("angles", home, .5),
        ], "Flexing...")

    def start_nod_yes(self):
        """Nod the wrist up and down (yes gesture)."""
        home = {j: self.arm.angles[j] for j in self.arm.JOINTS}
        return self._run([
            ("angles", dict(home, J2=30, J3=50), .4),
            ("loop_start", 3, 0),
            ("angles", dict(home, J2=30, J3=140, J4=40), .2),
            ("angles", dict(home, J2=30, J3=140, J4=-20), .2),
            ("loop_end", None, 0),
            ("angles", dict(home, J2=30, J3=50, J4=0), .2),
            ("angles", home, .5),
        ], "Nodding yes...")

    def start_shake_no(self):
        """Shake the wrist side to side (no gesture) using J5 rotation."""
        home = {j: self.arm.angles[j] for j in self.arm.JOINTS}
        return self._run([
            ("angles", dict(home, J2=20, J3=45), .4),
            ("loop_start", 3, 0),
            ("angles", dict(home, J2=20, J3=80, J5=60), .2),
            ("angles", dict(home, J2=20, J3=80, J5=-60), .2),
            ("loop_end", None, 0),
            ("angles", dict(home, J2=20, J3=45, J5=0), .2),
            ("angles", home, .5),
        ], "Shaking no...")

    def start_pickup_wave(self, color):
        """Pick up a block, wave it around, then place it back."""
        b = self.phys.find_by_color(color)
        if not b: return f"No '{color}' block"
        bp = b.pos.copy()
        return self._run([
            ("open",None,.2), ("move",bp+[0,80,0],.6),
            ("move",bp+[0,10,0],.4), ("close",None,.3),
            ("move",bp+[0,120,0],.5),
            # Wave with block
            ("loop_start", 2, 0),
            ("angles", {"J1":50,"J2":30,"J3":50,"J4":20,"J5":0,"J6":10}, .35),
            ("angles", {"J1":-50,"J2":30,"J3":50,"J4":-20,"J5":0,"J6":10}, .35),
            ("loop_end", None, 0),
            ("angles", {"J1":0,"J2":30,"J3":50,"J4":0,"J5":0,"J6":10}, .3),
            # Place back
            ("move",bp+[0,80,0],.6),
            ("move",bp+[0,15,0],.4),
            ("open",None,.3), ("move",bp+[0,100,0],.4),
        ], f"Wave with {b.label}...")

    def start_toss(self, color):
        """Pick up a block and toss it forward."""
        b = self.phys.find_by_color(color)
        if not b: return f"No '{color}' block"
        bp = b.pos.copy()
        return self._run([
            ("open",None,.2), ("move",bp+[0,80,0],.5),
            ("move",bp+[0,10,0],.3), ("close",None,.25),
            ("move",bp+[0,80,0],.3),
            # Wind up
            ("angles", {"J1":0,"J2":80,"J3":130,"J4":50,"J5":0,"J6":10}, .4),
            # Throw! (fast forward swing + open claw)
            ("angles", {"J1":0,"J2":20,"J3":50,"J4":-30,"J5":0,"J6":73}, .15),
            # Follow through
            ("angles", {"J1":0,"J2":10,"J3":40,"J4":-40,"J5":0,"J6":73}, .3),
        ], f"Tossing {b.label}!")

    def _begin(self):
        if self._si >= len(self._steps):
            self.active = False; return
        k, p, d = self._steps[self._si]

        # Loop control
        if k == "loop_start":
            self._loop_start = self._si
            self._loops_left = p  # number of iterations
            self._si += 1; self._begin(); return
        if k == "loop_end":
            self._loops_left -= 1
            if self._loops_left > 0:
                self._si = self._loop_start + 1  # jump back after loop_start
            else:
                self._si += 1
            self._begin(); return

        self._dur = d; self._t = 0.
        self._st = {j: self.arm.angles[j] for j in self.arm.JOINTS}

        if k == "open":
            self._tgt = dict(self._st); self._tgt["J6"] = 73.
        elif k == "close":
            self._tgt = dict(self._st); self._tgt["J6"] = 10.
        elif k == "move" and p is not None:
            ik = self.arm.solve_angles_for_position(p)
            self._tgt = ik if ik else dict(self._st)
        elif k == "angles" and isinstance(p, dict):
            self._tgt = dict(self._st)
            self._tgt.update(p)
        else:
            self._tgt = dict(self._st)

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

# Key → (joint, direction)
KEY_MAP = {
    'a': ('J1', +1), 'd': ('J1', -1),
    'w': ('J2', -1), 's': ('J2', +1),
    'q': ('J3', -1), 'e': ('J3', +1),
    'r': ('J4', -1), 'f': ('J4', +1),
    't': ('J5', -1), 'g': ('J5', +1),
    'z': ('J6', -1), 'x': ('J6', +1),
}


# ══════════════════════════════════════════════════════════════════════
class App:

    def __init__(self, root):
        self.root = root
        root.title("MARS — Keyboard Robot Arm Sim")
        root.configure(bg=BG); root.geometry("1200x850"); root.minsize(900,600)
        root.report_callback_exception = self._on_err

        self.arm = ArmKinematics()
        self.physics = BlockPhysics(); self.physics.spawn_default_set()
        self.macro = MacroEngine(self.arm, self.physics)

        self._view_elev = 25.; self._view_azim = -60.
        self._view_mode = "PCA"  # "MANUAL", "PCA", "NORM", "LINEAR", "UNNORM"
        self._drag_active = False
        self._view_locked = False
        self._mpl_rotation_disabled = False

        # Claw move mode
        self._claw_move = False

        # Per-joint home offsets
        self._homes = {j: SERVO[j][2] for j in [f"J{i}" for i in range(1,7)]}

        # Held keys for continuous movement
        self._keys_held = set()
        self._move_speed = 5.0  # degrees per tick
        self._cam_smooth = 0.01  # camera blend factor (lower = smoother)

        self._last_time = time.time()

        self._style(); self._build(); self._init_3d(); self._draw_arm()
        self._log("Ready — keyboard control")
        self._log("A/D=J1  W/S=J2  Q/E=J3")
        self._log("R/F=J4  T/G=J5  Z/X=J6")
        self._tick()

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
        ttk.Label(top, text="MARS — Keyboard Control", style="T.TLabel").pack(side=tk.LEFT)
        tk.Button(top, text="? Help", font=("Consolas",9,"bold"), bg="#1e3a5c",
                  fg="white", activebackground="#264b73", relief=tk.FLAT,
                  padx=8, pady=1, command=self._show_help).pack(side=tk.RIGHT)

        body = ttk.Frame(self.root, style="D.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)
        body.columnconfigure(0, weight=5); body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # LEFT — 3D view (bigger now without camera panel)
        cf = ttk.LabelFrame(body, text=" 3D View ", style="D.TLabelframe")
        cf.grid(row=0, column=0, sticky="nsew", padx=(0,3))
        self.fig = Figure(figsize=(7,5), dpi=100, facecolor=BG)
        self.canvas3d = FigureCanvasTkAgg(self.fig, master=cf)
        self.canvas3d.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.ee_lbl = ttk.Label(cf, text="EE: --", style="S.TLabel")
        self.ee_lbl.pack(fill=tk.X, padx=3, pady=(0,1))
        # View cube
        vcf = ttk.Frame(cf, style="D.TFrame"); vcf.pack(fill=tk.X, padx=3, pady=(0,2))
        for nm in VIEW_PRESETS:
            tk.Button(vcf, text=nm, font=("Consolas",7,"bold"), bg="#1e293b",
                      fg="#94a3b8", activebackground="#334155", relief=tk.FLAT,
                      padx=3, pady=1, width=5,
                      command=lambda n=nm: self._set_view_preset(n)).pack(side=tk.LEFT, padx=1)
        self.btn_view = tk.Button(vcf, text="PCA", font=("Consolas",7,"bold"),
                                   bg="#2d3a2d", fg="white", relief=tk.FLAT,
                                   padx=3, pady=1, width=7, command=self._cycle_view_mode)
        self.btn_view.pack(side=tk.RIGHT, padx=1)
        self.btn_lock = tk.Button(vcf, text="LOCK", font=("Consolas",7,"bold"),
                                   bg="#2d2d2d", fg="#888", relief=tk.FLAT,
                                   padx=3, pady=1, command=self._toggle_lock_view)
        self.btn_lock.pack(side=tk.RIGHT, padx=1)

        # Keyboard help below view
        help_text = "A/D→J1  W/S→J2  Q/E→J3  |  R/F→J4  T/G→J5  Z/X→J6  |  H=Home  B=Block  1-4=Macro"
        ttk.Label(cf, text=help_text, style="D.TLabel",
                  font=("Consolas", 8)).pack(fill=tk.X, padx=3, pady=(0,2))

        # RIGHT — controls (scrollable)
        scroll_container = ttk.Frame(body, style="D.TFrame")
        scroll_container.grid(row=0, column=1, sticky="nsew", padx=(3,0))
        scroll_container.rowconfigure(0, weight=1)
        scroll_container.columnconfigure(0, weight=1)
        rf_canvas = tk.Canvas(scroll_container, bg=BG, highlightthickness=0)
        rf_canvas.grid(row=0, column=0, sticky="nsew")
        rf_scrollbar = ttk.Scrollbar(scroll_container, orient="vertical",
                                      command=rf_canvas.yview)
        rf_scrollbar.grid(row=0, column=1, sticky="ns")
        rf_canvas.configure(yscrollcommand=rf_scrollbar.set)
        rf = ttk.Frame(rf_canvas, style="D.TFrame")
        rf_window = rf_canvas.create_window((0, 0), window=rf, anchor="nw")
        rf.bind("<Configure>",
                lambda e: rf_canvas.configure(scrollregion=rf_canvas.bbox("all")))
        rf_canvas.bind("<Configure>",
                       lambda e: rf_canvas.itemconfigure(rf_window, width=e.width))
        def _on_mousewheel(event):
            if sys.platform == "win32":
                rf_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                rf_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        def _bind_scroll(e):
            rf_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            rf_canvas.bind_all("<Button-4>",
                               lambda e: rf_canvas.yview_scroll(-1, "units"))
            rf_canvas.bind_all("<Button-5>",
                               lambda e: rf_canvas.yview_scroll(1, "units"))
        def _unbind_scroll(e):
            rf_canvas.unbind_all("<MouseWheel>")
            rf_canvas.unbind_all("<Button-4>")
            rf_canvas.unbind_all("<Button-5>")
        scroll_container.bind("<Enter>", _bind_scroll)
        scroll_container.bind("<Leave>", _unbind_scroll)

        def btn(p, t, bg_c, cmd, **kw):
            b = tk.Button(p, text=t, font=("Consolas",8,"bold"), bg=bg_c,
                          fg="white", activebackground="#444", relief=tk.FLAT,
                          padx=4, pady=1, command=cmd, **kw)
            b.pack(fill=tk.X, padx=3, pady=1); return b

        bf = ttk.LabelFrame(rf, text=" Controls ", style="D.TLabelframe")
        bf.pack(fill=tk.X, pady=(0,2))
        self.btn_home = btn(bf, "Home (H)", "#333", self._go_home)
        self.btn_claw_move = btn(bf, "Claw Move: OFF", "#2d2d3a", self._toggle_claw_move)

        # ── Speed slider ─────────────────────────────────────────────
        spf = ttk.LabelFrame(rf, text=" Key Speed ", style="D.TLabelframe")
        spf.pack(fill=tk.X, pady=2)
        spr = ttk.Frame(spf, style="D.TFrame"); spr.pack(fill=tk.X, padx=4, pady=2)
        self.speed_lbl = ttk.Label(spr, text="5.0°/tick", width=8,
                                    style="D.TLabel", font=("Consolas",8))
        self.speed_lbl.pack(side=tk.RIGHT)
        self.speed_slider = tk.Scale(spr, from_=0.5, to=10.0, resolution=0.5,
            orient=tk.HORIZONTAL, bg=BG, fg=FG, troughcolor=PANEL,
            highlightbackground=BG, activebackground="#4a7c59", length=80,
            showvalue=False, command=self._on_speed)
        self.speed_slider.set(5.0)
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # ── Camera Smoothing slider ──────────────────────────────────
        csf = ttk.LabelFrame(rf, text=" Cam Smooth ", style="D.TLabelframe")
        csf.pack(fill=tk.X, pady=2)
        csr = ttk.Frame(csf, style="D.TFrame"); csr.pack(fill=tk.X, padx=4, pady=2)
        self.cam_smooth_lbl = ttk.Label(csr, text="1%", width=4,
                                         style="D.TLabel", font=("Consolas",8))
        self.cam_smooth_lbl.pack(side=tk.RIGHT)
        self.cam_smooth_slider = tk.Scale(csr, from_=1, to=30, resolution=1,
            orient=tk.HORIZONTAL, bg=BG, fg=FG, troughcolor=PANEL,
            highlightbackground=BG, activebackground="#4a7c59", length=80,
            showvalue=False, command=self._on_cam_smooth)
        self.cam_smooth_slider.set(1)
        self.cam_smooth_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # ── Home Offsets ─────────────────────────────────────────────
        hf = ttk.LabelFrame(rf, text=" Home Offsets ", style="D.TLabelframe")
        hf.pack(fill=tk.X, pady=2)
        self._home_dials = {}
        dr1 = ttk.Frame(hf, style="D.TFrame")
        dr1.pack(fill=tk.X, padx=2, pady=1)
        for jn, lbl in [("J1", "J1"), ("J5", "J5")]:
            box = ttk.Frame(dr1, style="D.TFrame")
            box.pack(side=tk.LEFT, expand=True)
            d = DegreeDial(box, size=64, value=0, label=lbl,
                           command=lambda v, j=jn: self._on_dial(j, v))
            d.pack(padx=1)
            self._home_dials[jn] = d
        dr2 = ttk.Frame(hf, style="D.TFrame")
        dr2.pack(fill=tk.X, padx=2, pady=1)
        for jn, lbl in [("J2", "J2"), ("J3", "J3"), ("J4", "J4")]:
            box = ttk.Frame(dr2, style="D.TFrame")
            box.pack(side=tk.LEFT, expand=True)
            d = DegreeDial(box, size=56, value=0, label=lbl,
                           command=lambda v, j=jn: self._on_dial(j, v))
            d.pack(padx=1)
            self._home_dials[jn] = d

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
        jl = {"J1":"J1 A/D","J2":"J2 W/S","J3":"J3 Q/E",
              "J4":"J4 R/F","J5":"J5 T/G","J6":"J6 Z/X"}
        for jn in ArmKinematics.JOINTS:
            lo,hi,hm = SERVO[jn]
            r = ttk.Frame(sf, style="D.TFrame"); r.pack(fill=tk.X, padx=3, pady=0)
            ttk.Label(r, text=f"{jl[jn]}:", width=7, style="D.TLabel",
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
        mf = ttk.LabelFrame(rf, text=" Pickup Macros ", style="D.TLabelframe")
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

        # ── Preset Motions ───────────────────────────────────────────
        motf = ttk.LabelFrame(rf, text=" Motions ", style="D.TLabelframe")
        motf.pack(fill=tk.X, pady=2)

        # Gestures row
        gr = ttk.Frame(motf, style="D.TFrame"); gr.pack(fill=tk.X, padx=3, pady=1)
        for i,(nm,cmd,col) in enumerate([
            ("Wave", lambda: self._run_motion("wave"), "#2563eb"),
            ("Bow", lambda: self._run_motion("bow"), "#7c3aed"),
            ("Nod", lambda: self._run_motion("nod"), "#0891b2"),
            ("Shake", lambda: self._run_motion("shake"), "#b45309"),
        ]):
            tk.Button(gr, text=nm, font=("Consolas",7,"bold"), bg=col, fg="white",
                      relief=tk.FLAT, padx=2, pady=0, width=4,
                      command=cmd).grid(row=0, column=i, sticky="ew", padx=1)
        gr.columnconfigure(0,weight=1); gr.columnconfigure(1,weight=1)
        gr.columnconfigure(2,weight=1); gr.columnconfigure(3,weight=1)

        # Tricks row
        tr = ttk.Frame(motf, style="D.TFrame"); tr.pack(fill=tk.X, padx=3, pady=1)
        for i,(nm,cmd,col) in enumerate([
            ("Spin", lambda: self._run_motion("spin"), "#dc2626"),
            ("Flex", lambda: self._run_motion("flex"), "#16a34a"),
        ]):
            tk.Button(tr, text=nm, font=("Consolas",7,"bold"), bg=col, fg="white",
                      relief=tk.FLAT, padx=2, pady=0, width=4,
                      command=cmd).grid(row=0, column=i, sticky="ew", padx=1)
        tr.columnconfigure(0,weight=1); tr.columnconfigure(1,weight=1)

        # Block tricks row
        btr = ttk.Frame(motf, style="D.TFrame"); btr.pack(fill=tk.X, padx=3, pady=1)
        ttk.Label(btr, text="Block:", style="D.TLabel",
                  font=("Consolas",7)).pack(side=tk.LEFT, padx=2)
        for nm,cmd,col in [
            ("Wave", lambda: self._run_motion("block_wave"), "#d97706"),
            ("Toss", lambda: self._run_motion("toss"), "#e11d48"),
            ("Stack", lambda: self._run_motion("stack"), "#059669"),
        ]:
            tk.Button(btr, text=nm, font=("Consolas",7,"bold"), bg=col, fg="white",
                      relief=tk.FLAT, padx=2, pady=0, width=4,
                      command=cmd).pack(side=tk.LEFT, padx=1)

        btn(motf, "Stop Motion", "#5c1a1a", self._stop_motion)

        # ── Active keys display ──────────────────────────────────────
        self.keys_lbl = ttk.Label(rf, text="Keys: -", style="D.TLabel",
                                   font=("Consolas", 9))
        self.keys_lbl.pack(fill=tk.X, padx=4, pady=2)

        # ── Log ──────────────────────────────────────────────────────
        logf = ttk.LabelFrame(rf, text=" Log ", style="D.TLabelframe")
        logf.pack(fill=tk.X, pady=(2,0))
        self.log_w = tk.Text(logf, bg="#0d1117", fg="#58a6ff",
                             font=("Consolas",7), height=6, width=20,
                             wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT)
        self.log_w.pack(fill=tk.X, padx=3, pady=3)

    # ══════════════════════════════════════════════════════════════════
    # 3D
    # ══════════════════════════════════════════════════════════════════
    def _init_3d(self):
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#0d1117")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas3d.mpl_connect('button_press_event', self._on_3d_press)
        self.canvas3d.mpl_connect('button_release_event', self._on_3d_release)

    def _disable_mpl_rotation(self):
        if not self._mpl_rotation_disabled:
            try: self.ax.disable_mouse_rotation()
            except: pass
            self._mpl_rotation_disabled = True

    def _enable_mpl_rotation(self):
        if self._mpl_rotation_disabled:
            try: self.ax.mouse_init()
            except: pass
            self._mpl_rotation_disabled = False

    def _screen_to_ground(self, event):
        if event.x is None or event.y is None: return None
        ax = self.ax; M = ax.get_proj(); ph = self.arm.platform_h
        best_dist = float('inf'); best_x = best_z = 0.0
        for gx in np.linspace(-250, 250, 26):
            for gz in np.linspace(-250, 250, 26):
                x2, y2, _ = proj3d.proj_transform(gx, gz, ph, M)
                dx, dy = ax.transData.transform((x2, y2))
                d = (dx - event.x)**2 + (dy - event.y)**2
                if d < best_dist: best_dist = d; best_x, best_z = gx, gz
        for gx in np.linspace(best_x-20, best_x+20, 21):
            for gz in np.linspace(best_z-20, best_z+20, 21):
                x2, y2, _ = proj3d.proj_transform(gx, gz, ph, M)
                dx, dy = ax.transData.transform((x2, y2))
                d = (dx - event.x)**2 + (dy - event.y)**2
                if d < best_dist: best_dist = d; best_x, best_z = gx, gz
        return best_x, best_z

    def _on_3d_press(self, event):
        if event.button != 1: return
        if self._claw_move:
            result = self._screen_to_ground(event)
            if result:
                gx, gz = result
                r = math.sqrt(gx**2 + gz**2)
                max_r = J2_J3 + J3_J4 - 30
                if r > max_r and r > 1:
                    gx *= max_r / r; gz *= max_r / r
                target = np.array([gx, 25.0, gz])
                ik = self.arm.solve_angles_for_position(target)
                if ik:
                    for j, v in ik.items(): self.arm.set_angle(j, v)
                    self.arm.enforce_ground_constraint()
                    self._sync_sliders()
                    self._log(f"Move → X={gx:.0f} Z={gz:.0f}")
                else: self._log("Unreachable")
                self._draw_arm()
            return
        if not self._view_locked:
            self._drag_active = True; self._enable_mpl_rotation()
        else: self._disable_mpl_rotation()

    def _on_3d_release(self, event):
        self._drag_active = False

    def _toggle_claw_move(self):
        self._claw_move = not self._claw_move
        if self._claw_move: self._disable_mpl_rotation()
        elif not self._view_locked: self._enable_mpl_rotation()
        self.btn_claw_move.config(
            text=f"Claw Move: {'ON' if self._claw_move else 'OFF'}",
            bg="#3a5c1a" if self._claw_move else "#2d2d3a")
        self._log(f"Claw move {'ON — click 3D' if self._claw_move else 'OFF'}")

    def _set_view_preset(self, name):
        if name in VIEW_PRESETS:
            self._view_elev, self._view_azim = VIEW_PRESETS[name]
            self._view_mode = "MANUAL"; self._view_locked = True
            self._disable_mpl_rotation()
            self.btn_view.config(text="MANUAL", bg="#3a2d2d")
            self.btn_lock.config(text="LOCKED", bg="#5c3a1a", fg="white")
            self._draw_arm()

    _VIEW_MODES = ["MANUAL", "BASIC", "PCA", "NORM", "LINEAR", "UNNORM"]
    _VIEW_COLORS = {"MANUAL": "#3a2d2d", "BASIC": "#2d2d2d", "PCA": "#5c3a1a",
                    "NORM": "#1a3a2d", "LINEAR": "#1a2d3a", "UNNORM": "#3a1a3a"}

    def _cycle_view_mode(self):
        """Cycle through: MANUAL → PCA → NORM → LINEAR → UNNORM → MANUAL..."""
        idx = self._VIEW_MODES.index(self._view_mode)
        self._view_mode = self._VIEW_MODES[(idx + 1) % len(self._VIEW_MODES)]
        is_auto = self._view_mode != "MANUAL"
        if is_auto:
            self._view_locked = False
            if not self._claw_move: self._enable_mpl_rotation()
            self.btn_lock.config(text="LOCK", bg="#2d2d2d", fg="#888")
        self.btn_view.config(text=self._view_mode,
                             bg=self._VIEW_COLORS[self._view_mode])
        self._log(f"View: {self._view_mode}")

    def _toggle_lock_view(self):
        self._view_locked = not self._view_locked
        if self._view_locked:
            self._view_mode = "MANUAL"
            self._disable_mpl_rotation()
            self.btn_lock.config(text="LOCKED", bg="#5c3a1a", fg="white")
            self.btn_view.config(text="MANUAL", bg="#3a2d2d")
        else:
            if not self._claw_move: self._enable_mpl_rotation()
            self.btn_lock.config(text="LOCK", bg="#2d2d2d", fg="#888")

    # ══════════════════════════════════════════════════════════════════
    # AUTO-CAMERA ALGORITHMS
    # ══════════════════════════════════════════════════════════════════

    def _compute_auto_view(self, pts):
        """Dispatch to the active auto-camera algorithm."""
        mode = self._view_mode
        if mode == "BASIC":
            return self._cam_basic(pts)
        elif mode == "PCA":
            return self._cam_pca(pts)
        elif mode == "NORM":
            return self._cam_ortho(pts, weighted=False)
        elif mode == "LINEAR":
            return self._cam_ortho(pts, weighted=True)
        elif mode == "UNNORM":
            return self._cam_unnorm(pts)
        return self._view_elev, self._view_azim

    def _eigvec_to_view(self, cam, pts):
        """Convert a 3D eigenvector (plot coords) to (elev, azim) for matplotlib.

        Azimuth: from the eigenvector (which horizontal direction minimizes info loss).
        Elevation: from the arm's geometry (height span vs horizontal reach).

        When the arm reaches out horizontally → look from above (high elev).
        When the arm reaches up vertically → look from the side (low elev).
        When compact/folded → moderate angle to see all joints.
        """
        # Handle eigenvector sign ambiguity — prefer looking from above
        if cam[2] < 0:
            cam = -cam

        # ── Azimuth from eigenvector ─────────────────────────────────
        azim = math.degrees(math.atan2(cam[1], cam[0]))

        # ── Elevation from arm geometry ──────────────────────────────
        # Plot coords: pp[:,0]=armX, pp[:,1]=armZ, pp[:,2]=armY(height)
        pp = np.array([[p[0], p[2], p[1]] for p in pts])

        # Horizontal reach: max distance from base in XZ plane
        base_xz = pp[0, :2]  # base X,Z
        horiz_dists = [np.linalg.norm(p[:2] - base_xz) for p in pp]
        h_reach = max(horiz_dists) if horiz_dists else 1.0

        # Vertical span: range of heights
        heights = pp[:, 2]
        v_span = heights.max() - heights.min()

        # Ratio: high h_reach relative to v_span → look from above
        #         high v_span relative to h_reach → look from side
        total = h_reach + v_span + 1e-6
        horiz_ratio = h_reach / total  # 0 = pure vertical, 1 = pure horizontal

        # Map: 0.0 (vertical arm) → 12° (side view)
        #       0.5 (balanced)     → 35° (isometric-ish)
        #       1.0 (horizontal)   → 65° (looking down)
        elev = 12 + horiz_ratio * 53

        # Also consider the eigenvector's vertical component as a nudge
        eig_horiz = math.sqrt(cam[0]**2 + cam[1]**2)
        eig_elev = math.degrees(math.atan2(abs(cam[2]), eig_horiz + 1e-9))
        # Blend: 70% geometry-based, 30% eigenvector-based
        elev = elev * 0.7 + eig_elev * 0.3

        elev = max(5, min(85, elev))
        return elev, azim

    def _cam_basic(self, pts):
        """Simple base-to-end-effector perpendicular camera.
        Azimuth: 90° offset from the base→tip direction in XZ plane.
        Elevation: fixed based on arm height vs reach ratio.
        No eigendecomposition — just geometry."""
        pp = np.array([[p[0], p[2], p[1]] for p in pts])
        # Base-to-tip direction in ground plane
        rg = np.array([pp[-1][0] - pp[0][0], pp[-1][1] - pp[0][1]])
        rl = np.linalg.norm(rg)
        # Azimuth: perpendicular to base→tip line
        azim = (math.degrees(math.atan2(rg[1], rg[0])) - 90) if rl > 1e-3 else self._view_azim
        # Elevation: height range vs horizontal reach
        hs = [p[2] for p in pp]; hr = max(hs) - min(hs)
        v = hr / (hr + max(rl, 1.))
        elev = 15 + v * 35
        return max(8, min(65, elev)), azim

    def _cam_pca(self, pts):
        """PCA on joint positions — camera along min-variance eigenvector.
        Maximizes spatial spread of projected points on screen.
        Implicitly weights longer links more (positions farther apart)."""
        pp = np.array([[p[0], p[2], p[1]] for p in pts])
        centroid = pp.mean(axis=0)
        centered = pp - centroid
        cov = centered.T @ centered / len(pp)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        return self._eigvec_to_view(eigenvectors[:, 0], pts)

    def _cam_ortho(self, pts, weighted=False):
        """Orthogonality on link directions — camera perpendicular to all links.
        Normalized (weighted=False): M = Σ d̂ᵢd̂ᵢᵀ — equal weight per link.
        Linear (weighted=True): M = Σ Lᵢ·d̂ᵢd̂ᵢᵀ — longer links matter more.
        Camera along minimum eigenvector of M."""
        pp = np.array([[p[0], p[2], p[1]] for p in pts])
        M = np.zeros((3, 3))
        for i in range(len(pp) - 1):
            d = pp[i+1] - pp[i]
            length = np.linalg.norm(d)
            if length < 1e-6: continue
            u = d / length
            if weighted:
                M += length * np.outer(u, u)
            else:
                M += np.outer(u, u)
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        return self._eigvec_to_view(eigenvectors[:, 0], pts)

    def _cam_unnorm(self, pts):
        """Unnormalized: M = Σ dᵢdᵢᵀ = Σ Lᵢ²·d̂ᵢd̂ᵢᵀ.
        Long links dominate quadratically. Good when one link
        is much more important than others."""
        pp = np.array([[p[0], p[2], p[1]] for p in pts])
        M = np.zeros((3, 3))
        for i in range(len(pp) - 1):
            d = pp[i+1] - pp[i]
            if np.linalg.norm(d) < 1e-6: continue
            M += np.outer(d, d)
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        return self._eigvec_to_view(eigenvectors[:, 0], pts)

    # ══════════════════════════════════════════════════════════════════
    def _draw_arm(self):
        try:
            ax = self.ax
            if hasattr(ax,'elev') and hasattr(ax,'azim'):
                if (self._view_mode == "MANUAL" or self._drag_active) and not self._view_locked:
                    self._view_elev = ax.elev; self._view_azim = ax.azim

            ax.clear(); ax.set_facecolor("#0d1117")
            pts = self.arm.forward_kinematics(); ph = self.arm.platform_h
            def p(v): return (v[0], v[2], v[1])

            # Platform
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

            # Home arrows
            yaw = np.radians(self.arm.angles["J1"])
            j1_home = self._homes.get("J1", 0)
            j1_rad = math.radians(j1_home)
            alen = 65
            apx = alen*math.cos(j1_rad); apy = alen*math.sin(j1_rad)
            ax.plot([apx*.25,apx],[apy*.25,apy],[ph,ph], color="#facc15", lw=2.5, alpha=.8)
            bpx = apx-12*math.cos(j1_rad); bpy = apy-12*math.sin(j1_rad)
            ppx = -8*math.sin(j1_rad); ppy = 8*math.cos(j1_rad)
            ax.plot([apx,bpx+ppx],[apy,bpy+ppy],[ph,ph], color="#facc15", lw=2, alpha=.6)
            ax.plot([apx,bpx-ppx],[apy,bpy-ppy],[ph,ph], color="#facc15", lw=2, alpha=.6)
            ax.text(apx*1.15, apy*1.15, ph+6, "J1H", color="#facc15", fontsize=5, ha="center", alpha=.5)

            arm_fwd = np.array([math.cos(yaw), 0, math.sin(yaw)])
            arm_up = np.array([0, 1, 0])
            for jn, idx, col, al in [("J2",1,"#22d3ee",25),("J3",2,"#22d3ee",20),("J4",3,"#a78bfa",18)]:
                offset = self._homes.get(jn, SERVO[jn][2]) - SERVO[jn][2]
                if abs(offset) > 0.5:
                    jpos = pts[idx]
                    off_rad = math.radians(offset)
                    direction = arm_fwd*math.sin(off_rad) + arm_up*math.cos(off_rad)
                    tip = jpos + direction * al
                    s = p(jpos); e = p(tip)
                    ax.plot([s[0],e[0]],[s[1],e[1]],[s[2],e[2]], color=col, lw=1.8, alpha=.6)
                    ax.text(e[0],e[1],e[2]+5, f"{jn}H", color=col, fontsize=4, ha="center", alpha=.5)

            # J5 arrow
            j5_home = self._homes.get("J5", 0)
            if abs(j5_home) > 0.5 and len(pts) > 4:
                j5_pos = pts[4]
                roll_rad = math.radians(j5_home)
                perp = np.array([-math.sin(yaw), 0, math.cos(yaw)])
                roll_dir = perp * math.cos(roll_rad) + arm_up * math.sin(roll_rad)
                tip5 = j5_pos + roll_dir * 22
                s5 = p(j5_pos); e5 = p(tip5)
                ax.plot([s5[0],e5[0]],[s5[1],e5[1]],[s5[2],e5[2]], color="#fb923c", lw=1.8, alpha=.6)
                ax.text(e5[0],e5[1],e5[2]+5, "J5H", color="#fb923c", fontsize=4, ha="center", alpha=.5)

            # Link boxes
            for lk in self.arm.link_boxes():
                fs = _box_faces(lk["start"], lk["end"], lk["width"])
                if fs:
                    rgb = _hex_rgb(lk["color"])
                    ax.add_collection3d(Poly3DCollection([[p(v) for v in f] for f in fs],
                        alpha=lk["alpha"], facecolor=rgb, edgecolor=(*rgb,.6), linewidths=.4))

            # Skeleton + joints
            xs=[pt[0] for pt in pts]; ys=[pt[2] for pt in pts]; zs=[pt[1] for pt in pts]
            ax.plot(xs,ys,zs, color="white", lw=.8, alpha=.3, ls="--")
            jc=["#475569","#0ea5e9","#0ea5e9","#6366f1","#a78bfa","#ec4899","#f472b6"]
            for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
                ax.scatter(x,y,z, color=jc[min(i,6)], s=40 if i<2 else 55-i*5,
                           edgecolors="white", linewidths=.2, zorder=5, depthshade=True)

            # Finger boxes
            for fb in self.arm.finger_boxes():
                fs = _box_faces(fb["start"], fb["end"], fb["width"])
                if fs:
                    rgb = _hex_rgb(fb["color"])
                    ax.add_collection3d(Poly3DCollection([[p(v) for v in f] for f in fs],
                        alpha=fb["alpha"], facecolor=rgb, edgecolor=(*rgb,.7), linewidths=.5))

            fg = self.arm.get_finger_geometry()
            for tk_ in ["left_tip","right_tip"]:
                t = fg[tk_]; b = fg["claw_base"]
                ax.plot([p(b)[0],p(t)[0]],[p(b)[1],p(t)[1]],[p(b)[2],p(t)[2]],
                        color="#f472b6", lw=.8, alpha=.4, ls="--")
            gc = fg["grip_center"]
            gc_col = "#51cf66" if fg["is_closed"] else "#ff6b6b"
            ax.scatter(*p(gc), color=gc_col, s=25, marker="o", zorder=7, depthshade=False, alpha=.6)

            # Joint axes
            for ja in self.arm.joint_axes():
                av = ja["axis"]*ja["length"]
                s = p(ja["pos"]-av*.5); e = p(ja["pos"]+av*.5)
                ax.plot([s[0],e[0]],[s[1],e[1]],[s[2],e[2]], color=ja["color"], lw=1.2, alpha=.5)

            # Blocks
            for bs in self.physics.get_block_states():
                brgb = _hex_rgb(bs["color"]); ba = .7 if bs["grabbed"] else .5
                ax.add_collection3d(Poly3DCollection([[p(v) for v in f] for f in bs["faces"]],
                    alpha=ba, facecolor=brgb,
                    edgecolor=(*brgb,.4) if not bs["grabbed"] else (1,1,1,.3), linewidths=.4))
                bp = bs["pos"]
                ax.text(p(bp)[0],p(bp)[1],p(bp)[2]+bs["size"]+6,
                        bs["label"], color=bs["color"], fontsize=4, ha="center", alpha=.7)

            # Ground grid
            for g in np.linspace(-200,200,9):
                ax.plot([g,g],[-200,200],[0,0], color="#1e293b", lw=.15, alpha=.3)
                ax.plot([-200,200],[g,g],[0,0], color="#1e293b", lw=.15, alpha=.3)

            reach = J2_J3+J3_J4+J4_J5
            tw = np.linspace(-np.pi/2, np.pi/2, 50)
            ax.plot(reach*np.cos(tw), reach*np.sin(tw), np.full(50,ph+J1_J2),
                    color="#334155", lw=.5, ls=":", alpha=.2)

            L = 280; zt = max(L+80, ph+L)
            ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_zlim(-20,zt)
            ax.tick_params(colors="#475569", labelsize=4)
            ax.set_xlabel("X", color="#475569", fontsize=5)
            ax.set_ylabel("Z", color="#475569", fontsize=5)
            ax.set_zlabel("Y", color="#475569", fontsize=5)

            if self._view_mode != "MANUAL" and not self._drag_active and not self._view_locked:
                te, ta = self._compute_auto_view(pts)
                bl = self._cam_smooth
                da = ta - self._view_azim
                if da > 180: da -= 360
                if da < -180: da += 360
                self._view_azim += da * bl
                self._view_elev += (te - self._view_elev) * bl

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
    # MAIN TICK (keyboard + physics)
    # ══════════════════════════════════════════════════════════════════
    def _tick(self):
        now = time.time()
        dt = min(now - self._last_time, 0.1)
        self._last_time = now

        # Process held keys
        moved = False
        if not self.macro.active:
            for key in list(self._keys_held):
                if key in KEY_MAP:
                    joint, direction = KEY_MAP[key]
                    cur = self.arm.angles[joint]
                    self.arm.set_angle(joint, cur + direction * self._move_speed)
                    moved = True
            if moved:
                self.arm.enforce_ground_constraint()
                self._sync_sliders()

        # Update keys display
        if self._keys_held:
            active = sorted(self._keys_held)
            parts = []
            for k in active:
                if k in KEY_MAP:
                    j, d = KEY_MAP[k]
                    parts.append(f"{j}{'+'if d>0 else '-'}")
            self.keys_lbl.config(text=f"Keys: {' '.join(parts)}" if parts else "Keys: -")
        else:
            self.keys_lbl.config(text="Keys: -")

        # Macro
        if self.macro.active:
            self.macro.update(dt)
            self._sync_sliders()

        # Physics
        fg = self.arm.get_finger_geometry()
        fg["grip_velocity"] = self.arm._grip_velocity.copy()
        self.physics.step(dt, fg)

        # Redraw if anything changed
        if moved or self.macro.active or any(not b.resting or b.grabbed for b in self.physics.blocks):
            self._draw_arm()

        self.root.after(33, self._tick)

    # ══════════════════════════════════════════════════════════════════
    # ACTIONS
    # ══════════════════════════════════════════════════════════════════
    def _on_dial(self, joint, offset_deg):
        absolute = SERVO[joint][2] + offset_deg
        absolute = max(SERVO[joint][0], min(SERVO[joint][1], absolute))
        self._homes[joint] = absolute
        self._draw_arm()

    def _on_speed(self, v):
        self._move_speed = float(v)
        self.speed_lbl.config(text=f"{float(v):.1f}°/tick")

    def _on_cam_smooth(self, v):
        self._cam_smooth = int(float(v)) / 100.0
        self.cam_smooth_lbl.config(text=f"{int(float(v))}%")

    def _go_home(self):
        self.macro.cancel()
        for j in self.arm.JOINTS:
            self.arm.set_angle(j, self._homes.get(j, SERVO[j][2]))
        self.arm.enforce_ground_constraint()
        self._sync_sliders(); self._draw_arm()

    def _slider(self, j, v):
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
        for j, sc in self.sliders.items():
            a = self.arm.angles[j]; sc.set(a)
            self.slbl[j].config(text=f"{a:.0f}")

    def _spawn_block(self):
        b = self.physics.spawn_block(); self._log(f"Spawned {b.label}"); self._draw_arm()
    def _clear_blocks(self):
        self.physics.clear(); self._log("Cleared"); self._draw_arm()
    def _run_macro(self, color):
        if self.macro.active: self.macro.cancel()
        self._log(self.macro.start_pickup(color))

    def _run_motion(self, name):
        """Dispatch preset motions."""
        if self.macro.active: self.macro.cancel()
        if name == "wave":
            self._log(self.macro.start_wave())
        elif name == "bow":
            self._log(self.macro.start_bow())
        elif name == "nod":
            self._log(self.macro.start_nod_yes())
        elif name == "shake":
            self._log(self.macro.start_shake_no())
        elif name == "spin":
            self._log(self.macro.start_spin_show())
        elif name == "flex":
            self._log(self.macro.start_flex())
        elif name == "block_wave":
            # Find first available block
            for b in self.physics.blocks:
                if not b.grabbed:
                    self._log(self.macro.start_pickup_wave(b.label))
                    return
            self._log("No blocks available")
        elif name == "toss":
            for b in self.physics.blocks:
                if not b.grabbed:
                    self._log(self.macro.start_toss(b.label))
                    return
            self._log("No blocks available")
        elif name == "stack":
            ungrabbed = [b for b in self.physics.blocks if not b.grabbed]
            if len(ungrabbed) >= 2:
                self._log(self.macro.start_stack(ungrabbed[0].label, ungrabbed[1].label))
            else:
                self._log("Need 2+ blocks to stack")

    def _stop_motion(self):
        self.macro.cancel()
        self._log("Motion stopped")

    def _log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line)
        try:
            self.log_w.config(state=tk.NORMAL)
            self.log_w.insert(tk.END, line+"\n")
            self.log_w.see(tk.END); self.log_w.config(state=tk.DISABLED)
        except: pass

    def _on_err(self, *a):
        print(f"[TK] {''.join(traceback.format_exception(*a))}", file=sys.stderr)

    def _show_help(self):
        """Open a help window displaying the project README.md."""
        import os, re

        hw = tk.Toplevel(self.root)
        hw.title("MARS — Help & Guide")
        hw.configure(bg=BG)
        hw.geometry("720x850")
        hw.minsize(600, 500)

        # Scrollable text
        frame = tk.Frame(hw, bg=BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frame, bg="#0d1117", fg="#c9d1d9", font=("Consolas", 9),
                      wrap=tk.WORD, yscrollcommand=sb.set, relief=tk.FLAT,
                      insertbackground="#c9d1d9", padx=14, pady=14,
                      spacing1=1, spacing3=1)
        txt.pack(fill=tk.BOTH, expand=True)
        sb.config(command=txt.yview)

        # Tags for markdown rendering
        txt.tag_configure("h1", font=("Consolas", 16, "bold"), foreground=ACCENT,
                          spacing1=8, spacing3=4)
        txt.tag_configure("h2", font=("Consolas", 13, "bold"), foreground="#f59e0b",
                          spacing1=12, spacing3=4)
        txt.tag_configure("h3", font=("Consolas", 11, "bold"), foreground="#10b981",
                          spacing1=8, spacing3=2)
        txt.tag_configure("h4", font=("Consolas", 10, "bold"), foreground="#818cf8",
                          spacing1=6, spacing3=2)
        txt.tag_configure("code", font=("Consolas", 9), foreground="#7dd3fc",
                          background="#161b22")
        txt.tag_configure("codeblock", font=("Consolas", 8), foreground="#7dd3fc",
                          background="#161b22", lmargin1=20, lmargin2=20,
                          spacing1=0, spacing3=0)
        txt.tag_configure("bold", font=("Consolas", 9, "bold"), foreground="#e2e8f0")
        txt.tag_configure("hr", foreground="#30363d")
        txt.tag_configure("table", font=("Consolas", 8), foreground="#c9d1d9",
                          background="#0d1117", lmargin1=10, lmargin2=10)
        txt.tag_configure("tablehead", font=("Consolas", 8, "bold"),
                          foreground="#f59e0b", background="#0d1117",
                          lmargin1=10, lmargin2=10)
        txt.tag_configure("dim", foreground="#6e7681")
        txt.tag_configure("body", font=("Consolas", 9), foreground="#c9d1d9")

        # Find README.md
        readme_path = None
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in [
            os.path.join(script_dir, "README.md"),
            os.path.join(os.getcwd(), "README.md"),
        ]:
            if os.path.exists(candidate):
                readme_path = candidate
                break

        if not readme_path:
            txt.insert(tk.END, "README.md not found.\n\n", "h2")
            txt.insert(tk.END, f"Looked in:\n  {script_dir}\n  {os.getcwd()}\n", "dim")
            txt.insert(tk.END, "\nPlace README.md next to main.py.", "body")
            txt.config(state=tk.DISABLED)
            tk.Button(hw, text="Close", font=("Consolas", 10, "bold"),
                      bg="#333", fg="white", relief=tk.FLAT, padx=20, pady=4,
                      command=hw.destroy).pack(pady=(0, 10))
            return

        with open(readme_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Simple markdown renderer
        in_code_block = False
        i = 0
        while i < len(lines):
            line = lines[i].rstrip("\n")

            # Code block toggle
            if line.startswith("```"):
                in_code_block = not in_code_block
                if in_code_block:
                    txt.insert(tk.END, "\n")
                else:
                    txt.insert(tk.END, "\n")
                i += 1
                continue

            if in_code_block:
                txt.insert(tk.END, line + "\n", "codeblock")
                i += 1
                continue

            # Horizontal rule
            if line.strip() in ("---", "***", "___"):
                txt.insert(tk.END, "─" * 60 + "\n", "hr")
                i += 1
                continue

            # Headers
            if line.startswith("#### "):
                txt.insert(tk.END, line[5:] + "\n", "h4")
                i += 1
                continue
            if line.startswith("### "):
                txt.insert(tk.END, line[4:] + "\n", "h3")
                i += 1
                continue
            if line.startswith("## "):
                txt.insert(tk.END, line[3:] + "\n", "h2")
                i += 1
                continue
            if line.startswith("# "):
                txt.insert(tk.END, line[2:] + "\n", "h1")
                i += 1
                continue

            # Table rows
            if "|" in line and line.strip().startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                # Skip separator rows (|---|---|)
                if all(set(c.strip()) <= set("-: ") for c in cells):
                    i += 1
                    continue
                # Check if header row (next line is separator)
                is_header = False
                if i + 1 < len(lines):
                    next_l = lines[i+1].strip()
                    if "|" in next_l:
                        next_cells = [c.strip() for c in next_l.strip().strip("|").split("|")]
                        if all(set(c.strip()) <= set("-: ") for c in next_cells):
                            is_header = True
                row_text = "  ".join(f"{c:<20s}" if len(c) < 20 else c for c in cells)
                tag = "tablehead" if is_header else "table"
                txt.insert(tk.END, row_text + "\n", tag)
                i += 1
                continue

            # Empty line
            if not line.strip():
                txt.insert(tk.END, "\n")
                i += 1
                continue

            # Regular text with inline formatting
            self._render_inline(txt, line)
            txt.insert(tk.END, "\n")
            i += 1

        txt.config(state=tk.DISABLED)

        # Close button
        tk.Button(hw, text="Close", font=("Consolas", 10, "bold"),
                  bg="#333", fg="white", relief=tk.FLAT, padx=20, pady=4,
                  command=hw.destroy).pack(pady=(0, 10))

    def _render_inline(self, txt, line):
        """Render a line with inline markdown: **bold**, `code`, regular text."""
        import re
        parts = re.split(r'(\*\*[^*]+\*\*|`[^`]+`)', line)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                txt.insert(tk.END, part[2:-2], "bold")
            elif part.startswith("`") and part.endswith("`"):
                txt.insert(tk.END, part[1:-1], "code")
            else:
                txt.insert(tk.END, part, "body")

    def close(self):
        self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.close)

    # Key press/release for held-key continuous movement
    def on_key_press(e):
        k = e.char.lower() if e.char else ''
        if k in KEY_MAP:
            app._keys_held.add(k)
    def on_key_release(e):
        k = e.char.lower() if e.char else ''
        app._keys_held.discard(k)

    root.bind("<KeyPress>", on_key_press)
    root.bind("<KeyRelease>", on_key_release)

    # Single-press bindings
    root.bind("<h>", lambda e: app._go_home())
    root.bind("<b>", lambda e: app._spawn_block())
    root.bind("<v>", lambda e: app._cycle_view_mode())
    root.bind("<L>", lambda e: app._toggle_lock_view())  # Shift+L to avoid J5 conflict
    root.bind("<Escape>", lambda e: app.close())
    root.bind("<F1>", lambda e: app._show_help())
    root.bind("1", lambda e: app._run_macro("Red"))
    root.bind("2", lambda e: app._run_macro("Blue"))
    root.bind("3", lambda e: app._run_macro("Green"))
    root.bind("4", lambda e: app._run_macro("Yellow"))
    root.mainloop()

if __name__ == "__main__":
    main()