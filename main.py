"""
MARS — Two-Hand Control, Smooth Rendering, Block Pickup Macros

Left hand  → J1 base, J2 shoulder, J3 elbow (position)
Right hand → J4 wrist, J5 rotation, J6 claw (gestures)
"""
import sys,time,traceback,math
import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image,ImageTk
import matplotlib; matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import (SERVO,HOME,CAM_INDEX,CAM_W,CAM_H,
    J1_J2,J2_J3,J3_J4,J4_J5,J5_J6,CLAW,
    PLATFORM_H,PLATFORM_H_MIN,PLATFORM_H_MAX,PLATFORM_W,PLATFORM_D,
    BG,FG,ACCENT,ACCENT2,ACCENT3,PANEL)
from arm_kinematics import ArmKinematics
from body_tracker import HandPairTracker,detect_camera_fov
from block_physics import BlockPhysics

def _hex_rgb(h):
    h=h.lstrip("#"); return tuple(int(h[i:i+2],16)/255. for i in(0,2,4))

def _box_faces(p0,p1,width):
    d=p1-p0; l=np.linalg.norm(d)
    if l<.1: return []
    dn=d/l; up=np.array([0,1,0])
    if abs(np.dot(dn,up))>.99: up=np.array([1,0,0])
    a=np.cross(dn,up); a/=np.linalg.norm(a)+1e-9
    b=np.cross(dn,a); b/=np.linalg.norm(b)+1e-9
    hw=width/2.
    c=[base+a*s1+b*s2 for base in[p0,p1] for s1 in[-hw,hw] for s2 in[-hw,hw]]
    return [[c[0],c[1],c[3],c[2]],[c[4],c[5],c[7],c[6]],
            [c[0],c[1],c[5],c[4]],[c[2],c[3],c[7],c[6]],
            [c[0],c[2],c[6],c[4]],[c[1],c[3],c[7],c[5]]]

# ══════════════════════════════════════════════════════════════════════
# MACRO ENGINE — animated pickup sequences
# ══════════════════════════════════════════════════════════════════════
class MacroEngine:
    """Animates arm to pick up a block by color."""
    def __init__(self,arm:ArmKinematics,physics:BlockPhysics):
        self.arm=arm; self.phys=physics
        self.active=False; self._steps=[]; self._step_i=0
        self._target={}; self._start={}; self._t=0.; self._dur=0.

    def start_pickup(self,color_name:str)->str:
        """Begin auto-pickup macro. Returns status message."""
        block=self.phys.find_by_color(color_name)
        if not block: return f"No '{color_name}' block found"
        bp=block.pos.copy()
        # Plan: open claw → move above → lower → close → lift → done
        approach=bp.copy(); approach[1]=bp[1]+80  # 80mm above block
        grab_pos=bp.copy(); grab_pos[1]=bp[1]+10  # just above
        lift_pos=bp.copy(); lift_pos[1]=bp[1]+120

        self._steps=[
            ("open",  None, .3),    # open claw
            ("move",  approach, .8),  # move above block
            ("move",  grab_pos, .5),  # lower to block
            ("close", None, .3),    # close claw
            ("move",  lift_pos, .6),  # lift
        ]
        self._step_i=0; self.active=True; self._t=0.
        self._begin_step()
        return f"Picking up {block.label}..."

    def _begin_step(self):
        if self._step_i>=len(self._steps):
            self.active=False; return
        kind,param,dur=self._steps[self._step_i]
        self._dur=dur; self._t=0.
        self._start={j:self.arm.angles[j] for j in self.arm.JOINTS}
        if kind=="open":
            self._target=dict(self._start); self._target["J6"]=73.
        elif kind=="close":
            self._target=dict(self._start); self._target["J6"]=10.
        elif kind=="move" and param is not None:
            ik=self.arm.solve_angles_for_position(param)
            if ik: self._target=ik
            else: self._target=dict(self._start)
        else:
            self._target=dict(self._start)

    def update(self,dt:float):
        """Call every frame. Returns True if still animating."""
        if not self.active: return False
        self._t+=dt; frac=min(1.,self._t/self._dur)
        # Smooth ease-in-out
        frac=frac*frac*(3-2*frac)
        for j in self.arm.JOINTS:
            if j in self._start and j in self._target:
                v=self._start[j]*(1-frac)+self._target[j]*frac
                self.arm.set_angle(j,v)
        self.arm.enforce_ground_constraint()
        if frac>=1.:
            self._step_i+=1; self._begin_step()
        return self.active

    def cancel(self):
        self.active=False; self._steps=[]


# ══════════════════════════════════════════════════════════════════════
class App:
    _3D_THROTTLE=3  # only redraw 3D every Nth frame

    def __init__(self,root):
        self.root=root; root.title("MARS — Two-Hand Robot Arm Sim")
        root.configure(bg=BG); root.geometry("1550x900"); root.minsize(1100,650)
        root.report_callback_exception=self._on_err

        self.arm=ArmKinematics()
        self.tracker=None; self._tracker_err=None
        try: self.tracker=HandPairTracker()
        except Exception as e: self._tracker_err=str(e)

        self.cap=None; self.running=False; self.tracking_on=False
        self._last=None; self._ftimes=[]; self._last_time=time.time()
        self._frame_count=0

        self.physics=BlockPhysics(); self.physics.spawn_default_set()
        self.macro=MacroEngine(self.arm,self.physics)

        self._view_elev=25.; self._view_azim=-60.
        self._auto_track=True; self._drag_active=False

        self._style(); self._build(); self._init_3d(); self._draw_arm()
        if self._tracker_err:
            self._log("Tracker FAILED:")
            for l in self._tracker_err.split("\n"):
                if l.strip(): self._log(f"  {l.strip()}")
        else: self._log("Ready — two-hand control")
        self._physics_tick()

    def _style(self):
        s=ttk.Style(); s.theme_use("clam")
        for n,bg_c in[("D.TFrame",BG),("D.TLabel",BG),("T.TLabel",BG),
                       ("S.TLabel",PANEL),("D.TLabelframe",BG),("D.TLabelframe.Label",BG)]:
            kw=dict(background=bg_c)
            if "Label" in n and "frame" not in n.lower():
                kw["foreground"]=ACCENT if "T." in n else (ACCENT3 if "S." in n else FG)
                kw["font"]=("Consolas",12 if "T." in n else 9 if "S." in n else 10,"bold" if "T." in n else "")
            if "frame" in n.lower() and "Label" not in n.split(".")[-1]:
                kw["foreground"]=FG; kw["font"]=("Consolas",10,"bold")
            if n=="D.TLabelframe.Label":
                kw["foreground"]=ACCENT; kw["font"]=("Consolas",10,"bold")
            s.configure(n,**kw)

    def _build(self):
        top=ttk.Frame(self.root,style="D.TFrame"); top.pack(fill=tk.X,padx=10,pady=(3,1))
        ttk.Label(top,text="MARS — Two-Hand Control",style="T.TLabel").pack(side=tk.LEFT)
        self.fps_lbl=ttk.Label(top,text="",style="D.TLabel"); self.fps_lbl.pack(side=tk.RIGHT)

        body=ttk.Frame(self.root,style="D.TFrame"); body.pack(fill=tk.BOTH,expand=True,padx=10,pady=2)
        body.columnconfigure(0,weight=3); body.columnconfigure(1,weight=4); body.columnconfigure(2,weight=2)
        body.rowconfigure(0,weight=1)

        # LEFT — camera
        lf=ttk.LabelFrame(body,text=" Camera (L-hand=green R-hand=orange) ",style="D.TLabelframe")
        lf.grid(row=0,column=0,sticky="nsew",padx=(0,3))
        self.cam_lbl=tk.Label(lf,bg="#0d1117"); self.cam_lbl.pack(fill=tk.BOTH,expand=True,padx=3,pady=3)
        ir=ttk.Frame(lf,style="D.TFrame"); ir.pack(fill=tk.X,padx=3,pady=(0,3))
        self.info_lbl=ttk.Label(ir,text="Waiting...",style="D.TLabel"); self.info_lbl.pack(side=tk.LEFT)
        self.grip_lbl=ttk.Label(ir,text="",style="D.TLabel"); self.grip_lbl.pack(side=tk.RIGHT)

        # CENTRE — 3D
        cf=ttk.LabelFrame(body,text=" 3D View (drag to rotate) ",style="D.TLabelframe")
        cf.grid(row=0,column=1,sticky="nsew",padx=3)
        self.fig=Figure(figsize=(5,4),dpi=100,facecolor=BG)
        self.canvas3d=FigureCanvasTkAgg(self.fig,master=cf)
        self.canvas3d.get_tk_widget().pack(fill=tk.BOTH,expand=True,padx=2,pady=2)
        self.ee_lbl=ttk.Label(cf,text="EE: --",style="S.TLabel")
        self.ee_lbl.pack(fill=tk.X,padx=3,pady=(0,2))

        # RIGHT — controls (scrollable-ish via packing order)
        rf=ttk.Frame(body,style="D.TFrame"); rf.grid(row=0,column=2,sticky="nsew",padx=(3,0))
        def btn(p,t,bg_c,cmd,**kw):
            b=tk.Button(p,text=t,font=("Consolas",9,"bold"),bg=bg_c,fg="white",
                        activebackground="#444",relief=tk.FLAT,padx=5,pady=2,command=cmd,**kw)
            b.pack(fill=tk.X,padx=4,pady=1); return b

        bf=ttk.LabelFrame(rf,text=" Controls ",style="D.TLabelframe"); bf.pack(fill=tk.X,pady=(0,2))
        self.btn_cam=btn(bf,"Start Camera","#1b4332",self._toggle_cam)
        self.btn_track=btn(bf,"Enable Tracking (M)","#1a3a5c",self._toggle_track,state=tk.DISABLED)
        self.btn_cal=btn(bf,"Calibrate (Space)","#5c3a1a",self._calibrate,state=tk.DISABLED)
        self.btn_home=btn(bf,"Home (H)","#333",self._go_home)
        self.btn_view=btn(bf,"View: AUTO (V)","#2d3a2d",self._toggle_auto_view)

        # J1 home angle
        j1f=ttk.LabelFrame(rf,text=" J1 Home ",style="D.TLabelframe"); j1f.pack(fill=tk.X,pady=2)
        j1r=ttk.Frame(j1f,style="D.TFrame"); j1r.pack(fill=tk.X,padx=4,pady=2)
        self.j1_entry=tk.Entry(j1r,width=5,bg=PANEL,fg=FG,font=("Consolas",9),
                               insertbackground=ACCENT,relief=tk.FLAT)
        self.j1_entry.insert(0,"0"); self.j1_entry.pack(side=tk.LEFT,padx=2)
        tk.Button(j1r,text="Set",font=("Consolas",8,"bold"),bg="#2d4a3e",fg="white",
                  relief=tk.FLAT,padx=4,command=self._set_j1_home).pack(side=tk.LEFT)

        # Platform
        pf=ttk.LabelFrame(rf,text=" Platform ",style="D.TLabelframe"); pf.pack(fill=tk.X,pady=2)
        pr=ttk.Frame(pf,style="D.TFrame"); pr.pack(fill=tk.X,padx=4,pady=2)
        self.plat_lbl=ttk.Label(pr,text=f"{PLATFORM_H:.0f}mm",width=6,style="D.TLabel",font=("Consolas",8))
        self.plat_lbl.pack(side=tk.RIGHT)
        self.plat_slider=tk.Scale(pr,from_=PLATFORM_H_MIN,to=PLATFORM_H_MAX,orient=tk.HORIZONTAL,
            bg=BG,fg=FG,troughcolor=PANEL,highlightbackground=BG,activebackground="#4a7c59",
            length=90,showvalue=False,resolution=5,command=self._on_plat)
        self.plat_slider.set(PLATFORM_H); self.plat_slider.pack(side=tk.LEFT,fill=tk.X,expand=True,padx=2)

        # Servo sliders
        sf=ttk.LabelFrame(rf,text=" Servos ",style="D.TLabelframe"); sf.pack(fill=tk.X,pady=2)
        self.sliders={}; self.slbl={}
        jl={"J1":"J1 Base","J2":"J2 Shldr","J3":"J3 Elbow","J4":"J4 Wrist","J5":"J5 Rot","J6":"J6 Claw"}
        for jn in self.arm.JOINTS:
            lo,hi,hm=SERVO[jn]
            r=ttk.Frame(sf,style="D.TFrame"); r.pack(fill=tk.X,padx=3,pady=0)
            ttk.Label(r,text=f"{jl[jn]}:",width=8,style="D.TLabel",font=("Consolas",8)).pack(side=tk.LEFT)
            vl=ttk.Label(r,text=f"{hm:.0f}",width=4,style="D.TLabel",font=("Consolas",8))
            vl.pack(side=tk.RIGHT); self.slbl[jn]=vl
            sc=tk.Scale(r,from_=lo,to=hi,orient=tk.HORIZONTAL,bg=BG,fg=FG,troughcolor=PANEL,
                        highlightbackground=BG,activebackground=ACCENT,length=80,showvalue=False,
                        command=lambda v,j=jn:self._slider(j,float(v)))
            sc.set(hm); sc.pack(side=tk.LEFT,fill=tk.X,expand=True,padx=1); self.sliders[jn]=sc

        # Macro buttons
        mf=ttk.LabelFrame(rf,text=" Pickup Macros ",style="D.TLabelframe"); mf.pack(fill=tk.X,pady=2)
        mr=ttk.Frame(mf,style="D.TFrame"); mr.pack(fill=tk.X,padx=4,pady=2)
        for i,(col,name) in enumerate([("#ef4444","Red"),("#3b82f6","Blue"),
                                        ("#22c55e","Green"),("#eab308","Yellow")]):
            b=tk.Button(mr,text=name,font=("Consolas",8,"bold"),bg=col,fg="white",
                        relief=tk.FLAT,padx=4,pady=1,
                        command=lambda n=name:self._run_macro(n))
            b.grid(row=i//2,column=i%2,sticky="ew",padx=1,pady=1)
        mr.columnconfigure(0,weight=1); mr.columnconfigure(1,weight=1)
        btn(mf,"Spawn Block (B)","#3a2d3a",self._spawn_block)
        btn(mf,"Clear Blocks","#3a2d2d",self._clear_blocks)

        # Log
        logf=ttk.LabelFrame(rf,text=" Log ",style="D.TLabelframe"); logf.pack(fill=tk.BOTH,expand=True,pady=(2,0))
        self.log_w=tk.Text(logf,bg="#0d1117",fg="#58a6ff",font=("Consolas",7),
                           height=5,width=22,wrap=tk.WORD,state=tk.DISABLED,relief=tk.FLAT)
        self.log_w.pack(fill=tk.BOTH,expand=True,padx=3,pady=3)

    # ══════════════════════════════════════════════════════════════════
    # 3D
    # ══════════════════════════════════════════════════════════════════
    def _init_3d(self):
        self.ax=self.fig.add_subplot(111,projection="3d"); self.ax.set_facecolor("#0d1117")
        self.fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
        self.canvas3d.mpl_connect('button_press_event',lambda e:setattr(self,'_drag_active',True))
        self.canvas3d.mpl_connect('button_release_event',self._on_3d_release)

    def _on_3d_release(self,e):
        self._drag_active=False
        if self.ax: self._view_elev=self.ax.elev; self._view_azim=self.ax.azim

    def _toggle_auto_view(self):
        self._auto_track=not self._auto_track
        self.btn_view.config(text=f"View: {'AUTO' if self._auto_track else 'MANUAL'} (V)",
                             bg="#2d3a2d" if self._auto_track else "#3a2d2d")

    def _compute_best_view(self,pts):
        pp=np.array([[p[0],p[2],p[1]] for p in pts])
        rg=np.array([pp[-1][0]-pp[0][0],pp[-1][1]-pp[0][1]]); rl=np.linalg.norm(rg)
        azim=(math.degrees(math.atan2(rg[1],rg[0]))-90) if rl>1e-3 else self._view_azim
        hs=[p[2] for p in pp]; hr=max(hs)-min(hs); v=hr/(hr+max(rl,1.))
        elev=15+v*35
        cn=pp-np.mean(pp,axis=0)
        try:
            _,S,_=np.linalg.svd(cn,full_matrices=False)
            if len(S)>1 and S[1]/(S[0]+1e-9)<.15: elev=max(elev,35)
        except: pass
        return max(8,min(65,elev)),azim

    def _draw_arm(self):
        try:
            ax=self.ax; ax.clear(); ax.set_facecolor("#0d1117")
            pts=self.arm.forward_kinematics(); ph=self.arm.platform_h
            def p(v): return(v[0],v[2],v[1])

            # Platform
            if ph>.5:
                hw=PLATFORM_W/2.; hd=PLATFORM_D/2.
                c=[np.array([dx,dy,dz]) for dx in[-hw,hw] for dy in[-hd,hd] for dz in[0,ph]]
                pf=[[c[0],c[1],c[3],c[2]],[c[4],c[5],c[7],c[6]],
                    [c[0],c[1],c[5],c[4]],[c[2],c[3],c[7],c[6]],
                    [c[0],c[2],c[6],c[4]],[c[1],c[3],c[7],c[5]]]
                ax.add_collection3d(Poly3DCollection(pf,alpha=.5,facecolor=(.12,.16,.22),
                    edgecolor=(.2,.25,.34),linewidths=.7))

            # Base disc
            th=np.linspace(0,2*np.pi,30); bx,by=40*np.cos(th),40*np.sin(th)
            bz=np.full(30,ph); ax.plot(bx,by,bz,color="#475569",lw=1.5)
            ax.add_collection3d(Poly3DCollection([list(zip(bx,by,bz))],alpha=.3,
                facecolor="#1e3a5f",edgecolor="#475569"))

            # Link boxes
            for lk in self.arm.link_boxes():
                fs=_box_faces(lk["start"],lk["end"],lk["width"])
                if fs:
                    rgb=_hex_rgb(lk["color"])
                    ax.add_collection3d(Poly3DCollection([[p(v) for v in f] for f in fs],
                        alpha=lk["alpha"],facecolor=rgb,edgecolor=(*rgb,.6),linewidths=.4))

            # Skeleton + joints
            xs=[pt[0] for pt in pts]; ys=[pt[2] for pt in pts]; zs=[pt[1] for pt in pts]
            ax.plot(xs,ys,zs,color="white",lw=.8,alpha=.3,ls="--")
            jc=["#475569","#0ea5e9","#0ea5e9","#6366f1","#a78bfa","#ec4899","#f472b6"]
            for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
                ax.scatter(x,y,z,color=jc[min(i,6)],s=40 if i<2 else 55-i*5,
                           edgecolors="white",linewidths=.2,zorder=5,depthshade=True)

            # Finger boxes
            for fb in self.arm.finger_boxes():
                fs=_box_faces(fb["start"],fb["end"],fb["width"])
                if fs:
                    rgb=_hex_rgb(fb["color"])
                    ax.add_collection3d(Poly3DCollection([[p(v) for v in f] for f in fs],
                        alpha=fb["alpha"],facecolor=rgb,edgecolor=(*rgb,.7),linewidths=.5))

            # Finger skeleton + grip marker
            fg=self.arm.get_finger_geometry()
            for tk_ in ["left_tip","right_tip"]:
                t=fg[tk_]; b=fg["claw_base"]
                ax.plot([p(b)[0],p(t)[0]],[p(b)[1],p(t)[1]],[p(b)[2],p(t)[2]],
                        color="#f472b6",lw=.8,alpha=.4,ls="--")
            gc=fg["grip_center"]; gc_col="#51cf66" if fg["is_closed"] else "#ff6b6b"
            ax.scatter(*p(gc),color=gc_col,s=25,marker="o",zorder=7,depthshade=False,alpha=.6)

            # Joint axes
            for ja in self.arm.joint_axes():
                av=ja["axis"]*ja["length"]; s=p(ja["pos"]-av*.5); e=p(ja["pos"]+av*.5)
                ax.plot([s[0],e[0]],[s[1],e[1]],[s[2],e[2]],color=ja["color"],lw=1.2,alpha=.5)

            # Blocks
            for bs in self.physics.get_block_states():
                brgb=_hex_rgb(bs["color"]); ba=.7 if bs["grabbed"] else .5
                ax.add_collection3d(Poly3DCollection([[p(v) for v in f] for f in bs["faces"]],
                    alpha=ba,facecolor=brgb,edgecolor=(*brgb,.4) if not bs["grabbed"] else (1,1,1,.3),
                    linewidths=.4))
                bp=bs["pos"]
                ax.text(p(bp)[0],p(bp)[1],p(bp)[2]+bs["size"]+6,bs["label"],
                        color=bs["color"],fontsize=4,ha="center",alpha=.7)

            # Ground grid
            for g in np.linspace(-200,200,9):
                ax.plot([g,g],[-200,200],[0,0],color="#1e293b",lw=.15,alpha=.3)
                ax.plot([-200,200],[g,g],[0,0],color="#1e293b",lw=.15,alpha=.3)

            # Workspace arc
            reach=J2_J3+J3_J4+J4_J5; tw=np.linspace(-np.pi/2,np.pi/2,50)
            ax.plot(reach*np.cos(tw),reach*np.sin(tw),np.full(50,ph+J1_J2),
                    color="#334155",lw=.5,ls=":",alpha=.2)

            L=280; zt=max(L+80,ph+L)
            ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_zlim(-20,zt)
            ax.tick_params(colors="#475569",labelsize=4)
            ax.set_xlabel("X",color="#475569",fontsize=5)
            ax.set_ylabel("Z",color="#475569",fontsize=5)
            ax.set_zlabel("Y",color="#475569",fontsize=5)

            if self._auto_track and not self._drag_active:
                te,ta=self._compute_best_view(pts); bl=.12
                da=ta-self._view_azim
                if da>180: da-=360
                if da<-180: da+=360
                self._view_azim+=da*bl; self._view_elev+=(te-self._view_elev)*bl
            ax.view_init(elev=self._view_elev,azim=self._view_azim)
            ax.set_box_aspect([1,1,1])
            for pn in(ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane):
                pn.fill=False; pn.set_edgecolor("#1e293b")
            ax.grid(True,alpha=.08,color="#334155")

            ee=pts[-1]
            self.ee_lbl.config(text=f"EE X={ee[0]:.0f} Y={ee[1]:.0f} Z={ee[2]:.0f}")
            self.canvas3d.draw_idle()
        except Exception as e:
            print(f"[3D] {e}",file=sys.stderr)

    # ══════════════════════════════════════════════════════════════════
    # CAMERA LOOP — throttled 3D for smoothness
    # ══════════════════════════════════════════════════════════════════
    def _toggle_cam(self):
        if self.running: self._stop_cam()
        else: self._start_cam()

    def _start_cam(self):
        if not self.tracker: self._log("No tracker"); return
        try: self.cap=cv2.VideoCapture(CAM_INDEX)
        except Exception as e: self._log(f"Err: {e}"); return
        if not self.cap or not self.cap.isOpened(): self._log("No camera"); return
        ok,f=self.cap.read()
        if not ok: self._log("Read failed"); self.cap.release(); self.cap=None; return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS,30)
        fov=detect_camera_fov(self.cap); self.tracker.set_fov(fov)
        self._log(f"Camera FOV:{fov:.0f}")
        self.running=True; self._frame_count=0
        self.btn_cam.config(text="Stop Camera",bg="#5c1a1a")
        self.btn_track.config(state=tk.NORMAL); self.btn_cal.config(state=tk.NORMAL)
        self._log("Camera started — two-hand mode")
        self._loop()

    def _stop_cam(self):
        self.running=False; self.tracking_on=False
        self.btn_cam.config(text="Start Camera",bg="#1b4332")
        self.btn_track.config(text="Enable Tracking (M)",bg="#1a3a5c",state=tk.DISABLED)
        self.btn_cal.config(state=tk.DISABLED)
        for s in self.sliders.values(): s.config(state=tk.NORMAL)
        if self.cap:
            try: self.cap.release()
            except: pass
            self.cap=None

    def _loop(self):
        if not self.running or not self.cap: return
        try:
            t0=time.time()
            ok,frame=self.cap.read()
            if not ok or frame is None: self.root.after(50,self._loop); return
            frame=cv2.flip(frame,1)
            result=self.tracker.process(frame)
            self._last=result

            if result["detected"]:
                a=result["angles"]
                la="L:OK" if result.get("left_hand_ok") else "L:--"
                ra="R:OK" if result.get("right_hand_ok") else "R:--"
                ht=""  # hand status already in la/ra
                self.info_lbl.config(text=f"{la} {ra}")
                g=a.get("J6",73)
                self.grip_lbl.config(text=f"Claw:{'OPEN' if g>(SERVO['J6'][0]+SERVO['J6'][1])/2 else 'CLOSED'}")
                if self.tracking_on and not self.macro.active:
                    self.arm.apply_angles_smooth(a)
                    self._sync_sliders()
            else:
                self.info_lbl.config(text="No hands"); self.grip_lbl.config(text="")

            # Physics + macro
            now=time.time(); dt=min(now-self._last_time,.1); self._last_time=now
            if self.macro.active:
                self.macro.update(dt); self._sync_sliders()
            fg=self.arm.get_finger_geometry()
            fg["grip_velocity"]=self.arm._grip_velocity.copy()
            self.physics.step(dt,fg)

            # Show camera every frame, 3D only every Nth
            self._show(result["frame"])
            self._frame_count+=1
            if self._frame_count%self._3D_THROTTLE==0:
                self._draw_arm()

            dt2=time.time()-t0
            self._ftimes.append(dt2)
            if len(self._ftimes)>30: self._ftimes=self._ftimes[-30:]
            self.fps_lbl.config(text=f"FPS:{1./(np.mean(self._ftimes)+1e-9):.0f}")
            delay=max(1,int((1/30-dt2)*1000))
            self.root.after(delay,self._loop)
        except Exception as e:
            print(f"[loop] {e}\n{traceback.format_exc()}",file=sys.stderr)
            self.root.after(500,self._loop)

    def _show(self,bgr):
        try:
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            lw,lh=self.cam_lbl.winfo_width(),self.cam_lbl.winfo_height()
            if lw>30 and lh>30: rgb=cv2.resize(rgb,(lw,lh))
            im=ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.cam_lbl.imgtk=im; self.cam_lbl.config(image=im)
        except: pass

    def _physics_tick(self):
        if not self.running:
            now=time.time(); dt=min(now-self._last_time,.1); self._last_time=now
            if self.macro.active: self.macro.update(dt); self._sync_sliders()
            fg=self.arm.get_finger_geometry(); fg["grip_velocity"]=self.arm._grip_velocity.copy()
            self.physics.step(dt,fg)
            if any(not b.resting or b.grabbed for b in self.physics.blocks) or self.macro.active:
                self._draw_arm()
        self.root.after(33,self._physics_tick)

    # ══════════════════════════════════════════════════════════════════
    # ACTIONS
    # ══════════════════════════════════════════════════════════════════
    def _toggle_track(self):
        self.tracking_on=not self.tracking_on
        if self.tracking_on:
            self.btn_track.config(text="Disable Tracking (M)",bg="#264b73")
            for s in self.sliders.values(): s.config(state=tk.DISABLED)
            self._log("Tracking ON (L-hand→J1-3, R-hand→J4-6)")
        else:
            self.btn_track.config(text="Enable Tracking (M)",bg="#1a3a5c")
            for s in self.sliders.values(): s.config(state=tk.NORMAL)
            self._log("Tracking OFF")

    def _calibrate(self):
        if not self.tracker: return
        if self._last and self._last["detected"]:
            left_lms = self._last.get("left_hand_lms")
            self.tracker.calibrate(self._last["raw_angles"], left_lms)
            self.arm.go_home(); self._sync_sliders(); self._draw_arm()
            self._log("Calibrated — both hands")
        else: self._log("No hands detected")

    def _set_j1_home(self):
        try:
            v=float(self.j1_entry.get().strip())
            v=max(SERVO["J1"][0],min(SERVO["J1"][1],v))
            if self.tracker: self.tracker.set_j1_home(v)
            self._log(f"J1 home={v:.0f}")
        except: self._log("Invalid J1 value")

    def _go_home(self):
        self.macro.cancel(); self.arm.go_home(); self._sync_sliders(); self._draw_arm()

    def _slider(self,j,v):
        if self.tracking_on: return
        self.arm.set_angle(j,v); self.arm.enforce_ground_constraint()
        self.slbl[j].config(text=f"{self.arm.angles[j]:.0f}")
        fg=self.arm.get_finger_geometry(); fg["grip_velocity"]=self.arm._grip_velocity.copy()
        self.physics.step(.001,fg); self._sync_sliders(); self._draw_arm()

    def _on_plat(self,v):
        self.arm.platform_h=float(v); self.plat_lbl.config(text=f"{float(v):.0f}mm"); self._draw_arm()

    def _sync_sliders(self):
        for j,sc in self.sliders.items():
            a=self.arm.angles[j]; sc.set(a); self.slbl[j].config(text=f"{a:.0f}")

    def _spawn_block(self):
        b=self.physics.spawn_block(); self._log(f"Spawned {b.label}"); self._draw_arm()
    def _clear_blocks(self):
        self.physics.clear(); self._log("Cleared"); self._draw_arm()

    def _run_macro(self,color):
        if self.macro.active: self.macro.cancel()
        self.tracking_on=False
        self.btn_track.config(text="Enable Tracking (M)",bg="#1a3a5c")
        for s in self.sliders.values(): s.config(state=tk.DISABLED)
        msg=self.macro.start_pickup(color); self._log(msg)

    def _log(self,msg):
        line=f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line)
        try:
            self.log_w.config(state=tk.NORMAL); self.log_w.insert(tk.END,line+"\n")
            self.log_w.see(tk.END); self.log_w.config(state=tk.DISABLED)
        except: pass

    def _on_err(self,*a):
        print(f"[TK] {''.join(traceback.format_exception(*a))}",file=sys.stderr)

    def close(self):
        self.running=False
        if self.cap:
            try: self.cap.release()
            except: pass
        if self.tracker:
            try: self.tracker.release()
            except: pass
        self.root.destroy()

def main():
    root=tk.Tk(); app=App(root)
    root.protocol("WM_DELETE_WINDOW",app.close)
    root.bind("<space>",lambda e:app._calibrate())
    root.bind("<h>",lambda e:app._go_home())
    root.bind("<m>",lambda e:app._toggle_track())
    root.bind("<v>",lambda e:app._toggle_auto_view())
    root.bind("<b>",lambda e:app._spawn_block())
    root.bind("<Return>",lambda e:app._set_j1_home())
    root.bind("<Escape>",lambda e:app.close())
    # Number keys for macro shortcuts
    root.bind("1",lambda e:app._run_macro("Red"))
    root.bind("2",lambda e:app._run_macro("Blue"))
    root.bind("3",lambda e:app._run_macro("Green"))
    root.bind("4",lambda e:app._run_macro("Yellow"))
    root.mainloop()

if __name__=="__main__": main()
