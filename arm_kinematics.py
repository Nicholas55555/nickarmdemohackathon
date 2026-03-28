"""
FK, animated fingers, ground constraint, and simple IK for macros.
"""
import numpy as np
from config import (
    J1_J2, J2_J3, J3_J4, J4_J5, J5_J6, CLAW,
    SERVO, HOME, BLEND_ALPHA, LINK_W, PLATFORM_H,
)

def _clamp(v, lo, hi): return max(lo, min(hi, v))

class ArmKinematics:
    JOINTS = ["J1","J2","J3","J4","J5","J6"]

    def __init__(self):
        self.angles = dict(HOME)
        self.platform_h = PLATFORM_H
        self._prev_grip_center = None
        self._grip_velocity = np.zeros(3)

    def set_angle(self, j, d):
        lo,hi,_ = SERVO[j]; self.angles[j] = _clamp(d,lo,hi)

    def go_home(self): self.angles = dict(HOME)

    # ── FK ─────────────────────────────────────────────────────────────
    def forward_kinematics(self):
        yaw=np.radians(self.angles["J1"]); sp=np.radians(self.angles["J2"])
        ep=np.radians(self.angles["J3"]); wp=np.radians(self.angles["J4"])
        ph=self.platform_h
        def off(l,a):
            dx=l*np.sin(a); dy=l*np.cos(a)
            return np.array([dx*np.cos(yaw),dy,dx*np.sin(yaw)])
        pts=[np.array([0.,ph,0.])]
        p1=np.array([0.,ph+J1_J2,0.]); pts.append(p1)
        cum=sp; p2=p1+off(J2_J3,cum); pts.append(p2)
        cum+=np.pi-ep; p3=p2+off(J3_J4,cum); pts.append(p3)
        cum+=wp; p4=p3+off(J4_J5,cum); pts.append(p4)
        p5=p4+off(J5_J6,cum); pts.append(p5)
        p6=p5+off(CLAW,cum); pts.append(p6)
        return pts

    def get_end_effector(self): return self.forward_kinematics()[-1]

    # ── Finger geometry ────────────────────────────────────────────────
    def get_finger_geometry(self):
        yaw=np.radians(self.angles["J1"]); sp=np.radians(self.angles["J2"])
        ep=np.radians(self.angles["J3"]); wp=np.radians(self.angles["J4"])
        roll=np.radians(self.angles["J5"]); ph=self.platform_h
        def off(l,a):
            dx=l*np.sin(a); dy=l*np.cos(a)
            return np.array([dx*np.cos(yaw),dy,dx*np.sin(yaw)])
        p1=np.array([0.,ph+J1_J2,0.]); cum=sp
        p2=p1+off(J2_J3,cum); cum+=np.pi-ep
        p3=p2+off(J3_J4,cum); cum+=wp
        p4=p3+off(J4_J5,cum); j6p=p4+off(J5_J6,cum)
        j6_lo,j6_hi,_=SERVO["J6"]
        t=(self.angles["J6"]-j6_lo)/(j6_hi-j6_lo)
        sr=np.radians(2+t*28)
        lt=j6p+off(CLAW,cum+sr); rt=j6p+off(CLAW,cum-sr)
        gc=(lt+rt)/2; go=np.linalg.norm(lt-rt)
        gd=off(1,cum); gd=gd/(np.linalg.norm(gd)+1e-9)
        if self._prev_grip_center is not None:
            self._grip_velocity=gc-self._prev_grip_center
        self._prev_grip_center=gc.copy()
        return dict(claw_base=j6p, left_base=j6p.copy(), left_tip=lt,
                    right_base=j6p.copy(), right_tip=rt, grip_center=gc,
                    grip_opening=go, grip_dir=gd, is_closed=go<18,
                    perp=np.array([-np.sin(yaw)*np.cos(roll),np.sin(roll),
                                    np.cos(yaw)*np.cos(roll)]))

    def link_boxes(self):
        pts=self.forward_kinematics()
        W=[LINK_W["base"],LINK_W["J2_J3"],LINK_W["J3_J4"],LINK_W["J4_J5"],LINK_W["J5_J6"]]
        C=["#334155","#0284c7","#0ea5e9","#6366f1","#8b5cf6"]
        A=[0.4,0.5,0.5,0.45,0.4]
        return [dict(start=pts[i],end=pts[i+1],width=W[i],color=C[i],alpha=A[i])
                for i in range(min(len(pts)-2,5))]

    def finger_boxes(self):
        fg=self.get_finger_geometry(); fw=LINK_W["claw"]
        return [dict(start=fg["left_base"],end=fg["left_tip"],width=fw,color="#ec4899",alpha=.45),
                dict(start=fg["right_base"],end=fg["right_tip"],width=fw,color="#ec4899",alpha=.45)]

    def joint_axes(self):
        pts=self.forward_kinematics(); yaw=np.radians(self.angles["J1"])
        perp=np.array([-np.sin(yaw),0,np.cos(yaw)])
        ax=[dict(pos=pts[0],axis=np.array([0,1,0]),label="J1",color="#facc15",length=35),
            dict(pos=pts[1],axis=perp,label="J2",color="#22d3ee",length=30),
            dict(pos=pts[2],axis=perp,label="J3",color="#22d3ee",length=25),
            dict(pos=pts[3],axis=perp,label="J4",color="#a78bfa",length=20)]
        if len(pts)>4:
            fd=pts[4]-pts[3]; fn=np.linalg.norm(fd)
            ax.append(dict(pos=pts[4],axis=fd/fn if fn>1e-6 else np.array([1,0,0]),
                           label="J5",color="#fb923c",length=20))
        ax.append(dict(pos=pts[5],axis=perp,label="J6",color="#f472b6",length=15))
        return ax

    # ── Smooth apply + ground constraint ──────────────────────────────
    def apply_angles_smooth(self, target, alpha=BLEND_ALPHA):
        for j in self.JOINTS:
            if j in target:
                self.set_angle(j, self.angles[j]*(1-alpha)+target[j]*alpha)
        self.enforce_ground_constraint()

    def enforce_ground_constraint(self, ground_y=0.0, max_iter=30):
        for _ in range(max_iter):
            pts=self.forward_kinematics(); fg=self.get_finger_geometry()
            ap=list(pts)+[fg["left_tip"],fg["right_tip"]]
            ys=[p[1] for p in ap]; my=min(ys)
            if my>=ground_y-0.5: return
            mi=int(np.argmin(ys)); viol=ground_y-my
            if mi>=4: tj=["J4","J3","J2"]
            elif mi>=3: tj=["J3","J2"]
            elif mi>=2: tj=["J2"]
            else: return
            for j in tj:
                ov=self.angles[j]; lo,hi,_=SERVO[j]
                bd=0; bm=my
                for d in [-1.,1.]:
                    tv=_clamp(ov+d*2,lo,hi)
                    if abs(tv-ov)<.1: continue
                    self.angles[j]=tv
                    tp=self.forward_kinematics(); tfg=self.get_finger_geometry()
                    tm=min(p[1] for p in list(tp)+[tfg["left_tip"],tfg["right_tip"]])
                    if tm>bm: bm=tm; bd=d
                    self.angles[j]=ov
                if bd!=0:
                    self.angles[j]=_clamp(ov+bd*max(2,min(viol*.8,25)),lo,hi)
                    break

    # ── IK for macro pickup ───────────────────────────────────────────
    def solve_angles_for_position(self, target_pos):
        """
        Compute J1,J2,J3,J4 angles to reach target_pos (mm) with gripper
        pointing downward. Returns dict of angles or None if unreachable.
        """
        x,y,z = target_pos
        ph = self.platform_h
        # J1: base rotation
        r = np.sqrt(x**2+z**2)
        j1 = np.degrees(np.arctan2(z,x)) if r>1 else 0
        # Work in vertical plane
        L1,L2 = J2_J3, J3_J4
        EE = J4_J5+J5_J6+CLAW
        # Wrist target (gripper down: wrist is above target by EE length)
        wr = r; wy = y+EE-ph-J1_J2
        dist = np.sqrt(wr**2+wy**2)
        if dist>L1+L2-1: return None  # unreachable
        if dist<abs(L1-L2)+1: return None
        # 2-link IK
        ce=(L1**2+L2**2-dist**2)/(2*L1*L2)
        ce=_clamp(ce,-1,1); er=np.arccos(ce)
        al=np.arctan2(wy,wr)
        cb=(L1**2+dist**2-L2**2)/(2*L1*dist)
        cb=_clamp(cb,-1,1); be=np.arccos(cb)
        sr=np.pi/2-(al+be)
        # Wrist pitch: make total cum angle = π so gripper points straight down
        # cum = sr + (π - er) + wp = π  →  wp = er - sr
        wp=er-sr
        return {"J1":_clamp(j1,*SERVO["J1"][:2]),
                "J2":_clamp(np.degrees(sr),*SERVO["J2"][:2]),
                "J3":_clamp(np.degrees(er),*SERVO["J3"][:2]),
                "J4":_clamp(np.degrees(wp),*SERVO["J4"][:2]),
                "J5":0, "J6":self.angles["J6"]}
