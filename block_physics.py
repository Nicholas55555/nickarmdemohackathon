"""
Block physics: gravity, grab/throw, color lookup for macros.
"""
import numpy as np

GRAVITY=-4000.; GROUND_Y=0.; BOUNCE=.3; FRICTION=.92
GRAB_RADIUS=45.; BLOCK_SIZE=25.

COLORS=[("#ef4444","Red"),("#3b82f6","Blue"),("#22c55e","Green"),
        ("#eab308","Yellow"),("#a855f7","Purple"),("#f97316","Orange")]

class Block:
    _id=0
    def __init__(self,pos,color="#ef4444",label=""):
        Block._id+=1; self.id=Block._id
        self.pos=pos.astype(float); self.vel=np.zeros(3)
        self.color=color; self.label=label or f"B{self.id}"
        self.size=BLOCK_SIZE; self.grabbed=False; self.resting=False

    def faces_3d(self):
        s=self.size; p=self.pos
        c=[p+np.array([dx,dy,dz]) for dx in[-s,s] for dy in[-s,s] for dz in[-s,s]]
        return [[c[0],c[1],c[3],c[2]],[c[4],c[5],c[7],c[6]],
                [c[0],c[1],c[5],c[4]],[c[2],c[3],c[7],c[6]],
                [c[0],c[2],c[6],c[4]],[c[1],c[3],c[7],c[5]]]

class BlockPhysics:
    def __init__(self):
        self.blocks: list[Block]=[]; self._ci=0

    def spawn_block(self,pos=None):
        if pos is None:
            a=np.random.uniform(-1,1); d=np.random.uniform(80,180)
            pos=np.array([d*np.cos(a),BLOCK_SIZE,d*np.sin(a)])
        c,l=COLORS[self._ci%len(COLORS)]; self._ci+=1
        b=Block(pos,c,l); self.blocks.append(b); return b

    def spawn_default_set(self):
        for p in [np.array([120.,BLOCK_SIZE,0.]),np.array([100.,BLOCK_SIZE,80.]),
                  np.array([60.,BLOCK_SIZE,-60.]),np.array([150.,BLOCK_SIZE,40.])]:
            self.spawn_block(p)

    def clear(self): self.blocks.clear(); self._ci=0

    def find_by_color(self, color_name: str):
        """Find first ungrabbed block matching color name (case-insensitive)."""
        cn=color_name.strip().lower()
        for b in self.blocks:
            if not b.grabbed and b.label.lower()==cn:
                return b
        # Try partial match
        for b in self.blocks:
            if not b.grabbed and cn in b.label.lower():
                return b
        return None

    def get_color_list(self):
        """Return list of (label, color_hex) for ungrabbed blocks."""
        return [(b.label, b.color) for b in self.blocks if not b.grabbed]

    def step(self,dt,grip_info):
        gc=grip_info["grip_center"]; go=grip_info["grip_opening"]
        ic=grip_info["is_closed"]; gv=grip_info.get("grip_velocity",np.zeros(3))
        for b in self.blocks:
            if b.grabbed:
                if not ic:
                    b.grabbed=False; b.vel=gv*(1./max(dt,1e-3))*.6; b.resting=False
                else:
                    b.pos=gc.copy(); b.vel=np.zeros(3)
                continue
            if ic and not b.grabbed:
                if np.linalg.norm(b.pos-gc)<GRAB_RADIUS:
                    b.grabbed=True; b.vel=np.zeros(3); b.resting=False; continue
            if b.resting: continue
            b.vel[1]+=GRAVITY*dt; b.pos+=b.vel*dt
            if b.pos[1]-b.size<GROUND_Y:
                b.pos[1]=GROUND_Y+b.size
                if abs(b.vel[1])<30: b.vel=np.zeros(3); b.resting=True
                else: b.vel[1]*=-BOUNCE; b.vel[0]*=FRICTION; b.vel[2]*=FRICTION
            b.vel*=.998

    def get_block_states(self):
        return [dict(pos=b.pos.copy(),color=b.color,label=b.label,
                     size=b.size,grabbed=b.grabbed,faces=b.faces_3d()) for b in self.blocks]
