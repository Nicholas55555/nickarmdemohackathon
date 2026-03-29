
# MARS — Robotic Arm Simulation

Keyboard-controlled 6DOF robot arm simulator with real-time block physics,
IK-driven macros, and four mathematically distinct auto-camera algorithms
based on eigendecomposition of scatter matrices.

Built for EagleHacks 2026. Simulates a LewanSoul LeArm with measured
dimensions and servo limits.

## Setup

```bash
pip install -r requirements.txt   # numpy, Pillow, matplotlib
python main.py
````

Press **F1** or click **? Help** for the full in-app guide.

## Quick Reference

```
A/D → J1 base    R/F → J4 wrist    H = Home     V = Cycle camera
W/S → J2 shoulder T/G → J5 roll    B = Spawn    Shift+L = Lock view
Q/E → J3 elbow   Z/X → J6 claw    1-4 = Macro  F1 = Help
```

---

## Mathematical Foundations

### 1. Forward Kinematics (FK)

The arm is a 6DOF serial chain. Each joint either rotates about the vertical
axis (J1) or pitches in the arm's swing plane (J2–J4). The FK chain computes
all joint positions from the base to the claw tip.

**Coordinate system:** X = right, Y = up, Z = forward. The arm base sits at
the origin, raised by `platform_h` mm.

**Link offset function.** For a link of length `L` at cumulative pitch angle
`θ_cum`, the offset in the arm's swing plane is:

```
dx = L · sin(θ_cum)
dy = L · cos(θ_cum)
```

This 2D offset is then projected into 3D using the J1 yaw angle `ψ`:

```
offset = ( dx · cos(ψ),  dy,  dx · sin(ψ) )
```

**Chain computation.** Starting from the base at `(0, platform_h + J1_J2, 0)`:

```
P₁ = base
θ_cum = J2                          P₂ = P₁ + off(J2_J3, θ_cum)
θ_cum += π − J3                     P₃ = P₂ + off(J3_J4, θ_cum)
θ_cum += J4                         P₄ = P₃ + off(J4_J5, θ_cum)
                                     P₅ = P₄ + off(J5_J6, θ_cum)
                                     P₆ = P₅ + off(CLAW, θ_cum)
```

The `π − J3` term accounts for the elbow joint's opposing direction: as J3
increases from 0° to 180°, the forearm folds from straight to fully doubled back.

**Measured link lengths (inches → mm):**

| Link  | Inches | mm    | Description      |
| ----- | ------ | ----- | ---------------- |
| J1–J2 | 5/8"   | 15.9  | Base to shoulder |
| J2–J3 | 4 1/8" | 104.8 | Upper arm        |
| J3–J4 | 3 1/2" | 88.9  | Forearm          |
| J4–J5 | 2 5/8" | 66.7  | Wrist housing    |
| J5–J6 | 1 1/8" | 28.6  | Rotation housing |
| Claw  | 2 1/4" | 57.2  | Finger length    |

---

### 2. Claw Finger Geometry & J5 Roll (Rodrigues Rotation)

The claw has two fingers that spread symmetrically around the arm's forward
axis. J6 controls the spread angle, and J5 rotates the spread plane.

**Spread angle.** J6 maps linearly to a spread half-angle `α`:

```
t = (J6 − J6_min) / (J6_max − J6_min)      ∈ [0, 1]
α = 2° + t · 28°                             range: 2°–30°
```

**Base perpendicular.** Without J5 roll, the fingers spread along the
direction perpendicular to the arm in the horizontal plane:

```
perp_base = (−sin(ψ), 0, cos(ψ))
```

**J5 roll via Rodrigues rotation.** To rotate `perp_base` around the arm's
forward axis `k = arm_fwd / ‖arm_fwd‖` by angle `ρ` (J5 in radians):

```
v_rot = v · cos(ρ) + (k × v) · sin(ρ) + k · (k · v) · (1 − cos(ρ))
```

This gives `perp_rolled`, the actual spread direction after roll.

**Finger tip positions:**

```
left_tip  = claw_base + arm_fwd · L·cos(α) + perp_rolled · L·sin(α)
right_tip = claw_base + arm_fwd · L·cos(α) − perp_rolled · L·sin(α)
```

where `L = CLAW = 57.2 mm`.

---

### 3. Inverse Kinematics (IK)

The IK solver computes J1–J4 to place the gripper at a target position
pointing straight down (for block pickup). It uses analytical 2-link IK.

**Step 1: J1 base rotation.**

```
r = √(x² + z²)
J1 = atan2(z, x)
```

**Step 2: Reduce to 2D.** Work in the vertical plane containing the target.
The end-effector chain (J4–J5–J6–CLAW) has fixed length `EE = J4_J5 + J5_J6 + CLAW`.
For the gripper to point down, the wrist must be directly above the target at
height `y + EE`. The 2-link problem uses the upper arm (L₁ = J2_J3) and
forearm (L₂ = J3_J4) to reach the wrist position:

```
w_r = r                                    (horizontal distance)
w_y = y + EE − platform_h − J1_J2          (vertical, relative to shoulder)
d   = √(w_r² + w_y²)                       (straight-line distance to wrist)
```

**Reachability check:**

```
|L₁ − L₂| + 1 ≤ d ≤ L₁ + L₂ − 1
```

**Step 3: Two-link IK (law of cosines).**

```
cos(elbow) = (L₁² + L₂² − d²) / (2·L₁·L₂)
J3 = arccos(cos(elbow))

α = atan2(w_y, w_r)
cos(β) = (L₁² + d² − L₂²) / (2·L₁·d)
β = arccos(cos(β))

J2 = π/2 − (α + β)
```

**Step 4: Wrist pitch.** For the gripper to point straight down, the
cumulative pitch angle must equal π:

```
cum = J2 + (π − J3) + J4 = π
→ J4 = J3 − J2
```

---

### 4. Ground Constraint Solver

A greedy iterative solver prevents the arm from penetrating the ground.
Each iteration:

1. Compute FK and finger geometry
2. Find the lowest Y-coordinate across all points (joints + fingertips)
3. If below ground, identify the responsible joints (J2–J4)
4. Try ±2° perturbation on each candidate joint
5. Accept the direction that raises the lowest point most
6. Apply a proportional correction: `clamp(violation × 0.8, 2°, 25°)`
7. Repeat up to 30 iterations until all points are above ground

---

### 5. Auto-Camera Algorithms

All four algorithms share the same structure: build a 3×3 scatter matrix from
the arm's geometry, eigendecompose it, and point the camera along the
**minimum eigenvector** — the direction where the arm has the least visual
structure, so collapsing 3D → 2D along that axis loses the least information.

The camera smoothly tracks via exponential blending each frame:

```
azim += (target_azim − azim) × 0.12        (with 360° wrapping)
elev += (target_elev − elev) × 0.12
```

#### 5.1 PCA — Position Covariance

Builds the covariance matrix from joint **positions**:

```
C = (1/n) · Σᵢ (pᵢ − p̄)(pᵢ − p̄)ᵀ
```

where `pᵢ` are the 7 joint positions and `p̄` is their centroid.

Camera direction = eigenvector of C with smallest eigenvalue.

**Properties:** Implicitly weights longer links more because they push joints
farther apart, increasing their contribution to the covariance. Analogous to
Euclidean distance — optimizes for maximum spatial spread of projected points.

**Best for:** Arm extended, reaching for blocks, pick-and-place operations
where spatial awareness matters most.

#### 5.2 NORM — Normalized Orthogonality

Builds the direction scatter matrix from **unit** link direction vectors:

```
M = Σᵢ d̂ᵢ d̂ᵢᵀ
```

where `d̂ᵢ = (pᵢ₊₁ − pᵢ) / ‖pᵢ₊₁ − pᵢ‖` is the unit direction of link `i`.

Camera direction = eigenvector of M with smallest eigenvalue = direction
most perpendicular to all links simultaneously.

**Properties:** Every link gets equal vote regardless of length. A 15.9mm
base link has the same influence as the 104.8mm upper arm. Pure directional
diversity analysis. Analogous to cosine similarity — cares only about
directions, not magnitudes.

**Best for:** Debugging joint configurations, folded/compact poses where PCA
sees a degenerate blob.

#### 5.3 LINEAR — Length-Weighted Orthogonality

Same as NORM but each link's contribution is weighted by its length:

```
M = Σᵢ Lᵢ · d̂ᵢ d̂ᵢᵀ
```

**Properties:** Hybrid approach. The 104.8mm upper arm gets 6.6× more vote
than the 15.9mm base (vs 1× in NORM, vs 43× in UNNORM). Keeps the
directional framework but adds importance scaling.

**Best for:** General all-rounder when you want both spatial awareness and
joint detail.

#### 5.4 UNNORM — Unnormalized (L²-Weighted)

Uses raw (non-normalized) link vectors:

```
M = Σᵢ dᵢ dᵢᵀ = Σᵢ Lᵢ² · d̂ᵢ d̂ᵢᵀ
```

**Properties:** Length-squared weighting. The 104.8mm upper arm has
L² = 10,983 while the 15.9mm base has L² = 253 — a 43:1 ratio. Essentially
"point the camera perpendicular to the longest link."

**Best for:** When one link dominates the visual. Rarely the best general
choice.

#### 5.5 Comparison — PCA vs Orthogonality

Both approaches are eigendecompositions of scatter matrices, but they operate
on different data:

| Aspect    | PCA (positions)            | NORM (directions)         |
| --------- | -------------------------- | ------------------------- |
| Input     | Joint positions pᵢ         | Unit link vectors d̂ᵢ     |
| Matrix    | Position covariance        | Direction scatter         |
| Optimizes | Max projected point spread | Max projected link length |
| Weighting | Implicit (via distance)    | Equal (unit vectors)      |
| Analogy   | Euclidean distance         | Cosine similarity         |

They agree when the arm is in a planar pose (both find the perpendicular to
the plane). They diverge most when the arm is folded into a zigzag — PCA
sees spatially close joints and picks an arbitrary angle, while NORM finds
the direction that shows every link bend clearly.

The LINEAR variant is equivalent to TF-IDF in the analogy: directional
framework with importance weighting.

---

### 6. Block Physics

Simple rigid-body simulation with semi-implicit Euler integration.

**Constants:**

| Parameter   | Value | Description                           |
| ----------- | ----- | ------------------------------------- |
| Gravity     | −4000 | mm/s² (accelerated for visual impact) |
| Bounce      | 0.3   | Coefficient of restitution            |
| Friction    | 0.92  | Velocity damping per bounce           |
| Drag        | 0.998 | Air resistance per step               |
| Grab radius | 45 mm | Distance for claw to grab             |
| Block size  | 25 mm | Half-width of cube                    |

**Integration step (per frame):**

```
velocity.y += GRAVITY × dt
position   += velocity × dt
velocity   ×= 0.998                         (air drag)
```

**Ground collision:**

```
if position.y − block_size < 0:
    position.y = block_size
    if |velocity.y| < 30:                    (rest threshold)
        velocity = 0;  resting = true
    else:
        velocity.y ×= −BOUNCE
        velocity.x ×= FRICTION
        velocity.z ×= FRICTION
```

**Grab/release:** When the claw is closed and a block's center is within
`GRAB_RADIUS` (45mm) of the grip center, the block snaps to the grip.
On release, the block inherits 60% of the grip's velocity for throwing.

---

### 7. Macro Engine

Step sequencer with smoothstep interpolation.

**Step types:**

| Type     | Parameter         | Effect                      |
| -------- | ----------------- | --------------------------- |
| `open`   | —                 | Set J6 = 73° (fully open)   |
| `close`  | —                 | Set J6 = 10° (fully closed) |
| `move`   | [x, y, z] target  | IK-solve to position        |
| `angles` | {J1:v, J2:v, ...} | Direct angle targets        |

**Smoothstep interpolation.** Each step blends from start to target using
the smoothstep function for natural-looking motion (ease in/out):

```
t = clamp(elapsed / duration, 0, 1)
f = t² · (3 − 2t)                           (smoothstep)
angle = start × (1 − f) + target × f
```

---

### 8. Claw Move (Click-to-Position)

Screen clicks are projected onto the arm's ground plane using a two-pass
grid search against matplotlib's 3D projection matrix.

**Pass 1 (coarse):** Sample a 26×26 grid of points on the ground plane
(Y = platform_h) spanning ±250mm. For each, project to screen coordinates
via `proj3d.proj_transform` + `ax.transData.transform`, find the closest
match to the click position.

**Pass 2 (fine):** Refine with a 21×21 grid (±20mm) around the best coarse
match, achieving ~2mm accuracy.

The resulting ground XZ coordinate is clamped to the arm's workspace radius
(`J2_J3 + J3_J4 − 30 mm`) and passed to the IK solver at Y = 25mm (hover
height for block approach).

---

## Files

| File                | Lines | Description                                               |
| ------------------- | ----- | --------------------------------------------------------- |
| `main.py`           | 1370  | GUI, keyboard, 3D view, 4 camera algorithms, macros, help |
| `arm_kinematics.py` | 185   | FK chain, IK solver, finger geometry, ground constraint   |
| `block_physics.py`  | 85    | Gravity, collision, grab/throw mechanics                  |
| `config.py`         | 73    | Measured dimensions, servo limits, GUI colors             |

## Camera Modes (press V)

| Mode   | Matrix          | Weighting   | Best For                   |
| ------ | --------------- | ----------- | -------------------------- |
| MANUAL | —               | —           | Presentations, inspection  |
| PCA    | Pos. covariance | Implicit    | Extended reach, pick-place |
| NORM   | Σ d̂ᵢd̂ᵢᵀ       | Equal (1×)  | Debugging joints, folded   |
| LINEAR | Σ Lᵢ·d̂ᵢd̂ᵢᵀ    | Length (L×) | General all-rounder        |
| UNNORM | Σ dᵢdᵢᵀ         | L²×         | Single-link dominated      |

## Preset Motions

| Gesture    | Description                          |
| ---------- | ------------------------------------ |
| Wave       | J1 ±40°, J3=140° extended, 3 cycles  |
| Bow        | J2→100°, J3→130° forward dip + hold  |
| Nod        | J3=140°, J4 ±40° wrist pitch, 3×     |
| Shake      | J3=80°, J5 ±60° wrist roll, 3×       |
| Spin       | J1: 0°→90°→−90°→0° full sweep        |
| Flex       | J2/J3 curl (90°/140°) + extend, 2×   |
| Block Wave | Pickup → wave ±50° J1 → place back   |
| Toss       | Pickup → windup → fast throw (0.15s) |
| Stack      | Pickup block A → place on block B    |

```

```
