"""
Configuration — robot dimensions, servo limits, camera, tuning.

Measured joint distances (inches → mm):
  J1-M  base rotation        0"        0.0 mm
  J2-S  shoulder pivot       5/8"     15.9 mm
  J3-S  elbow               4 1/8"   104.8 mm
  J4-S  wrist pitch         3 1/2"    88.9 mm
  J5-R  wrist rotation      2 5/8"    66.7 mm
  J6-C  claw base           1 1/8"    28.6 mm
  Claw finger length        2 1/4"    57.2 mm
"""

# ── Link lengths (mm) ────────────────────────────────────────────────
J1_J2 =  15.9
J2_J3 = 104.8
J3_J4 =  88.9
J4_J5 =  66.7
J5_J6 =  28.6
CLAW  =  57.2

# ── Link widths for 3D box model (mm) ────────────────────────────────
LINK_W = {
    "base":  50.0,   # base platform radius
    "J2_J3": 22.0,   # upper arm width
    "J3_J4": 18.0,   # forearm width
    "J4_J5": 14.0,   # wrist width
    "J5_J6": 12.0,   # rotation housing
    "claw":  10.0,   # claw finger width
}

# ── Raised base platform (mm) ───────────────────────────────────────
# The arm sits on a platform/pedestal that raises it off the ground.
# Height is adjustable via GUI slider.
PLATFORM_H     = 100.0    # default height (mm)
PLATFORM_H_MIN =   0.0
PLATFORM_H_MAX = 400.0
PLATFORM_W     =  80.0    # width (X) of the platform box (mm)
PLATFORM_D     =  80.0    # depth (Z) of the platform box (mm)

# ── Servo limits (min, max, home) degrees ────────────────────────────
SERVO = {
    "J1": (-135.0, 135.0,   0.0),
    "J2": ( -20.0, 120.0,  45.0),
    "J3": ( -30.0, 150.0,  90.0),
    "J4": ( -90.0,  90.0,   0.0),
    "J5": (-135.0, 135.0,   0.0),
    "J6": (  10.0,  73.0,  73.0),
}
HOME = {k: v[2] for k, v in SERVO.items()}

# ── Camera ───────────────────────────────────────────────────────────
CAM_INDEX  = 0
CAM_W      = 640
CAM_H      = 480
CAM_FOV    = 60.0        # fallback FOV if auto-detection fails
CAM_FOV_AUTO = True      # try to detect FOV from camera properties

# ── Human reference ──────────────────────────────────────────────────
KNOWN_SHOULDER_WIDTH = 40.0   # cm (measure your own)

# ── Tracking tuning ──────────────────────────────────────────────────
SMOOTH_FRAMES  = 5
BLEND_ALPHA    = 0.20

# ── GUI ──────────────────────────────────────────────────────────────
BG      = "#1a1a2e"
FG      = "#e0e0e0"
ACCENT  = "#00d4ff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#51cf66"
PANEL   = "#16213e"
