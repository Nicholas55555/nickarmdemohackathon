"""
Two-Hand Tracker — controls 6 joints from two hands.

LEFT HAND (positioning):
  Wrist X position  → J1 base rotation
  Wrist Y position  → J2 shoulder pitch
  Hand depth (scale) → J3 elbow bend

RIGHT HAND (claw):
  Hand pitch (tilt)  → J4 wrist pitch
  Palm roll          → J5 wrist rotation
  Thumb-vs-4-fingers → J6 claw open/close

Uses MediaPipe Hands (max_num_hands=2) with handedness classification
to distinguish left from right.  No Pose model needed.
"""

import os, math, sys
import numpy as np
import cv2
from collections import deque
from config import (
    CAM_W, CAM_H, CAM_FOV, CAM_FOV_AUTO,
    SMOOTH_FRAMES, SERVO,
)

import mediapipe as mp

# ══════════════════════════════════════════════════════════════════════
# API detection
# ══════════════════════════════════════════════════════════════════════
_API = None
try:
    _hands_mod = mp.solutions.hands
    _draw = mp.solutions.drawing_utils
    _API = "solutions"
except AttributeError:
    pass

if _API is None:
    try:
        _HL = mp.tasks.vision.HandLandmarker
        _HLOpts = mp.tasks.vision.HandLandmarkerOptions
        _RunMode = mp.tasks.vision.RunningMode
        _BaseOpt = mp.tasks.BaseOptions
        _MpImage = mp.Image
        _ImgFmt = mp.ImageFormat
        _API = "tasks"
    except AttributeError:
        pass

if _API is None:
    raise RuntimeError("No usable MediaPipe API.  pip install mediapipe==0.10.14")

print(f"[Tracker] mediapipe {mp.__version__}  API={_API}")
_DIR = os.path.dirname(os.path.abspath(__file__))
_HAND_MODEL = os.path.join(_DIR, "hand_landmarker.task")

# ══════════════════════════════════════════════════════════════════════
# Hand landmark indices
# ══════════════════════════════════════════════════════════════════════
WRIST = 0
THUMB_CMC = 1; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7; INDEX_TIP = 8
MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP = 13; RING_PIP = 14; RING_DIP = 15; RING_TIP = 16
PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

_HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),  # thumb
    (0,5),(5,6),(6,7),(7,8),  # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),  # palm
]

# ══════════════════════════════════════════════════════════════════════
# Math helpers
# ══════════════════════════════════════════════════════════════════════
def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _remap(v, slo, shi, dlo, dhi):
    t = max(0.0, min(1.0, (v - slo) / (shi - slo + 1e-9)))
    return dlo + t * (dhi - dlo)

def _signed_angle_2d(v, ref):
    """Signed angle of v relative to ref in 2D (degrees). CCW positive."""
    cross = ref[0] * v[1] - ref[1] * v[0]
    dot = ref[0] * v[0] + ref[1] * v[1]
    return math.degrees(math.atan2(cross, dot))

def detect_camera_fov(cap):
    if not CAM_FOV_AUTO:
        return CAM_FOV
    aw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ah = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if aw > 0 and ah > 0:
        asp = aw / ah
        fov = 75.0 if asp > 1.6 else 68.0 if asp > 1.5 else 60.0
        print(f"[FOV] {aw:.0f}x{ah:.0f} -> {fov:.0f} deg")
        return fov
    return CAM_FOV


# ══════════════════════════════════════════════════════════════════════
class HandPairTracker:
    """
    Tracks two hands: left for arm positioning, right for claw control.
    """

    def __init__(self):
        self.fov_deg = CAM_FOV
        self.focal_px = (CAM_W / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)

        # Per-joint smoothing
        self._bufs = {f"J{i}": deque(maxlen=SMOOTH_FRAMES) for i in range(1, 7)}

        # Calibration
        self.calibrated = False
        self._cal = {f"J{i}": 0.0 for i in range(1, 7)}
        self._j1_home = 0.0

        # Reference hand scale at calibration (for depth estimation)
        self._cal_hand_scale = None

        self._init_detector()

    def set_fov(self, f):
        self.fov_deg = f
        self.focal_px = (CAM_W / 2.0) / math.tan(math.radians(f) / 2.0)

    def set_j1_home(self, d):
        self._j1_home = d

    def _init_detector(self):
        if _API == "solutions":
            self._hands = _hands_mod.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
                model_complexity=1,
            )
        else:
            if not os.path.exists(_HAND_MODEL):
                raise RuntimeError(
                    f"Hand model not found: {_HAND_MODEL}\n"
                    f"OR: pip install mediapipe==0.10.14"
                )
            self._hands = _HL.create_from_options(_HLOpts(
                base_options=_BaseOpt(model_asset_path=_HAND_MODEL),
                running_mode=_RunMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ))

    # ══════════════════════════════════════════════════════════════════
    # DETECTION
    # ══════════════════════════════════════════════════════════════════

    def _detect(self, bgr):
        """
        Returns list of dicts: [{landmarks: [(x,y,z)...], label: 'Left'/'Right'}]
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if _API == "solutions":
            res = self._hands.process(rgb)
            if not res.multi_hand_landmarks:
                return []
            hands = []
            for i, hlms in enumerate(res.multi_hand_landmarks):
                lms = [(lm.x, lm.y, lm.z) for lm in hlms.landmark]
                # Handedness: MediaPipe labels from camera's perspective
                # After cv2.flip, "Left" = person's left, "Right" = person's right
                label = "Right"
                if res.multi_handedness and i < len(res.multi_handedness):
                    label = res.multi_handedness[i].classification[0].label
                hands.append(dict(landmarks=lms, label=label))
            return hands
        else:
            mi = _MpImage(image_format=_ImgFmt.SRGB, data=rgb)
            res = self._hands.detect(mi)
            if not res.hand_landmarks:
                return []
            hands = []
            for i, hlms in enumerate(res.hand_landmarks):
                lms = [(lm.x, lm.y, lm.z) for lm in hlms]
                label = "Right"
                if res.handedness and i < len(res.handedness):
                    label = res.handedness[i][0].category_name
                hands.append(dict(landmarks=lms, label=label))
            return hands

    # ══════════════════════════════════════════════════════════════════
    # LEFT HAND → J1, J2, J3 (positioning)
    # ══════════════════════════════════════════════════════════════════

    def _extract_position_joints(self, lms, h, w):
        """
        Left hand controls arm position:
          Wrist X → J1 (base rotation)
          Wrist Y → J2 (shoulder pitch)
          Hand scale (depth proxy) → J3 (elbow)
        """
        wrist = np.array([lms[WRIST][0], lms[WRIST][1]])
        mid_mcp = np.array([lms[MIDDLE_MCP][0], lms[MIDDLE_MCP][1]])
        mid_tip = np.array([lms[MIDDLE_TIP][0], lms[MIDDLE_TIP][1]])
        idx_mcp = np.array([lms[INDEX_MCP][0], lms[INDEX_MCP][1]])
        pk_mcp = np.array([lms[PINKY_MCP][0], lms[PINKY_MCP][1]])

        # Hand scale: wrist-to-middle-MCP distance (depth proxy)
        hand_scale = np.linalg.norm(wrist - mid_mcp)

        # J1: wrist X position, centered at 0.5, mirrored
        # Left of frame (x<0.5) → negative rotation, right → positive
        j1_raw = -(wrist[0] - 0.5) * 2.0  # -1..+1, mirrored for natural feel
        j1 = j1_raw * 90.0  # ±90° range
        j1 = _clamp(j1, *SERVO["J1"][:2])

        # J2: wrist Y position
        # High hand (y small) → arm up (low J2), low hand (y large) → arm forward (high J2)
        j2 = _remap(wrist[1], 0.2, 0.8, SERVO["J2"][0], SERVO["J2"][1])

        # J3: depth from hand scale
        # Larger hand = closer = more extended (higher J3)
        # Smaller hand = further = more bent (lower J3)
        if self._cal_hand_scale is not None and self._cal_hand_scale > 0.01:
            depth_ratio = hand_scale / self._cal_hand_scale  # >1 = closer, <1 = farther
            j3 = _remap(depth_ratio, 0.5, 1.5, SERVO["J3"][0] + 10, SERVO["J3"][1] - 10)
        else:
            j3 = _remap(hand_scale, 0.05, 0.20, SERVO["J3"][0] + 10, SERVO["J3"][1] - 10)
        j3 = _clamp(j3, *SERVO["J3"][:2])

        return dict(J1=j1, J2=j2, J3=j3), hand_scale

    # ══════════════════════════════════════════════════════════════════
    # RIGHT HAND → J4, J5, J6 (claw control)
    # ══════════════════════════════════════════════════════════════════

    def _extract_claw_joints(self, lms, h, w):
        """
        Right hand controls claw:
          Hand pitch (tilt angle) → J4 (wrist pitch)
          Palm roll              → J5 (wrist rotation)
          Thumb vs 4 fingertips  → J6 (claw open/close)
        """
        wrist = np.array([lms[WRIST][0], lms[WRIST][1], lms[WRIST][2]])
        mid_mcp = np.array([lms[MIDDLE_MCP][0], lms[MIDDLE_MCP][1], lms[MIDDLE_MCP][2]])
        mid_tip = np.array([lms[MIDDLE_TIP][0], lms[MIDDLE_TIP][1], lms[MIDDLE_TIP][2]])
        idx_mcp = np.array([lms[INDEX_MCP][0], lms[INDEX_MCP][1], lms[INDEX_MCP][2]])
        pk_mcp = np.array([lms[PINKY_MCP][0], lms[PINKY_MCP][1], lms[PINKY_MCP][2]])
        thumb_tip = np.array([lms[THUMB_TIP][0], lms[THUMB_TIP][1]])
        idx_tip = np.array([lms[INDEX_TIP][0], lms[INDEX_TIP][1]])
        mid_tip_2d = np.array([lms[MIDDLE_TIP][0], lms[MIDDLE_TIP][1]])
        ring_tip = np.array([lms[RING_TIP][0], lms[RING_TIP][1]])
        pk_tip = np.array([lms[PINKY_TIP][0], lms[PINKY_TIP][1]])

        # ── J4: wrist pitch ───────────────────────────────────────────
        # Angle of hand direction (wrist→middle-tip) vs horizontal
        hand_dir = np.array([mid_tip[0] - wrist[0], mid_tip[1] - wrist[1]])
        horizontal = np.array([1.0, 0.0])
        if np.linalg.norm(hand_dir) > 1e-6:
            j4_raw = _signed_angle_2d(hand_dir, horizontal)
            # Adjust: hand pointing right = 0°, pointing down = +90°, pointing up = -90°
            j4 = _clamp(j4_raw, *SERVO["J4"][:2])
        else:
            j4 = 0.0

        # ── J5: palm roll ─────────────────────────────────────────────
        # Angle of index-to-pinky line vs horizontal
        palm_vec = np.array([pk_mcp[0] - idx_mcp[0], pk_mcp[1] - idx_mcp[1]])
        if np.linalg.norm(palm_vec) > 1e-6:
            j5_raw = _signed_angle_2d(palm_vec, horizontal)
            j5 = _clamp(j5_raw, *SERVO["J5"][:2])
        else:
            j5 = 0.0

        # ── J6: claw ──────────────────────────────────────────────────
        # Thumb tip vs average of 4 fingertips (claw gesture)
        four_tips_avg = np.mean([idx_tip, mid_tip_2d, ring_tip, pk_tip], axis=0)
        pinch_dist = np.linalg.norm(thumb_tip - four_tips_avg)
        # Normalize by palm size
        palm_size = np.linalg.norm(np.array([idx_mcp[0], idx_mcp[1]]) -
                                    np.array([pk_mcp[0], pk_mcp[1]]))
        if palm_size > 1e-6:
            grip_ratio = pinch_dist / palm_size
        else:
            grip_ratio = 1.0
        # grip_ratio ~0.3 = closed (thumb touching fingers), ~1.5 = wide open
        grip_01 = _clamp((grip_ratio - 0.3) / 1.0, 0.0, 1.0)
        j6 = _remap(grip_01, 0, 1, SERVO["J6"][0], SERVO["J6"][1])

        return dict(J4=j4, J5=j5, J6=j6), grip_01

    # ══════════════════════════════════════════════════════════════════
    # MAIN PROCESS
    # ══════════════════════════════════════════════════════════════════

    def process(self, bgr_frame):
        h, w = bgr_frame.shape[:2]
        out = dict(
            detected=False,
            angles={f"J{i}": SERVO[f"J{i}"][2] for i in range(1, 7)},
            raw_angles={},
            frame=bgr_frame.copy(),
            left_hand_ok=False,
            right_hand_ok=False,
            grip_01=1.0,
            fov=self.fov_deg,
        )

        hands = self._detect(bgr_frame)
        if not hands:
            out["frame"] = bgr_frame.copy()
            return out

        frame = bgr_frame.copy()

        # ── Sort into left / right ────────────────────────────────────
        # After cv2.flip, MediaPipe's "Right" label = person's RIGHT hand
        # (because flip mirrors the handedness classification too)
        left_hand = None
        right_hand = None
        for hand in hands:
            # MediaPipe labels from the image's perspective.
            # After flip: "Right" in MP = appears on right side of screen = person's RIGHT
            if hand["label"] == "Right":
                right_hand = hand
            else:
                left_hand = hand

        # If only one hand, try to assign by X position
        if len(hands) == 1 and left_hand is None and right_hand is None:
            h0 = hands[0]
            wx = h0["landmarks"][WRIST][0]
            if wx < 0.5:
                right_hand = h0   # right side of screen (person's right after flip)
            else:
                left_hand = h0

        # If both labels are the same, sort by X position
        if left_hand is None and right_hand is not None and len(hands) == 2:
            # Both labeled "Right" — split by position
            h0, h1 = hands[0], hands[1]
            if h0["landmarks"][WRIST][0] < h1["landmarks"][WRIST][0]:
                right_hand, left_hand = h0, h1
            else:
                right_hand, left_hand = h1, h0
        elif right_hand is None and left_hand is not None and len(hands) == 2:
            h0, h1 = hands[0], hands[1]
            if h0["landmarks"][WRIST][0] < h1["landmarks"][WRIST][0]:
                right_hand, left_hand = h0, h1
            else:
                right_hand, left_hand = h1, h0

        raw = {}

        # ── Process left hand (J1, J2, J3) ────────────────────────────
        if left_hand is not None:
            lms = left_hand["landmarks"]
            j123, hand_scale = self._extract_position_joints(lms, h, w)
            raw.update(j123)
            out["left_hand_ok"] = True
            out["left_hand_lms"] = lms  # for calibration depth reference

            # Draw left hand (green)
            pts = [(int(l[0] * w), int(l[1] * h)) for l in lms]
            for a, b in _HAND_CONN:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for pt in pts:
                cv2.circle(frame, pt, 3, (0, 200, 0), -1)
            cv2.putText(frame, "L-POS", (pts[0][0] - 20, pts[0][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ── Process right hand (J4, J5, J6) ───────────────────────────
        grip_01 = 1.0
        if right_hand is not None:
            lms = right_hand["landmarks"]
            j456, grip_01 = self._extract_claw_joints(lms, h, w)
            raw.update(j456)
            out["right_hand_ok"] = True
            out["grip_01"] = grip_01

            # Draw right hand (orange)
            pts = [(int(l[0] * w), int(l[1] * h)) for l in lms]
            for a, b in _HAND_CONN:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (0, 165, 255), 2)
            for pt in pts:
                cv2.circle(frame, pt, 3, (0, 140, 220), -1)
            cv2.putText(frame, "R-CLAW", (pts[0][0] - 25, pts[0][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        if not raw:
            out["frame"] = frame
            return out

        # ── Calibration offset ────────────────────────────────────────
        if self.calibrated:
            for j in raw:
                if j == "J6":
                    continue
                home = self._j1_home if j == "J1" else SERVO[j][2]
                raw[j] = _clamp((raw[j] - self._cal[j]) + home, *SERVO[j][:2])

        out["raw_angles"] = dict(raw)

        # ── Smooth ────────────────────────────────────────────────────
        smoothed = {}
        for j, val in raw.items():
            self._bufs[j].append(val)
            smoothed[j] = float(np.mean(self._bufs[j]))
        out["angles"] = smoothed
        out["detected"] = True

        # ── Info overlay ──────────────────────────────────────────────
        lh = "L:OK" if out["left_hand_ok"] else "L:--"
        rh = "R:OK" if out["right_hand_ok"] else "R:--"
        info = [
            f"{lh} {rh}  FOV:{self.fov_deg:.0f}",
            f"J1:{smoothed.get('J1', 0):+5.0f} J2:{smoothed.get('J2', 0):5.0f} J3:{smoothed.get('J3', 0):5.0f}",
            f"J4:{smoothed.get('J4', 0):+5.0f} J5:{smoothed.get('J5', 0):+5.0f} J6:{smoothed.get('J6', 0):5.0f}",
            f"Claw: {'CLOSED' if grip_01 < 0.4 else 'OPEN'}",
        ]
        for i, t in enumerate(info):
            cv2.putText(frame, t, (8, 22 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 212, 255), 2)

        cal_txt = "CALIBRATED" if self.calibrated else "SPACE to calibrate"
        cal_col = (81, 207, 102) if self.calibrated else (120, 120, 180)
        cv2.putText(frame, cal_txt, (8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, cal_col, 1)

        out["frame"] = frame
        return out

    # ══════════════════════════════════════════════════════════════════
    # CALIBRATION
    # ══════════════════════════════════════════════════════════════════

    def calibrate(self, raw_angles, left_hand_lms=None):
        """
        Store current angles as home reference.
        If left hand landmarks available, store hand scale for depth reference.
        """
        self._cal = dict(raw_angles)
        self.calibrated = True
        for b in self._bufs.values():
            b.clear()

        # Store hand scale for depth calibration
        if left_hand_lms is not None:
            wrist = np.array([left_hand_lms[WRIST][0], left_hand_lms[WRIST][1]])
            mid_mcp = np.array([left_hand_lms[MIDDLE_MCP][0], left_hand_lms[MIDDLE_MCP][1]])
            self._cal_hand_scale = np.linalg.norm(wrist - mid_mcp)
            if self._cal_hand_scale < 0.01:
                self._cal_hand_scale = None

        print(f"[Cal] J1={raw_angles.get('J1', 0):.1f} home={self._j1_home:.1f}"
              f"  scale={'%.3f' % self._cal_hand_scale if self._cal_hand_scale else 'none'}")

    def release(self):
        try:
            self._hands.close()
        except:
            pass
