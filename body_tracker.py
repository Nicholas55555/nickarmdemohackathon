"""
Two-Hand Tracker with Forearm Integration.

LEFT HAND  → J1 base, J2 shoulder, J3 elbow (position from wrist coords)
RIGHT HAND → J4 wrist pitch, J5 roll, J6 claw (gestures)

MediaPipe Pose runs alongside Hands to provide forearm direction
(elbow→wrist vector). This gives J4/J5 a stable reference frame
instead of measuring against horizontal.

Pose landmarks used:
  - Right elbow + wrist → forearm direction for J4 pitch reference
  - Right shoulder → arm plane normal for J5 roll reference
  - Shoulder width → depth estimation (backup)
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
    _pose_mod = mp.solutions.pose
    _draw = mp.solutions.drawing_utils
    _API = "solutions"
except AttributeError:
    pass

if _API is None:
    try:
        _HL = mp.tasks.vision.HandLandmarker
        _HLOpts = mp.tasks.vision.HandLandmarkerOptions
        _PL = mp.tasks.vision.PoseLandmarker
        _PLOpts = mp.tasks.vision.PoseLandmarkerOptions
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
_POSE_MODEL = os.path.join(_DIR, "pose_landmarker_lite.task")

# ══════════════════════════════════════════════════════════════════════
# Landmark indices
# ══════════════════════════════════════════════════════════════════════
# Hand
WRIST = 0
THUMB_CMC = 1; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7; INDEX_TIP = 8
MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP = 13; RING_TIP = 16
PINKY_MCP = 17; PINKY_TIP = 20

# Pose (after cv2.flip: MP LEFT = person's RIGHT)
P_L_SHOULDER = 11; P_L_ELBOW = 13; P_L_WRIST = 15
P_R_SHOULDER = 12; P_R_ELBOW = 14; P_R_WRIST = 16

_HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]
_SKEL = [(11,13),(13,15),(12,14),(14,16),(11,12)]

# ══════════════════════════════════════════════════════════════════════
# Math
# ══════════════════════════════════════════════════════════════════════
def _clamp(v, lo, hi): return max(lo, min(hi, v))
def _remap(v, slo, shi, dlo, dhi):
    return dlo + max(0., min(1., (v-slo)/(shi-slo+1e-9))) * (dhi-dlo)

def _signed_angle_2d(v, ref):
    cross = ref[0]*v[1] - ref[1]*v[0]
    dot = ref[0]*v[0] + ref[1]*v[1]
    return math.degrees(math.atan2(cross, dot))

def detect_camera_fov(cap):
    if not CAM_FOV_AUTO: return CAM_FOV
    aw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ah = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if aw > 0 and ah > 0:
        fov = 75. if aw/ah > 1.6 else 68. if aw/ah > 1.5 else 60.
        print(f"[FOV] {aw:.0f}x{ah:.0f} -> {fov:.0f} deg"); return fov
    return CAM_FOV


# ══════════════════════════════════════════════════════════════════════
class HandPairTracker:
    """Two-hand tracker with Pose forearm integration."""

    def __init__(self):
        self.fov_deg = CAM_FOV
        self.focal_px = (CAM_W / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)
        self._bufs = {f"J{i}": deque(maxlen=SMOOTH_FRAMES) for i in range(1, 7)}
        self.calibrated = False
        self._cal = {f"J{i}": 0.0 for i in range(1, 7)}
        self._homes = {j: SERVO[j][2] for j in [f"J{i}" for i in range(1, 7)]}
        self._cal_hand_scale = None
        self._pose_available = False
        # Forearm sensitivity: 0=hand-only, 100=forearm-dominant
        self.j1_forearm_sens = 70
        self.j2_forearm_sens = 50
        self.j4_forearm_sens = 0   # claw pitch — default off
        self.j5_forearm_sens = 0   # claw roll — default off
        self._init_detectors()

    def set_fov(self, f):
        self.fov_deg = f
        self.focal_px = (CAM_W / 2.0) / math.tan(math.radians(f) / 2.0)

    def set_j1_home(self, d): self._homes["J1"] = d
    def set_joint_home(self, joint, deg): self._homes[joint] = deg

    def _init_detectors(self):
        if _API == "solutions":
            self._hands = _hands_mod.Hands(
                static_image_mode=False, max_num_hands=2,
                min_detection_confidence=0.6, min_tracking_confidence=0.5,
                model_complexity=1)
            try:
                self._pose = _pose_mod.Pose(
                    static_image_mode=False, model_complexity=0,  # lite for speed
                    min_detection_confidence=0.5, min_tracking_confidence=0.4)
                self._pose_available = True
                print("[Tracker] Pose model loaded (forearm integration ON)")
            except Exception as e:
                self._pose = None
                print(f"[Tracker] Pose model unavailable: {e}")
        else:
            if not os.path.exists(_HAND_MODEL):
                raise RuntimeError(f"Hand model not found: {_HAND_MODEL}\nOR: pip install mediapipe==0.10.14")
            self._hands = _HL.create_from_options(_HLOpts(
                base_options=_BaseOpt(model_asset_path=_HAND_MODEL),
                running_mode=_RunMode.IMAGE, num_hands=2,
                min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))
            if os.path.exists(_POSE_MODEL):
                try:
                    self._pose = _PL.create_from_options(_PLOpts(
                        base_options=_BaseOpt(model_asset_path=_POSE_MODEL),
                        running_mode=_RunMode.IMAGE, num_poses=1,
                        min_pose_detection_confidence=0.5, min_tracking_confidence=0.4))
                    self._pose_available = True
                    print("[Tracker] Pose model loaded (forearm integration ON)")
                except:
                    self._pose = None
            else:
                self._pose = None
                print("[Tracker] Pose model not found — forearm integration OFF")

    # ══════════════════════════════════════════════════════════════════
    # DETECTION
    # ══════════════════════════════════════════════════════════════════

    def _detect_hands(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if _API == "solutions":
            res = self._hands.process(rgb)
            if not res.multi_hand_landmarks: return []
            hands = []
            for i, hlms in enumerate(res.multi_hand_landmarks):
                lms = [(lm.x, lm.y, lm.z) for lm in hlms.landmark]
                label = "Right"
                if res.multi_handedness and i < len(res.multi_handedness):
                    label = res.multi_handedness[i].classification[0].label
                hands.append(dict(landmarks=lms, label=label))
            return hands
        else:
            mi = _MpImage(image_format=_ImgFmt.SRGB, data=rgb)
            res = self._hands.detect(mi)
            if not res.hand_landmarks: return []
            hands = []
            for i, hlms in enumerate(res.hand_landmarks):
                lms = [(lm.x, lm.y, lm.z) for lm in hlms]
                label = "Right"
                if res.handedness and i < len(res.handedness):
                    label = res.handedness[i][0].category_name
                hands.append(dict(landmarks=lms, label=label))
            return hands

    def _detect_pose(self, bgr):
        """Returns (pose_px, frame) or (None, frame). pose_px = list of (x_norm,y_norm,z)."""
        if not self._pose_available or self._pose is None:
            return None, bgr.copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = bgr.copy()
        h, w = bgr.shape[:2]
        if _API == "solutions":
            res = self._pose.process(rgb)
            if not res.pose_landmarks:
                return None, frame
            # Draw skeleton lightly
            for a, b in _SKEL:
                lms = res.pose_landmarks.landmark
                if a < len(lms) and b < len(lms):
                    pa = (int(lms[a].x*w), int(lms[a].y*h))
                    pb = (int(lms[b].x*w), int(lms[b].y*h))
                    cv2.line(frame, pa, pb, (50, 50, 50), 1)
            return [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark], frame
        else:
            mi = _MpImage(image_format=_ImgFmt.SRGB, data=rgb)
            res = self._pose.detect(mi)
            if not res.pose_landmarks or len(res.pose_landmarks) == 0:
                return None, frame
            pl = res.pose_landmarks[0]
            for a, b in _SKEL:
                if a < len(pl) and b < len(pl):
                    pa = (int(pl[a].x*w), int(pl[a].y*h))
                    pb = (int(pl[b].x*w), int(pl[b].y*h))
                    cv2.line(frame, pa, pb, (50, 50, 50), 1)
            return [(lm.x, lm.y, lm.z) for lm in pl], frame

    # ══════════════════════════════════════════════════════════════════
    # LEFT HAND → J1, J2, J3
    # ══════════════════════════════════════════════════════════════════

    def _extract_position_joints(self, lms, h, w, forearm_dir=None):
        """
        Left hand controls arm position.

        Forearm sensitivity (0-100) controls the blend:
          0   = pure hand position (wrist X/Y → full range)
          100 = forearm dominant (forearm angle → full range, hand → ±15° fine)
        """
        wrist = np.array([lms[WRIST][0], lms[WRIST][1]])
        mid_mcp = np.array([lms[MIDDLE_MCP][0], lms[MIDDLE_MCP][1]])
        hand_scale = np.linalg.norm(wrist - mid_mcp)

        j1_s = self.j1_forearm_sens / 100.0  # 0..1
        j2_s = self.j2_forearm_sens / 100.0

        # ── J1: base rotation ────────────────────────────────────────
        # Hand component: wrist X → ±90°
        j1_hand = -(wrist[0] - 0.5) * 2.0 * 90.0
        if forearm_dir is not None and np.linalg.norm(forearm_dir) > 0.01 and j1_s > 0.01:
            j1_forearm = math.degrees(math.atan2(-forearm_dir[0], forearm_dir[1]))
            # Blend: at sens=1.0, forearm provides full + hand adds ±15° fine
            # At sens=0.5, 50/50 blend
            # At sens=0.0, pure hand
            j1_fine_range = 15.0 * j1_s  # shrink hand range as forearm takes over
            j1_hand_scaled = -(wrist[0] - 0.5) * 2.0 * (90.0 * (1 - j1_s) + j1_fine_range)
            j1 = j1_forearm * j1_s + j1_hand_scaled
        else:
            j1 = j1_hand
        j1 = _clamp(j1, *SERVO["J1"][:2])

        # ── J2: shoulder pitch ───────────────────────────────────────
        j2_hand = _remap(wrist[1], 0.2, 0.8, SERVO["J2"][0], SERVO["J2"][1])
        if forearm_dir is not None and np.linalg.norm(forearm_dir) > 0.01 and j2_s > 0.01:
            vert_ratio = forearm_dir[1] / (np.linalg.norm(forearm_dir) + 1e-9)
            j2_forearm = _remap(vert_ratio, -0.8, 0.8, SERVO["J2"][0], SERVO["J2"][1])
            j2_fine = (wrist[1] - 0.5) * 2.0 * 20.0 * j2_s
            j2 = j2_forearm * j2_s + j2_hand * (1 - j2_s) + j2_fine * j2_s
        else:
            j2 = j2_hand
        j2 = _clamp(j2, *SERVO["J2"][:2])

        # ── J3: elbow (hand scale only) ──────────────────────────────
        if self._cal_hand_scale is not None and self._cal_hand_scale > 0.01:
            depth_ratio = hand_scale / self._cal_hand_scale
            j3 = _remap(depth_ratio, 0.5, 1.5, SERVO["J3"][0]+10, SERVO["J3"][1]-10)
        else:
            j3 = _remap(hand_scale, 0.05, 0.20, SERVO["J3"][0]+10, SERVO["J3"][1]-10)
        j3 = _clamp(j3, *SERVO["J3"][:2])
        return dict(J1=j1, J2=j2, J3=j3), hand_scale

    # ══════════════════════════════════════════════════════════════════
    # RIGHT HAND → J4, J5, J6 (with optional forearm blending)
    # ══════════════════════════════════════════════════════════════════

    def _extract_claw_joints(self, lms, h, w, forearm_dir=None):
        """
        Right hand controls claw.
        At sensitivity 0: pure horizontal reference (hand gestures only).
        At sensitivity 100: forearm-relative (wrist bend = J4, palm roll = J5).
        """
        wrist = np.array([lms[WRIST][0], lms[WRIST][1]])
        mid_tip = np.array([lms[MIDDLE_TIP][0], lms[MIDDLE_TIP][1]])
        idx_mcp = np.array([lms[INDEX_MCP][0], lms[INDEX_MCP][1]])
        pk_mcp = np.array([lms[PINKY_MCP][0], lms[PINKY_MCP][1]])
        thumb_tip = np.array([lms[THUMB_TIP][0], lms[THUMB_TIP][1]])
        idx_tip = np.array([lms[INDEX_TIP][0], lms[INDEX_TIP][1]])
        mid_tip_2d = mid_tip
        ring_tip = np.array([lms[RING_TIP][0], lms[RING_TIP][1]])
        pk_tip = np.array([lms[PINKY_TIP][0], lms[PINKY_TIP][1]])

        horizontal = np.array([1.0, 0.0])
        hand_dir = mid_tip - wrist
        palm_vec = pk_mcp - idx_mcp

        j4_s = self.j4_forearm_sens / 100.0
        j5_s = self.j5_forearm_sens / 100.0
        has_fa = forearm_dir is not None and np.linalg.norm(forearm_dir) > 1e-6

        # ── J4: wrist pitch ───────────────────────────────────────────
        j4_hand = _signed_angle_2d(hand_dir, horizontal) if np.linalg.norm(hand_dir) > 1e-6 else 0.0
        if has_fa and j4_s > 0.01:
            j4_forearm = _signed_angle_2d(hand_dir, forearm_dir)
            j4 = j4_forearm * j4_s + j4_hand * (1 - j4_s)
        else:
            j4 = j4_hand
        j4 = _clamp(j4, *SERVO["J4"][:2])

        # ── J5: palm roll ─────────────────────────────────────────────
        j5_hand = _signed_angle_2d(palm_vec, horizontal) if np.linalg.norm(palm_vec) > 1e-6 else 0.0
        if has_fa and j5_s > 0.01:
            perp_fa = np.array([-forearm_dir[1], forearm_dir[0]])
            j5_forearm = _signed_angle_2d(palm_vec, perp_fa)
            j5 = j5_forearm * j5_s + j5_hand * (1 - j5_s)
        else:
            j5 = j5_hand
        j5 = _clamp(j5, *SERVO["J5"][:2])

        # ── J6: claw (unchanged) ─────────────────────────────────────
        four_tips_avg = np.mean([idx_tip, mid_tip_2d, ring_tip, pk_tip], axis=0)
        pinch_dist = np.linalg.norm(thumb_tip - four_tips_avg)
        palm_size = np.linalg.norm(idx_mcp - pk_mcp)
        grip_ratio = pinch_dist / (palm_size + 1e-6)
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
            raw_angles={}, frame=bgr_frame.copy(),
            left_hand_ok=False, right_hand_ok=False,
            grip_01=1.0, fov=self.fov_deg,
            forearm_ok=False, l_forearm_ok=False,
        )

        # Run Pose (for both forearm directions)
        pose_px, frame = self._detect_pose(bgr_frame)
        l_forearm_dir = None
        r_forearm_dir = None
        if pose_px is not None:
            # Person's left arm = MP RIGHT after flip → J1/J2
            l_el = np.array([pose_px[P_R_ELBOW][0], pose_px[P_R_ELBOW][1]])
            l_wr = np.array([pose_px[P_R_WRIST][0], pose_px[P_R_WRIST][1]])
            fd_l = l_wr - l_el
            if np.linalg.norm(fd_l) > 0.01:
                l_forearm_dir = fd_l
                out["l_forearm_ok"] = True
                ep = (int(l_el[0]*w), int(l_el[1]*h))
                wp = (int(l_wr[0]*w), int(l_wr[1]*h))
                cv2.line(frame, ep, wp, (60, 120, 60), 2)
                cv2.circle(frame, ep, 4, (60, 120, 60), -1)

            # Person's right arm = MP LEFT after flip → J4/J5
            r_el = np.array([pose_px[P_L_ELBOW][0], pose_px[P_L_ELBOW][1]])
            r_wr = np.array([pose_px[P_L_WRIST][0], pose_px[P_L_WRIST][1]])
            fd_r = r_wr - r_el
            if np.linalg.norm(fd_r) > 0.01:
                r_forearm_dir = fd_r
                out["forearm_ok"] = True
                ep2 = (int(r_el[0]*w), int(r_el[1]*h))
                wp2 = (int(r_wr[0]*w), int(r_wr[1]*h))
                cv2.line(frame, ep2, wp2, (80, 80, 120), 2)
                cv2.circle(frame, ep2, 4, (80, 80, 120), -1)

        # Run Hands
        hands = self._detect_hands(bgr_frame)
        if not hands:
            out["frame"] = frame
            return out

        # Sort into left / right
        left_hand = None; right_hand = None
        for hand in hands:
            if hand["label"] == "Right": right_hand = hand
            else: left_hand = hand

        if len(hands) == 1 and left_hand is None and right_hand is None:
            h0 = hands[0]
            if h0["landmarks"][WRIST][0] < 0.5: right_hand = h0
            else: left_hand = h0

        if left_hand is None and right_hand is not None and len(hands) == 2:
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

        # Left hand → J1, J2, J3 (with left forearm for J1)
        if left_hand is not None:
            lms = left_hand["landmarks"]
            j123, hs = self._extract_position_joints(lms, h, w, l_forearm_dir)
            raw.update(j123)
            out["left_hand_ok"] = True
            out["left_hand_lms"] = lms
            pts = [(int(l[0]*w), int(l[1]*h)) for l in lms]
            for a, b in _HAND_CONN:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for pt in pts: cv2.circle(frame, pt, 3, (0, 200, 0), -1)
            lfa_tag = "+FA" if l_forearm_dir is not None else ""
            cv2.putText(frame, f"L-POS{lfa_tag}", (pts[0][0]-20, pts[0][1]-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Right hand → J4, J5, J6 (with right forearm blending)
        grip_01 = 1.0
        if right_hand is not None:
            lms = right_hand["landmarks"]
            j456, grip_01 = self._extract_claw_joints(lms, h, w, r_forearm_dir)
            raw.update(j456)
            out["right_hand_ok"] = True
            out["grip_01"] = grip_01
            pts = [(int(l[0]*w), int(l[1]*h)) for l in lms]
            for a, b in _HAND_CONN:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (0, 165, 255), 2)
            for pt in pts: cv2.circle(frame, pt, 3, (0, 140, 220), -1)
            fa_tag = "+FA" if r_forearm_dir is not None and (self.j4_forearm_sens > 0 or self.j5_forearm_sens > 0) else ""
            cv2.putText(frame, f"R-CLAW{fa_tag}", (pts[0][0]-25, pts[0][1]-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        if not raw:
            out["frame"] = frame; return out

        # Calibration offset
        if self.calibrated:
            for j in list(raw.keys()):
                if j == "J6": continue
                if j not in self._cal: continue
                home = self._homes.get(j, SERVO[j][2])
                raw[j] = _clamp((raw[j] - self._cal[j]) + home, *SERVO[j][:2])

        out["raw_angles"] = dict(raw)
        smoothed = {}
        for j, val in raw.items():
            self._bufs[j].append(val)
            smoothed[j] = float(np.mean(self._bufs[j]))
        out["angles"] = smoothed
        out["detected"] = True

        # Info overlay
        lh = "L:OK" if out["left_hand_ok"] else "L:--"
        rh = "R:OK" if out["right_hand_ok"] else "R:--"
        lf = "LF" if out.get("l_forearm_ok") else ""
        rf = "RF" if out.get("forearm_ok") else ""
        fa = f" {lf}{rf}" if (lf or rf) else ""
        info = [
            f"{lh}{fa} {rh}",
            f"J1:{smoothed.get('J1',0):+5.0f} J2:{smoothed.get('J2',0):5.0f} J3:{smoothed.get('J3',0):5.0f}",
            f"J4:{smoothed.get('J4',0):+5.0f} J5:{smoothed.get('J5',0):+5.0f} J6:{smoothed.get('J6',0):5.0f}",
            f"Claw: {'CLOSED' if grip_01 < 0.4 else 'OPEN'}",
        ]
        for i, t in enumerate(info):
            cv2.putText(frame, t, (8, 22+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 212, 255), 2)
        ct = "CALIBRATED" if self.calibrated else "SPACE to calibrate"
        cv2.putText(frame, ct, (8, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (81,207,102) if self.calibrated else (120,120,180), 1)
        out["frame"] = frame
        return out

    # ══════════════════════════════════════════════════════════════════
    # CALIBRATION
    # ══════════════════════════════════════════════════════════════════

    def calibrate(self, raw_angles, left_hand_lms=None):
        for j, v in raw_angles.items():
            self._cal[j] = v
        self.calibrated = True
        for b in self._bufs.values(): b.clear()
        if left_hand_lms is not None:
            wrist = np.array([left_hand_lms[WRIST][0], left_hand_lms[WRIST][1]])
            mid_mcp = np.array([left_hand_lms[MIDDLE_MCP][0], left_hand_lms[MIDDLE_MCP][1]])
            self._cal_hand_scale = np.linalg.norm(wrist - mid_mcp)
            if self._cal_hand_scale < 0.01: self._cal_hand_scale = None
        print(f"[Cal] homes={','.join(f'{j}={self._homes[j]:.0f}' for j in ['J1','J2','J3','J4','J5'])}")

    def release(self):
        try: self._hands.close()
        except: pass
        try:
            if self._pose: self._pose.close()
        except: pass