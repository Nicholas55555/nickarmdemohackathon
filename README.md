# MARS — Two-Hand Robot Arm Sim

## Setup
```bash
pip install -r requirements.txt
python main.py
```
If model error: `pip install mediapipe==0.10.14`

## Two-Hand Control

No Pose model needed — only MediaPipe Hands (detects 2 hands simultaneously).

```
LEFT HAND (green, positioning)        RIGHT HAND (orange, claw)
─────────────────────────────         ────────────────────────────
Wrist X position  → J1 base rot      Hand tilt angle → J4 wrist pitch
Wrist Y position  → J2 shoulder      Palm roll       → J5 wrist rotation
Hand depth (scale) → J3 elbow        Thumb vs 4 tips → J6 claw open/close
```

**Left hand** controls WHERE the arm goes — move it left/right for base rotation,
up/down for shoulder, closer/farther from camera for elbow extension.

**Right hand** controls the CLAW — tilt your hand to pitch the wrist, roll your
palm to rotate, and pinch your thumb against your four fingers to close the claw
(like a claw machine gesture).

## Controls
| Key   | Action                          |
|-------|---------------------------------|
| Space | Calibrate both hands            |
| M     | Toggle hand tracking            |
| V     | Toggle auto / manual 3D view    |
| B     | Spawn a new block               |
| H     | Home position (cancels macros)  |
| 1     | Auto-pickup Red block           |
| 2     | Auto-pickup Blue block          |
| 3     | Auto-pickup Green block         |
| 4     | Auto-pickup Yellow block        |
| Enter | Apply J1 home angle             |
| Esc   | Quit                            |

## How To Use

1. Start camera
2. Hold BOTH hands in front of the webcam
3. Press **Space** — calibrates current hand positions as the arm's home
4. Press **M** — enables tracking
5. Move your LEFT hand to position the arm
6. Use your RIGHT hand to control the claw:
   - Tilt hand up/down for wrist pitch
   - Roll palm left/right for wrist rotation
   - Pinch thumb against all 4 fingers to close claw (grab blocks)
   - Open hand to release (throw)

## Hand Assignment

The tracker uses MediaPipe's handedness classification (after mirror flip).
If both hands get the same label, it falls back to X-position sorting
(left side of screen = right hand for claw, right side = left hand for position).

## Calibration

Press Space with both hands visible. This captures:
- Current joint angles as home offset (all J1–J5)
- Left hand scale (wrist-to-MCP distance) as depth reference for J3

The J1 Home Angle text field lets you set what angle J1 calibrates to
(default 0°, change to orient the base differently).
