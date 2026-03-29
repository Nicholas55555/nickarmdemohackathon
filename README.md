# MARS — Keyboard Robot Arm Sim

## Setup
```bash
pip install -r requirements.txt
python main.py
```

No camera, no MediaPipe — just keyboard and mouse.

## Keyboard Controls (hold for continuous movement)

```
LEFT HAND (position)     RIGHT HAND (claw)
─────────────────────    ─────────────────────
 A / D  →  J1 base       R / F  →  J4 wrist
 W / S  →  J2 shoulder   T / G  →  J5 rotation
 Q / E  →  J3 elbow      Z / X  →  J6 claw
```

| Key     | Action                |
|---------|-----------------------|
| H       | Home position         |
| B       | Spawn block           |
| V       | Auto/manual 3D view   |
| Shift+L | Lock/unlock view      |
| 1-4     | Macro pickup (R/B/G/Y)|
| Esc     | Quit                  |

## Speed
Adjust the **Key Speed** slider (0.5–10.0 °/tick) to control how
fast held keys move the joints.

## Claw Move
Click **Claw Move: ON** then click anywhere in the 3D view.
The arm IK-solves to that ground position instantly.

## Files
- `main.py` — GUI, keyboard, 3D view, physics
- `arm_kinematics.py` — FK, IK, finger geometry
- `block_physics.py` — gravity, grab/throw
- `config.py` — dimensions, servo limits, tuning
