# MARS — Two-Hand Robot Arm Sim

## Setup
```bash
pip install -r requirements.txt
python main.py
```

## Controls
| Key   | Action                |
|-------|-----------------------|
| Space | Calibrate             |
| M     | Toggle hand tracking  |
| V     | Auto/manual 3D view   |
| L     | Lock/unlock view      |
| B     | Spawn block           |
| H     | Home (uses offsets)   |
| 1-4   | Macro pickup (R/B/G/Y)|
| Esc   | Quit                  |

## Wink to Close Claw
FaceMesh tracks both eyes. A wink (one eye closed, other open)
toggles the claw open/closed. Eye indicators show on the camera feed
with EAR values. Cooldown prevents rapid toggling.

## Finger Sensitivity
Two sliders control how responsive each hand is:
- **L-Hand**: multiplier on left hand's contribution to J1/J2/J3
- **R-Hand**: multiplier on right hand's contribution to J4/J5

Low = sluggish/stable, High = twitchy/precise.

## Mouse Drag
- **Claw Drag ON**: horizontal drag = move in view-perpendicular XZ plane,
  vertical drag = move up/down
- **Grabbed block**: same controls — drag the block, arm follows via IK
- View is locked during drag to prevent accidental rotation

## Home Offsets (Dials)
All dials show offset from default home. Dial at 0 = no change.
J1/J5 = rotation dials, J2/J3/J4 = pitch dials. Arrows in 3D show direction.
