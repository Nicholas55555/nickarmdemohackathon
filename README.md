# MARS — Keyboard Robot Arm Sim

## Setup
```bash
pip install -r requirements.txt
python main.py
```

Press **F1** or click **? Help** for the full in-app guide.

## Quick Reference

```
A/D → J1 base    R/F → J4 wrist    H = Home     V = Cycle camera
W/S → J2 shoulder T/G → J5 roll    B = Spawn    Shift+L = Lock view
Q/E → J3 elbow   Z/X → J6 claw    1-4 = Macro  F1 = Help
```

## Camera Modes (press V)

| Mode   | Optimizes | Best For |
|--------|-----------|----------|
| MANUAL | User control | Presentations, precise inspection |
| PCA    | Point spread | Extended reach, pick-and-place |
| NORM   | Equal link visibility | Debugging joints, folded poses |
| LINEAR | Weighted link visibility | General all-rounder |
| UNNORM | Longest link visibility | Single-link dominated poses |