# opencv_handtracking

## Conda setup (recommended)

```bash
conda env create -f environment.yml
conda activate opencv-handtracking
python -c "import cv2, mediapipe, numpy, scipy; print('Imports OK')"
python handtracking.py
```

If you change dependencies later:

```bash
conda env update -f environment.yml --prune
```

## Pip-only setup (fallback)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
python handtracking.py
```

## Notes

- The model file `hand_landmarker.task` must be present next to `handtracking.py`.
- If the webcam doesn’t open, try changing the camera index in the script (e.g. `cv2.VideoCapture(1)`).
