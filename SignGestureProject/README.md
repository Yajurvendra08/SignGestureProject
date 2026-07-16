# Sign Gesture Recognition ✋🤖

A beginner-friendly Python project that recognizes hand gestures through your
webcam and translates them into English text. No GPU or deep learning
framework required.

There are three ways to run this project:

| | **Pretrained (zero training)** | **Landmark pipeline** | **Pixel pipeline (legacy)** |
|---|---|---|---|
| Setup needed | None — just run it | Capture ~50-80 samples per gesture | Capture ~150+ images per gesture |
| What it recognizes | 7 fixed common gestures (see below) | Any custom gesture you teach it | Any custom gesture you teach it |
| How it "sees" your hand | Google's pretrained gesture model | Google's pretrained hand key points + your classifier | Raw image pixels, cropped to a box |
| Robust to lighting/background/skin tone | Yes | Yes, largely | Sensitive |
| Extra dependency | `mediapipe` | `mediapipe` | none |

**Start with the pretrained option** — it works the instant you install
dependencies, no data collection required. It recognizes 7 gestures people
actually use day to day: Open Palm (Hello), Closed Fist, Thumbs Up (Yes),
Thumbs Down (No), Victory/Peace, Pointing Up, and I Love You (an actual ASL
sign). If you want to teach it gestures beyond those 7, add the landmark
pipeline on top — the pretrained script automatically falls back to your
custom model for anything it doesn't recognize.

Important honesty note: this is *not* full sign-language recognition. Real
ASL/BSL/etc. have thousands of signs, many involving motion over time and
facial grammar — recognizing all of that reliably needs a large labeled
video dataset and a much bigger project. What's here is a genuinely useful,
zero-setup base for common gestures, extendable with your own signs.

## Project Structure
```
SignGestureProject/
├── src/
│   ├── landmark_utils.py         # shared helpers for both landmark-based scripts below
│   ├── recognize_pretrained.py   # Zero training: recognize 7 common gestures immediately
│   ├── capture_landmarks.py      # Landmark Step 1: collect samples (for custom gestures)
│   ├── train_landmark_model.py   # Landmark Step 2: train
│   ├── recognize_landmarks.py    # Landmark Step 3: live recognition (custom gestures only)
│   ├── capture_images.py         # Pixel Step 1: collect images
│   ├── train_model.py            # Pixel Step 2: train
│   └── recognize.py              # Pixel Step 3: live recognition
├── models/
│   ├── gesture_model.pkl         # sample pretrained pixel model
│   ├── gesture_recognizer.task   # auto-downloaded on first run (pretrained option)
│   └── hand_landmarker.task      # auto-downloaded on first run (landmark pipeline)
├── gesture_data/                  # created automatically when you capture data (not tracked in git)
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements
- Python 3.9–3.12 (required for `mediapipe`; if you're on the pixel pipeline only, any Python 3.8+ works)
- A webcam
- ~5 minutes for setup

## Installation

```bash
git clone https://github.com/Yajurvendra08/SignGestureProject.git
cd SignGestureProject

# Create a virtual environment
python -m venv venv

# Activate it
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

If `mediapipe` fails to install (e.g. your Python version isn't supported
yet), you can still use the pixel pipeline — just skip that one line, or
remove `mediapipe` from `requirements.txt` before installing.

---

## Option A: Pretrained (zero training) — start here

```bash
python src/recognize_pretrained.py
```
That's it. A window opens with a skeleton overlay on your hand and the
recognized gesture on screen — no capture step, no training step. The
first run downloads a small (~8 MB) pretrained model automatically. Press
**q** to quit.

Once you've also trained a custom model using Option B below, this same
script automatically uses your custom model as a fallback for any gesture
outside the built-in 7.

---

## Option B: Landmark pipeline (train your own signs)

### 1. Capture gesture samples
```bash
python src/capture_landmarks.py --label hello
```
A window opens with a skeleton overlay on your hand. It auto-captures a
sample every 0.1s by default — just show your hand and move it slightly
(rotate, tilt, move closer/further) for about 10 seconds. Press **ESC** to
stop early. The first run downloads a small (~8 MB) pretrained hand model
automatically.

Repeat for every gesture:
```bash
python src/capture_landmarks.py --label thanks
python src/capture_landmarks.py --label yes
python src/capture_landmarks.py --label no
```

Prefer to control each capture yourself? Add `--manual` and press **SPACE**
per sample instead of auto-capturing.

### 2. Train the model
```bash
python src/train_landmark_model.py
```
Prints an accuracy score and saves `models/gesture_landmark_model.pkl`.

### 3. Recognize gestures live
```bash
python src/recognize_landmarks.py
```
Show a gesture and see the predicted label with a confidence percentage.
Press **q** to quit.

---

## Option C: Pixel pipeline (legacy)

### 1. Capture gesture images
```bash
python src/capture_images.py --label hello
```
Keep your hand inside the green box. Press **SPACE** to save an image, or
add `--auto` to capture continuously on a timer instead:
```bash
python src/capture_images.py --label hello --auto
```
Aim for 100–200 images per gesture. Repeat for each gesture.

### 2. Train the model
```bash
python src/train_model.py
```

### 3. Recognize gestures live
```bash
python src/recognize.py
```

## About the included sample model
`models/gesture_model.pkl` (pixel pipeline) is a small demo model
recognizing four gestures: `hello`, `no`, `thanks`, `yes`. It's included so
you can try `recognize.py` immediately after installing, but for good
accuracy you should capture your own data and retrain — lighting,
background, and hand shape vary a lot between people and rooms. The
landmark pipeline has no bundled model since it needs so little data that
training your own only takes a couple of minutes.

## Tips for better accuracy
- Use good, even lighting on your hand
- Capture samples/images from slightly different angles and distances
- Capture a similar number of samples for each gesture
- Re-run the capture script any time to add more samples — it won't overwrite existing ones

## Troubleshooting
| Problem | Fix |
|---|---|
| `Could not open webcam` | Try `--camera 1` or `--camera 2`; close other apps (Zoom, Teams, etc.) using the camera |
| macOS asks for camera permission | Grant Terminal/your IDE access in System Settings → Privacy & Security → Camera |
| `ModuleNotFoundError` | Make sure your virtual environment is activated, then re-run `pip install -r requirements.txt` |
| `mediapipe` won't install | Check your Python version is 3.9–3.12, or use the pixel pipeline instead |
| Model download fails (landmark pipeline) | Check your internet/firewall, or manually download the URL printed in the error and save it to `models/hand_landmarker.task` |
| Predictions are inaccurate | Capture more samples per gesture and retrain; check lighting/background |

## How it works
**Pretrained option:** Google's MediaPipe Gesture Recognizer is a model
already trained end-to-end (hand detection + gesture classification) on a
large, diverse set of real people performing 7 common gestures. It runs
directly with no extra steps.

**Landmark pipeline:** MediaPipe's pretrained Hand Landmarker locates 21
key points on the hand per frame. Those points are shifted so the wrist is
the origin and scaled by hand size, making the resulting feature vector
independent of where your hand is in the frame or how close it is to the
camera. A Random Forest classifier learns to tell gestures apart from
these shape features.

**Pixel pipeline:** Each captured frame is cropped to a fixed region,
converted to grayscale, resized to 100×100 pixels, and flattened into a
feature vector. A linear Support Vector Machine (SVM) classifies new
frames based on raw pixel values.

## Contributing
Issues and pull requests are welcome.

## License
Released under the MIT License — see [LICENSE](LICENSE) for details.
