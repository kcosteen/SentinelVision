# 🛡️ SentinelVision

**An AI proctoring system — real-time online-exam monitoring that watches a webcam feed and flags suspicious behavior using computer vision.**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-00A67E)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-7C3AED)

SentinelVision analyzes a live camera stream frame-by-frame to detect the behaviors that matter during a remote exam — a candidate leaving the frame, a second person appearing, eyes drifting off-screen, or a phone coming into view — and turns those signals into a running **suspicion score** with a **risk status** (`Normal → Suspicious → High Risk`). Every flagged event is timestamped and written to a CSV audit log.

> Built as a portfolio project to explore practical computer vision end-to-end: face detection, facial-landmark analysis, iris-based gaze estimation, head-pose geometry, and real-time object detection — wired together into a single decision-making pipeline.

---

## 📽️ Demo

> _Add a short screen-recording GIF here (e.g. `assets/demo.gif`) showing the live score and alerts reacting to a phone / looking away._

---

## ✨ Key Features

| Feature | What it does | Tech |
|---|---|---|
| **Face presence & count** | Flags `No face detected` (candidate left) and `Multiple people detected` (someone else in frame) | MediaPipe Face Detection |
| **Looking-away detection** | Head-pose yaw past a **calibrated 30°** flags `Looking away`; iris gaze is the fallback when no pose is solved | MediaPipe Face Mesh + `cv2.solvePnP` |
| **Object / phone detection** | Detects phones, books, laptops and headphones in real time | **Fine-tuned YOLOv8n** (F1 0.923 vs 0.193 stock) |
| **Behavior scoring engine** | Converts raw detections into a weighted suspicion score, with a per-event cooldown so one event isn't double-counted | Custom rules engine |
| **Audit logging** | Timestamps every event and object detection to `logs/*.csv` | Python `csv` |
| **Live HUD** | Draws the current score, risk status, and active alerts onto the video feed | OpenCV |

---

## 🧠 How It Works

Each webcam frame is passed through three independent detectors in parallel. Their outputs feed a central **behavior analyzer** that maintains the score and decides the current risk status.

```mermaid
flowchart LR
    A[📷 Webcam Frame] --> B[Face Detection<br/>MediaPipe]
    A --> C[Gaze Estimation<br/>Iris Landmarks]
    A --> D[Object Detection<br/>YOLOv8]
    B --> E{{Behavior Analyzer}}
    C --> E
    D --> E
    E --> F[🎯 Suspicion Score<br/>+ Risk Status]
    E --> G[🗒️ Event Log CSV]
```

---

## 📊 Detection Logic

The analyzer assigns points to each flagged event and keeps a cumulative score:

| Event | Points |
|---|---:|
| Looking away | +10 |
| No face detected | +20 |
| Multiple people detected | +40 |
| Phone detected | +50 |

A **10-second cooldown** per event type prevents the same continuous behavior from inflating the score every frame.

The cumulative score maps to a risk status:

| Score | Status |
|---|---|
| `0 – 29` | 🟢 Normal |
| `30 – 69` | 🟡 Suspicious |
| `70+` | 🔴 High Risk |

---

## 📈 Measured Results

Nothing below is an estimate — each number comes from a script in this repo, run
against held-out data.

### Fine-tuning the phone detector

The stock COCO YOLOv8n is poor at exam-webcam phone detection, so it was
fine-tuned on a 25,173-frame online-proctoring dataset (CC BY 4.0). Both models
were scored by the **same** from-scratch AP/F1 code on the same held-out images,
because the two number their classes differently and each model's self-reported
metric isn't comparable:

| Model | Precision | Recall | **F1** |
|---|---:|---:|---:|
| Pre-trained YOLOv8n (COCO) | 0.215 | 0.176 | **0.193** |
| **Fine-tuned** (this project) | 0.909 | 0.936 | **0.923** |

The val split is regrouped **by source video** (`split_by_source.py`) — the
export's own split put ~87% of frames from one video and drew 100% of val from
it, so validating on it would have scored the model on the same person in the
same room it trained on.

### Thresholds: measured, not guessed

Every decision boundary lives in [`src/thresholds.py`](src/thresholds.py), each
tagged CALIBRATED (with the command that produced it) or UNCALIBRATED (someone
picked it). Two were measured:

| Threshold | Value | How it was measured | Result |
|---|---:|---|---|
| Phone confidence | `0.35` | Swept 0.05–0.95 over 700 held-out proctoring images | F1 **0.923**; flat 0.25–0.50, so a plateau not a knife-edge |
| Head yaw "looking away" | `30°` | Swept against 463 Gourier head-pose images whose filenames encode the true pan angle | F1 **0.869** (P 0.835 / R 0.905) |

Because the yaw threshold is the measured one, it is what the analyzer actually
decides on; the hand-picked iris-gaze cut-points are only a fallback for frames
where no head pose could be solved.

### Honest limitations

Measuring your own model's failures is the point of the exercise, so:

- **The detector's precision does not transfer to my camera.** On 741 phone-free
  frames of my own webcam clips it still fires `cell phone` on **56.4%** of them.
  Every false positive is the same wall shelf behind me. Public-set precision of
  0.909 is a statement about the public set, not about my room — a textbook
  domain shift. Fixing it needs hard negatives from my own environment.
- **Phones at the frame edge are missed.** Held low and half out of shot, the
  detector scores them 0.09–0.16, below any usable threshold.
- **The head-yaw rule is conservative by construction.** Zero false positives in
  624 frames of normal footage, but it was calibrated against "true pan ≥ 45°",
  so subtle turns (14–24°) fall under the cut, and looking *down* is invisible
  to a yaw-only rule.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV, MediaPipe (Face Detection, Face Mesh + iris)
- **Object Detection:** Ultralytics YOLOv8 — `yolov8n` fine-tuned on a proctoring dataset
- **Math / Geometry:** NumPy, `cv2.solvePnP` (head-pose estimation), Eye Aspect Ratio (EAR)
- **Logging:** CSV audit trail

---

## 📁 Project Structure

```
SentinelVision/
├── main.py                          # Entry point — runs the live proctoring loop
├── ROADMAP.md                       # Project plan & milestones
├── requirements.txt
├── requirements-dev.txt             # Test dependencies (pytest)
├── yolov8n.pt                       # Stock YOLOv8 weights (baseline / fallback)
├── docs/
│   ├── DESIGN_NOTES.md              # Architecture, tradeoffs, and glossary
│   ├── DATASETS.md                  # Phase 2 dataset survey, licences, baseline
│   └── ANNOTATION_GUIDE.md          # Labelling protocol for the Phase 2 frames
├── tests/                           # Unit tests (pytest)
├── evaluation/                      # Metrics harness (precision / recall / F1)
├── logs/
│   ├── events.csv                   # Flagged events + running score
│   └── detections.csv               # Raw object detections
└── src/
    ├── vision/
    │   ├── face_detection.py         # Face presence & count
    │   ├── gaze_detection.py         # Gaze direction from face mesh
    │   ├── blink_detection.py        # Blink detection (EAR)        ── built, integration pending
    │   └── head_pose_detection.py    # Head orientation (solvePnP)
    ├── features/
    │   ├── gaze_estimation.py        # Iris-center → gaze ratio
    │   ├── eye_analysis.py           # Eye Aspect Ratio math
    │   └── head_pose.py              # 3D→2D head-pose solver
    ├── object_detection/
    │   └── object_tracker.py         # YOLOv8 inference + logging
    ├── data/                         # Phase 1 — features → windowed training table
    │   ├── feature_extractor.py      # One frame → a row of features
    │   ├── record_session.py         # Log a webcam/video session to CSV
    │   ├── label_clips.py            # Batch-label clips/ from their filenames
    │   └── build_dataset.py          # Frames → sliding time windows
    ├── models/
    │   └── train.py                  # Phase 1 behavior classifier (grouped CV)
    ├── detection/                    # Phase 2 — training a real phone detector
    │   ├── sources.py                # Public-dataset registry + licences
    │   ├── roboflow_import.py        # Fetch Roboflow sets via ROBOFLOW_API_KEY
    │   ├── extract_frames.py         # Mine our clips for the baseline's misses
    │   ├── prepare_dataset.py        # COCO→YOLO, merge sources, clip-grouped split
    │   ├── detection_metrics.py      # IoU / AP@0.5 from scratch
    │   └── train_detector.py         # Fine-tune + compare against the baseline
    ├── behavior/
    │   └── proctor_analyzer.py       # Scoring + risk-status engine
    └── utils/
        ├── event_logger.py           # CSV event logging
        └── geometry.py               # Euclidean distance helper
```

---

## 🚀 Getting Started

**Prerequisites:** Python 3.9+ and a working webcam.

```bash
# 1. Clone the repository
git clone https://github.com/kcosteen/SentinelVision.git
cd SentinelVision

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

The webcam window opens with the live score and alerts. **Press `q` to quit.**
The YOLOv8 weights (`yolov8n.pt`) are already included, so no extra download is needed.

---

## 🧪 Testing & Evaluation

The project is measured, not just eyeballed:

```bash
pip install -r requirements-dev.txt

# Run the unit-test suite (geometry, EAR, gaze, scoring engine)
pytest

# Evaluate detection quality — precision / recall / F1 per behavior
python -m evaluation.evaluate
```

Detection quality is scored against human-labeled clips using
**precision / recall / F1** (see [`evaluation/`](evaluation/README.md) for how to
build a labeled test set). Design decisions and tradeoffs are documented in
[`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md).

---

## 🗺️ Roadmap

> Full plan with skills mapped to each phase: [`ROADMAP.md`](ROADMAP.md)

- [x] Real-time face detection & presence checks
- [x] Iris-based gaze estimation (Left / Right / Center)
- [x] YOLOv8 object & phone detection
- [x] Behavior scoring engine with risk status + event logging
- [x] Unit tests + evaluation harness (precision / recall / F1)
- [ ] Replace the rules engine with a trained temporal behavior model
- [x] Fine-tune the phone detector on a real proctoring dataset & report the delta
- [x] Integrate the calibrated head-pose signal into the live score
- [ ] Integrate blink-rate into the live score _(module built; EAR threshold still uncalibrated)_
- [ ] End-of-session summary report
- [ ] Save alert snapshots alongside the CSV log
- [ ] Configurable thresholds & multi-face identity tracking

---

## 📌 Note

This is an **educational / portfolio project** demonstrating computer-vision techniques, not production surveillance software. It runs entirely locally and stores no video — only timestamped event logs. Any real deployment would require explicit consent, privacy safeguards, and far more rigorous validation.
