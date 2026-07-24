# Design Notes

Engineering rationale behind SentinelVision — *why* each piece works the way it
does, what the tradeoffs are, and where the honest limitations lie. If you can
explain everything in this document, you can defend the project in an interview.

---

## 1. System architecture

Each webcam frame is processed by three **independent** detectors, whose outputs
feed one **behavior analyzer** that maintains a running suspicion score.

```
frame ─┬─► face detection  (how many faces?)          ─┐
       ├─► gaze estimation  (looking where?)           ─┤─► behavior analyzer ─► score + status
       └─► object detection (phone / book present?)    ─┘        │
                                                                 └─► CSV event log
```

**Why this shape?** The detectors are decoupled — each answers one narrow
question and knows nothing about the others. That makes them independently
testable and independently replaceable (Phase 2 can swap the object detector
without touching gaze). The analyzer is the only place that combines signals
into a decision, so all the "policy" lives in one file.

**Tradeoff:** running three models per frame costs CPU/GPU time. We use the
*smallest* variants (MediaPipe's lightweight models, YOLOv8-**nano**) to keep it
real-time, accepting lower accuracy than the larger variants would give.

---

## 2. Face detection — MediaPipe (BlazeFace)

Counts faces to catch two situations: **nobody present** (candidate walked off)
and **multiple people** (someone helping).

- MediaPipe's face detector is a lightweight SSD-style CNN built for real-time
  use on-device.
- **Why not train our own?** Face detection is a solved, commoditized problem;
  training a worse version would be wasted effort. Knowing *when to reuse* a
  strong pre-trained model is itself an engineering skill.

---

## 3. Gaze estimation — geometric, iris-ratio method

We take the iris landmarks (from MediaPipe Face Mesh with iris refinement) and
compute where the iris sits **between the two eye corners**:

```
ratio = (iris_x − left_corner_x) / (right_corner_x − left_corner_x)
```

`< 0.35 → LEFT`, `> 0.65 → RIGHT`, else `CENTER`. Sustained non-center gaze is
flagged as "looking away".

- **Why geometric instead of a trained gaze model?** It needs no training data
  and is fully interpretable — every decision traces back to a number you can
  print. Great for a first version.
- **Limitations (be ready for these):** the thresholds are hand-tuned, not
  learned; it only uses horizontal gaze (no up/down); and it doesn't separate
  *eye* movement from *head* movement. A real system would fuse this with head
  pose and ideally learn the thresholds from data. → This is a motivation for
  Phase 1.

---

## 4. Head pose — the Perspective-n-Point (PnP) problem

`head_pose.py` estimates head orientation (pitch/yaw/roll) by matching six 2D
facial landmarks to a generic 3D face model and solving for the camera-to-head
rotation with `cv2.solvePnP`, then converting the rotation vector to Euler angles.

- This is **classical computer vision / geometry**, not machine learning — worth
  saying explicitly so you don't overclaim "AI".
- **Limitation:** it assumes a generic 3D face (no per-person calibration) and a
  rough camera model, so absolute angles are approximate. Relative changes
  (turned left vs. facing forward) are reliable enough to be useful.

*Currently built but not yet wired into the live pipeline — Phase 1 folds it in.*

## 5. Blink detection — Eye Aspect Ratio (EAR)

EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2·‖p1−p4‖) — the ratio of eye height to width.
Open eyes give a high EAR; a blink makes it collapse toward zero.

- A classic, cheap heuristic (Soukupová & Čech, 2016).
- **Limitation:** a single global threshold doesn't generalize across people,
  glasses, or camera distance. Robust blink/drowsiness detection usually learns
  a per-person baseline or trains a small classifier — another Phase 1 hook.

---

## 6. Object detection — YOLOv8 (pre-trained on COCO)

`object_tracker.py` runs YOLOv8-nano and keeps only the classes relevant to
cheating (person, cell phone, book, laptop) with confidence > 0.6.

- **Single-stage detector:** YOLO predicts boxes and classes in one pass, which
  is what makes it fast enough for live video (vs. two-stage detectors like
  Faster R-CNN, which are more accurate but slower).
- **The 0.6 confidence threshold is a precision/recall knob.** Raise it → fewer
  false alarms (higher precision) but more missed objects (lower recall). Lower
  it → the opposite. Choosing this value *is* a modeling decision.
- **Limitation:** COCO's "cell phone" class wasn't trained on exam webcam angles,
  and it can't see earbuds or notes at all. → Phase 2 fine-tunes on a custom
  dataset and reports mAP against this baseline.

---

## 7. Behavior analyzer — rules today, model tomorrow

`proctor_analyzer.py` maps events to points (`Phone detected → +50`, etc.),
sums them into a score, and buckets the score into Normal / Suspicious /
High Risk.

- **Cooldown = debouncing.** A 10-second per-event cooldown stops one continuous
  behavior (e.g. looking away for 3 seconds = 90 frames) from being counted 90
  times. This is a standard trick for turning noisy per-frame signals into
  stable events.
- **Why this is the weakest link (and that's the point):** the weights (20, 40,
  10, 50) and thresholds (30, 70) are guesses, not learned from data. They can't
  capture *patterns over time* ("glances away every few seconds" vs. "one long
  look"). **Replacing this rules engine with a trained temporal model is
  Phase 1 — the flagship ML contribution.**

---

## 8. From guessing to learning: tuning vs. training vs. fine-tuning

A favorite interview probe: these three ways to "improve" a model are **not** the
same thing. Keep them distinct.

- **Threshold tuning** — choosing the magic constants (gaze `0.35/0.65`, the `0.6`
  confidence cutoff, the score weights `20/40/10/50`). Today these are *guessed*.
  With a labeled validation set you can instead *sweep* candidate values and keep
  whichever maximizes F1. This is **not training a model** — no weights are
  learned — but it already replaces guessing with evidence. Cheapest upgrade.
- **Training a (new) model** — learning the parameters of a *new* model from your
  data. **Phase 1**: feed the per-frame features (gaze, head pose, blink, objects)
  plus labels into a classifier so it *learns* how to weigh them, replacing the
  hand-coded rules in `proctor_analyzer.py`.
- **Fine-tuning** — taking an *existing* pre-trained network (YOLOv8, trained on
  COCO) and continuing its training on your own labeled images so its weights
  adapt to your domain (exam webcam angles, earbuds, notes). **Phase 2**. It
  reuses everything the model already knows about generic objects and only nudges
  it toward your specialty — far cheaper than training from scratch.

**Prerequisite for all three (except plain guessing): labeled data.** The *kind*
differs — threshold tuning and behavior training need labeled clips / feature
logs; YOLO fine-tuning needs images with bounding-box annotations. So the honest
order is: *collect + label data → tune thresholds on it (quick win) → train the
behavior model (Phase 1) → fine-tune the detector (Phase 2).*

> Being able to say *"I don't always need to train a model — sometimes tuning on a
> validation set is enough"* signals real engineering judgment.

---

## 9. How we evaluate

We measure per-behavior **precision, recall, and F1** against human-labeled
clips (see `evaluation/`).

- **Why not accuracy?** Cheating events are *rare*. If 95% of frames are
  "normal", a model that predicts "normal" always scores 95% accuracy while
  catching zero cheating. Precision/recall expose that failure; accuracy hides
  it. This class-imbalance point is a very common interview question.

---

## 10. Known limitations (own them honestly)

- No dataset or trained model **yet** — current intelligence is pre-trained
  models + hand-tuned rules (this is what the roadmap fixes).
- Thresholds are hand-picked, not learned.
- Single camera, single modality (vision only); no audio, no screen monitoring.
- Not privacy-hardened or validated for real deployment — it's a portfolio /
  learning project.

---

## 11. Quick glossary (interview review)

| Term | One-line meaning |
|---|---|
| Threshold tuning | Picking cutoff constants — can be guessed or chosen on a validation set (no training). |
| Training | Learning a model's parameters from labeled data. |
| Fine-tuning | Continuing to train a *pre-trained* model on your own data. |
| Multi-class | Each example gets exactly one of N mutually-exclusive labels. |
| Multi-label | Each example can carry several labels at once (looking away *and* on a phone). |
| Feature vs. label | A feature is a measured input; a label is the target to predict. |
| Precision | Of my positive predictions, how many were correct? |
| Recall | Of the actual positives, how many did I find? |
| F1 | Harmonic mean of precision and recall. |
| Confusion matrix | TP / FP / FN / TN table all the metrics come from. |
| mAP | Mean Average Precision — the standard object-detection score. |
| Confidence threshold | Cutoff that trades precision against recall. |
| Single-stage detector | Predicts boxes + classes in one pass (YOLO) — fast. |
| PnP (solvePnP) | Recover 3D pose from 2D↔3D point correspondences. |
| EAR | Eye Aspect Ratio — eye height/width, used to detect blinks. |
| Debouncing / cooldown | Collapsing repeated noisy signals into one event. |
| Class imbalance | When one label dominates, making accuracy misleading. |
