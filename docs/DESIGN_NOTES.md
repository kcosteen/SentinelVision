# Design Notes

Engineering rationale behind SentinelVision — *why* each piece works the way it
does, what the tradeoffs are, and where the honest limitations lie. If you can
explain everything in this document, you can defend the project in an interview.

---

## 1. System architecture

Each webcam frame is processed by three **independent** detectors, whose outputs
feed one **behavior analyzer** that maintains a running suspicion score.

```
frame ─┬─► face detection   (how many faces?)         ─┐
       ├─► face mesh        (gaze · EAR · head pose)  ─┤─► analyzer ─► score + status
       └─► object detection (phone? book?)            ─┘      │
                    │                                          └─► CSV event log
                    └─► static-region filter (is that furniture?)
```

**Why this shape?** Each detector answers one narrow question and knows nothing
about the others, which makes them independently testable and replaceable — the
detector was swapped for a fine-tuned one without touching gaze. The analyzer is
the only place signals combine into a decision, so all the *policy* lives in one
file and every threshold lives in `src/thresholds.py`.

**One rule worth stating explicitly:** the frame the models *measure* is never
the frame we *draw on*. `main.py` keeps a pristine `source` copy; all rendering
happens in `src/vision/hud.py`, which only ever receives the display frame. This
began as a bug — face-detection overlays were being painted onto the array YOLO
then read, and those ~3,000 altered pixels dropped a `person` detection from 0.63
to nothing. Splitting measurement from rendering makes it structurally impossible
rather than merely fixed.

**Tradeoff:** three models per frame costs CPU. We use the smallest variants
(MediaPipe lightweight, YOLOv8-**nano**) to stay real-time, accepting lower
accuracy than larger backbones would give.

---

## 2. Face detection — MediaPipe (BlazeFace)

Counts faces to catch **nobody present** (candidate walked off) and **multiple
people** (someone helping).

- A lightweight SSD-style CNN built for real-time on-device use.
- **Why not train our own?** Face detection is solved and commoditized; training
  a worse version would be wasted effort. Knowing *when to reuse* a strong
  pre-trained model is itself an engineering skill.

---

## 3. Gaze estimation — geometric, iris-ratio method

Where the iris sits between the two eye corners:

```
ratio = (iris_x − left_corner_x) / (right_corner_x − left_corner_x)
```

`< 0.35 → LEFT`, `> 0.65 → RIGHT`, else `CENTER`.

- **Why geometric instead of a trained gaze model?** No training data needed and
  fully interpretable — every decision traces to a number you can print.
- **This is the one threshold still UNCALIBRATED.** No public gaze dataset with
  usable ground truth turned up, so `0.35/0.65` remain hand-picked. That is
  recorded in `src/thresholds.py` rather than quietly presented as measured.
- **Consequence for the design:** because it is unverified, off-centre gaze
  raises its own weaker event (`Gaze off screen`, +5) instead of being folded
  into `Looking away` (+10, head-yaw based, calibrated). Weights track how much
  each signal is *trusted*, not only how bad the behaviour is. Collapsing them
  would launder a guess into looking like a measurement.

It still earns its place: head yaw only sees a turned *head*, so glancing at
notes beside the screen is invisible without it.

---

## 4. Head pose — the Perspective-n-Point (PnP) problem

`head_pose.py` matches six 2D facial landmarks to a generic 3D face model and
solves for rotation with `cv2.solvePnP`, then converts to Euler angles.

- This is **classical geometry, not machine learning** — worth saying so you
  don't overclaim "AI". There is nothing to train here, only a threshold to
  justify.
- **So we measured it.** `validate_head_pose.py` runs the real solver over 463
  Gourier head-pose images whose filenames encode the true pan angle:
  - **Magnitude works.** `|yaw|` tracks true `|pan|` closely (Spearman ρ 0.89),
    near-linear to 60° then saturating as landmarks occlude.
  - **Direction is broken.** Signed yaw sits near zero at every true pan — the
    known `cv2.RQDecomp3x3` sign flip. Fine for *"is the head turned away?"*,
    useless for *"which way?"*. The code uses `abs(yaw)` and says why.
  - Calibrated cut-point: **`|yaw| ≥ 30°`, F1 0.869** (P 0.835 / R 0.905).
- **This is now the primary looking-away signal** in the live pipeline — the
  measured one drives the decision, and gaze is only its fallback for frames
  where no pose solves.
- **Limitation:** a generic 3D face and a rough camera model, so absolute angles
  are approximate. 30° was calibrated against *"true pan ≥ 45°"*, making it a
  large-turn detector by construction: subtler turns fall under it, and looking
  *down* is invisible to a yaw-only rule.

---

## 5. Blink detection — Eye Aspect Ratio (EAR)

EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2·‖p1−p4‖) — eye height over width. Open eyes give a
high EAR; a blink collapses it toward zero. A classic cheap heuristic
(Soukupová & Čech, 2016).

**Calibrated**, on 1,999 labelled open/closed eye images:

| class | n | mean EAR | |
|---|---:|---:|---|
| closed | 1199 | 0.080 | p90 **0.164** |
| open | 800 | 0.360 | p10 **0.287** |

Cohen's d **4.11** — well separated, so EAR is a genuinely strong signal (unlike
the gaze ratio, which has no ground truth to check against at all).

**We adopted `0.23`, not the F1 argmax of `0.25`.** F1 peaks at 0.25 but
precision falls off a cliff immediately above it (131 false positives by 0.30),
while the two class distributions leave a wide empty gap from ~0.16 to ~0.29.
`0.23` gives up 0.003 F1 to sit mid-gap, so a modest shift from a different
camera doesn't cross it. **Picking the argmax of a curve measured on someone
else's capture setup is exactly the mistake documented in §7.**

---

## 6. Object detection — a fine-tuned YOLOv8n

Stock COCO YOLOv8n is poor at this: exam-webcam phones are small, dim,
hand-occluded and shot through cheap optics, nothing like COCO's clean centred
phones. Measured on 1,822 held-out proctoring images it reaches **F1 0.203**, and no
confidence threshold rescues it — that is a *model* problem, not a tuning one,
which is what justified fine-tuning.

| Model | Precision | Recall | **F1** |
|---|---:|---:|---:|
| Pre-trained YOLOv8n (COCO) | 0.215 | 0.192 | **0.203** |
| **Fine-tuned** | 0.910 | 0.944 | **0.927** |

- **Single-stage detector:** YOLO predicts boxes and classes in one pass, which
  is what makes it fast enough for live video (vs. two-stage detectors like
  Faster R-CNN — more accurate, slower).
- **Both models scored by the same from-scratch AP code.** The baseline emits
  COCO class 67, the fine-tune class 1; each model's self-reported metric isn't
  comparable to the other's. One scoring implementation removes that doubt.
- **The split was rebuilt.** The dataset's own split drew **100% of validation
  from a single source video** that was 87% of the data — validating on the same
  person in the same room it trained on. `split_by_source.py` regroups so
  validation is people the model never saw. **This is the difference between a
  number and a number that means something.**
- **Never hard-code a class id.** `cell phone` is 67 in COCO, 1 in the fine-tune,
  0 in a single-class set. A wrong id matches nothing and looks like a detector
  that found nothing rather than a lookup that was wrong — a silent failure.
  `class_ids.py` resolves by name for every model.

**Two confidence values, deliberately:** `0.35` is the F1 optimum on the public
set and stays the default for measurement, so published numbers remain
comparable. The live app uses `0.60`, an *operating point* for one room — see §7.

---

## 7. Domain shift, and fixing it structurally

**The most important result in this project is an unflattering one.** On a camera
the detector had never seen, it fired `cell phone` on **56.4% of 741 phone-free
frames**. Every false positive was the same wall shelf. Live it scored that shelf
**0.60–0.71** against a real phone's **0.72–0.79** — the ranges *touch*, so **no
confidence threshold separates them.** A blank-background control confirmed the
shelf was the sole source.

The public set's 0.910 precision was a statement about the public set, not about
that room. **This is textbook domain shift**, and it is why a single headline
metric is never enough.

Three fixes were tried and rejected **on measurements**, which is the part worth
explaining in an interview:

| Attempt | Killed by |
|---|---|
| Raise the confidence threshold | Ranges overlap — no cut separates them |
| Track boxes by IoU ≥ 0.80 | Matched only **12%** of consecutive frames: box *size* thrashes while its centre holds (8px p90) |
| Track boxes by centre proximity | The shelf emits several competing boxes that split and swap — per-box *association* is itself unsound |
| **Occupancy grid, no association** | ✅ works |

The working version divides the frame into cells and stops reporting detections
in cells continuously occupied for 2 seconds — furniture doesn't move. No
per-room configuration; it learns whatever is nailed down in front of it.

**Two premises had to be corrected**, and both are better lessons than the fix:

- **"A phone in use moves" is false.** A phone being *read* is held still — the
  main cheating case. Anything overlapping the detected person is therefore
  exempt from suppression, however still it is held. A filter that goes blind
  exactly when it matters is worse than no filter.
- **"Not seen lately" is not "gone".** Sitting forward occludes the shelf; a
  short forget-window wiped what had been learned, so it re-flagged on every
  lean-back. Confirmed scenery is now remembered *through* occlusion.

Thresholds here are **wall-clock seconds, not frames** — this pipeline's frame
rate is a property of the machine, so "60 frames" silently means 2s on one
computer and 10s on another.

---

## 8. Behavior analyzer — rules, with the trust made explicit

`proctor_analyzer.py` maps events to points, sums them, and buckets the score.

| Event | Points | Signal quality |
|---|---:|---|
| Gaze off screen | +5 | uncalibrated |
| Looking away | +10 | calibrated, F1 0.869 |
| No face detected | +20 | face count |
| Multiple people detected | +40 | face count |
| Phone detected | +50 | calibrated, F1 0.927 |

- **Cooldown = debouncing.** A 10-second per-event cooldown stops one continuous
  behaviour (looking away for 3s = 90 frames) being counted 90 times. Standard
  technique for turning noisy per-frame signals into stable events.
- **The score decays** 2 points/second while nothing fires. Without decay the
  score only ever climbs, so a *single* false positive pins the session at
  "Suspicious" forever — that isn't measuring current behaviour, it's measuring
  whether anything ever happened. Sustained behaviour still outruns decay
  comfortably (+50 per 10s cooldown against 20 shed).
- **A startup grace period** (3s) suppresses judgement while the static filter
  learns the scene, since during warm-up background clutter *will* be reported.
- **Still the weakest link, honestly:** the weights and the 30/70 bands are
  guesses. They also can't capture *patterns over time* — "glances away every few
  seconds" versus "one long look". A temporal model over these features is the
  natural next step; it is out of scope here because it needs labelled footage
  this project deliberately doesn't collect.

---

## 9. From guessing to learning: tuning vs. training vs. fine-tuning

A favourite interview probe: these are **not** the same thing. This project did
all three, so the distinction is concrete rather than theoretical.

- **Threshold tuning** — choosing magic constants. Sweep candidates against a
  labelled validation set and keep whichever maximises F1. **No weights are
  learned**, but it replaces guessing with evidence. Cheapest upgrade. *Done for
  phone confidence, head yaw, and EAR — three of five thresholds; the other two
  say plainly that they are guesses.*
- **Fine-tuning** — continuing training of an *existing* pre-trained network on
  your own labelled images, so its weights adapt to your domain. It reuses
  everything the model already knows about generic objects and only nudges it
  toward your specialty — far cheaper than training from scratch. *Done for the
  detector: F1 0.203 → 0.927, with the backbone frozen because a dataset this
  size would otherwise mostly memorise.*
- **Training a new model** — learning the parameters of a new model from scratch
  on your data. *Not done here.* An earlier attempt trained a behaviour
  classifier on recorded clips and was removed: its phone class had almost no
  real signal to learn from (the detector of the day found a phone in ~20% of
  frames), so it learned head-down *posture* as a proxy. **Diagnosing that a
  model learned a proxy is more valuable than shipping it.**

**Prerequisite for all but guessing: labelled data**, and the *kind* differs —
threshold tuning needs labelled examples of the measurement; detector
fine-tuning needs bounding boxes.

> Being able to say *"I don't always need to train a model — sometimes tuning on
> a validation set is enough, and sometimes the honest answer is that the model
> learned the wrong thing"* signals real engineering judgment.

---

## 10. How we evaluate

Per-behaviour **precision, recall, F1**, plus **AP@0.5 / IoU implemented from
scratch** for detection (`detection_metrics.py`, unit-tested).

- **Why not accuracy?** Cheating events are *rare*. If 95% of frames are normal,
  a model that always predicts "normal" scores 95% accuracy while catching zero
  cheating. Precision/recall expose that; accuracy hides it.
- **Why implement AP ourselves?** So two models with different class conventions
  can be scored by identical code, and so the comparison doesn't depend on each
  library's self-reported number.
- **Why the split matters more than the metric.** See §6: a leaky split produced
  a flattering number that described one person in one room.

---

## 11. Known limitations (own them honestly)

- **Precision does not transfer to an unseen camera** — 56.4% false positives on
  phone-free frames in an unfamiliar room (§7). Mitigated structurally, but the
  honest fix is hard negatives collected from the deployment environment.
- **Partially visible phones are missed** — held low and half out of frame they
  score 0.09–0.16, below any usable threshold.
- Gaze thresholds and the score bands remain **hand-picked**.
- Head pose gives **magnitude only** — it cannot tell left from right, and a
  yaw-only rule cannot see someone looking straight down.
- Single camera, vision only — no audio, no screen monitoring.
- **Not suitable for decisions about real people.** A false cheating accusation
  is a serious harm; this is a portfolio project, not production software.

---

## 12. Quick glossary (interview review)

| Term | One-line meaning |
|---|---|
| Threshold tuning | Picking cutoff constants — can be guessed or chosen on a validation set (no training). |
| Training | Learning a model's parameters from labeled data. |
| Fine-tuning | Continuing to train a *pre-trained* model on your own data. |
| Frozen backbone | Holding early layers fixed while fine-tuning, so a small dataset can't wreck general features. |
| Domain shift | Test data differs from training data, so metrics don't transfer. |
| Hard negative | A background example the model wrongly fires on — the most valuable thing to label. |
| Data leakage | Train and validation sharing information, inflating the score. |
| Multi-class | Each example gets exactly one of N mutually-exclusive labels. |
| Multi-label | Each example can carry several labels at once. |
| Feature vs. label | A feature is a measured input; a label is the target to predict. |
| Proxy signal | A model predicting from something correlated with, but not, the real cause. |
| Precision | Of my positive predictions, how many were correct? |
| Recall | Of the actual positives, how many did I find? |
| F1 | Harmonic mean of precision and recall. |
| Cohen's d | How many pooled standard deviations apart two class means are. |
| Confusion matrix | TP / FP / FN / TN table all the metrics come from. |
| mAP / AP@0.5 | Mean Average Precision — the standard object-detection score. |
| IoU | Intersection over union of two boxes — sensitive to size, not just position. |
| Confidence threshold | Cutoff that trades precision against recall. |
| Operating point | A threshold chosen for deployment conditions rather than for a benchmark. |
| Single-stage detector | Predicts boxes + classes in one pass (YOLO) — fast. |
| PnP (solvePnP) | Recover 3D pose from 2D↔3D point correspondences. |
| EAR | Eye Aspect Ratio — eye height/width, used to detect blinks. |
| Debouncing / cooldown | Collapsing repeated noisy signals into one event. |
| Class imbalance | When one label dominates, making accuracy misleading. |
