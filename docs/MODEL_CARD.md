---
license: agpl-3.0
tags:
  - object-detection
  - yolov8
  - ultralytics
  - proctoring
  - computer-vision
library_name: ultralytics
pipeline_tag: object-detection
base_model: Ultralytics/YOLOv8
---

# SentinelVision — YOLOv8n fine-tuned for exam proctoring

A YOLOv8-nano detector fine-tuned to spot the objects that matter in an online
exam: phones, books, headphones, laptops, TVs and people.

Used by [SentinelVision](https://github.com/kcosteen/SentinelVision), a real-time
webcam proctoring app.

## Why it exists

Stock COCO YOLOv8n is poor at this task. Phones in exam footage are small, dim,
hand-occluded and shot through a cheap webcam — nothing like the clean, centred
phones in COCO. Measured on held-out proctoring images, the pre-trained model
reaches an F1 of **0.193**, and no confidence threshold rescues it. That is a
model problem, not a tuning problem, which is what motivated fine-tuning.

## Results

Both models were scored by the **same** from-scratch AP/F1 implementation on the
**same** held-out images. The two number their classes differently (COCO class 67
vs class 1 here), so neither model's self-reported metric is comparable to the
other's.

| Model | Precision | Recall | **F1** |
|---|---:|---:|---:|
| Pre-trained YOLOv8n (COCO) | 0.215 | 0.176 | **0.193** |
| **This model** | 0.909 | 0.936 | **0.923** |

Measured at confidence **0.35**, chosen by sweeping 0.05–0.95 against 700
held-out images. The F1 curve is flat from 0.25–0.50, so that is a plateau rather
than a knife edge.

## Classes

```
0 book   1 cell phone   2 headphone   3 laptop   4 person   5 tv
```

Resolve classes **by name, not by index**. This model's `cell phone` is class 1;
in COCO it is 67. A hard-coded index fails silently — it matches nothing and
looks like a detector that found nothing, rather than a lookup that was wrong.

## Training data

[Online-proctoring-system](https://universe.roboflow.com/) via Roboflow Universe —
25,173 webcam-framed exam images, **CC BY 4.0**, 6 classes.

The dataset's own split is unusable: one source video is ~87% of the data **and**
supplies 100% of validation and test. Training on it and validating on frames of
the same person in the same room reports a number that says nothing about anyone
else. The split was regrouped **by source video**, so validation is people the
model never trained on.

Fine-tuned on a Kaggle GPU from `yolov8n.pt`, with the backbone frozen — with a
dataset this size, updating all layers mostly memorises, and the early layers
already encode generic edges and textures worth keeping.

## Intended use

Educational and research work on automated proctoring, and as a worked example of
fine-tuning a detector for a specific visual domain.

**Not suitable for making decisions about real people.** See the limitations
below: on an unfamiliar camera this model raises frequent false alarms, and a
false accusation of cheating is a serious harm. Any real deployment would need
explicit consent, a human in the loop, and validation in the actual deployment
environment.

## Limitations — measured, not guessed

**Precision does not transfer to an unseen camera.** On 741 phone-free frames
from a webcam the model had never seen, it fired `cell phone` on **56.4%** of
them. Every false positive was the same object: a wall shelf holding a keyboard.
Live it scored that shelf **0.60–0.71**, against a real phone at **0.72–0.79** —
the ranges *touch*, so no confidence threshold separates them. A blank-background
control confirmed the shelf was the sole source.

The 0.909 precision above is a statement about the public validation set, not
about any particular room. This is textbook domain shift, and it is the single
most important thing to know before using these weights.

**Partially visible phones are missed.** Held low and half out of frame, phones
score only 0.09–0.16 — below any usable threshold.

**Mitigation used downstream.** SentinelVision does not solve this with a
threshold. It suppresses detections in regions of the frame that never move,
since a shelf is bolted to the wall and a phone in use is not — with an exemption
for anything overlapping the detected person, because a phone being *read* is
held still. The honest fix remains hard negatives collected from the deployment
environment.

## Usage

```python
from ultralytics import YOLO

model = YOLO("proctoring_yolov8n_best.pt")

# Look the class up by NAME -- never hard-code the index.
phone = next(i for i, n in model.names.items() if n == "cell phone")

results = model("frame.jpg", classes=[phone], conf=0.35)
```

Use `conf=0.35` to reproduce the reported metrics. In a live room a higher floor
(~0.60) plus static-region suppression is what makes it usable — see the
repository.

## Licence

**AGPL-3.0**, inherited from Ultralytics YOLOv8, which these weights are derived
from. If you use this in a network service, AGPL obligations apply.

The training data is **CC BY 4.0** and requires attribution to the *Online
Proctoring System* dataset (online-exam-cheating-detection workspace, Roboflow
Universe).

The thresholds used alongside this model were calibrated against two further
public datasets, credited in full in the
[repository's dataset notes](https://github.com/kcosteen/SentinelVision/blob/main/docs/DATASETS.md):
the Gourier Head Pose Image Database (Gourier, Hall & Crowley, 2004; CC BY 4.0 as
redistributed) and `MichalMlodawski/closed-open-eyes` (ODC-BY).

## Citation

```bibtex
@software{sentinelvision_proctoring_yolov8n,
  title  = {SentinelVision: YOLOv8n fine-tuned for exam proctoring},
  url    = {https://github.com/kcosteen/SentinelVision},
  year   = {2026}
}
```
