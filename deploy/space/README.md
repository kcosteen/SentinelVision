---
title: SentinelVision — Exam Proctor Demo
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
short_description: Upload an exam clip; a fine-tuned YOLOv8n flags phones and looking away
models:
  - kcosteen/sentinelvision-proctoring-yolov8n
---

# SentinelVision — exam proctoring demo

Upload a short webcam clip. The app measures gaze, head pose and objects on every
frame with a **fine-tuned YOLOv8n**, then flags suspicious behaviour over time
using thresholds calibrated against public labelled data.

| | Precision | Recall | **F1** |
|---|---:|---:|---:|
| Pre-trained YOLOv8n (COCO) | 0.215 | 0.176 | **0.193** |
| **Fine-tuned (this model)** | 0.909 | 0.936 | **0.923** |

- **Model:** [kcosteen/sentinelvision-proctoring-yolov8n](https://huggingface.co/kcosteen/sentinelvision-proctoring-yolov8n)
- **Code:** [github.com/kcosteen/SentinelVision](https://github.com/kcosteen/SentinelVision)

## Worth knowing before you trust it

That 0.909 precision was measured on the public validation set, **not on your
room**. On an unfamiliar camera this detector fired `cell phone` on 56.4% of 741
phone-free frames — every false positive the same wall shelf, scoring 0.60–0.71
against a real phone's 0.72–0.79. Those ranges touch, so no confidence threshold
separates them.

The live app handles this with a static-region filter that stops reporting
detections in parts of the frame that never move. This upload demo scores whole
clips instead, so **expect false positives from background clutter** — cluttered
backgrounds are exactly the failure mode, and hiding it here would misrepresent
the model.

This is an educational / portfolio project, not production surveillance software,
and it is **not suitable for making decisions about real people**.
