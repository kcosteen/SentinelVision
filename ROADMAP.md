# Roadmap

The goal of this project is to grow from *wiring together pre-trained models*
into *training, evaluating, and shipping models* — the skills an AI/ML engineer
role screens for. Each phase adds a concrete, resume-worthy capability.

---

### ✅ Phase 0 — Make it credible (done)

Foundations that let every later phase be measured instead of guessed.

- [x] Unit tests for the pure logic (`geometry`, EAR, gaze ratio, scoring engine)
- [x] Evaluation harness with real metrics — precision / recall / F1 per behavior
- [x] Project packaging (`pyproject.toml`) + dev dependencies

**Skills shown:** evaluation, metrics, testing, reproducibility.

### 🔜 Phase 1 — Learn the behavior model (flagship)

Replace the hand-coded rules in `proctor_analyzer.py` with a *trained* model.

- [x] Log per-frame features to CSV (`src/data/`) — gaze, head pose, blink,
      face count, objects
- [ ] Record + label sessions as normal / suspicious segments -> a real dataset
- [ ] Aggregate frames into sliding time windows (blink rate, gaze variance, ...)
- [ ] Train a classifier: start simple (logistic regression / gradient boosting),
      then a temporal model (LSTM / GRU / 1D-CNN)
- [ ] Evaluate with train/val/test split, ROC-AUC, precision-recall, calibration

**Skills shown:** feature engineering, dataset creation, sequence modeling,
the full train -> evaluate loop.

### 🔜 Phase 2 — Train a detector

Motivated by a measured failure, not a hunch: pre-trained YOLOv8n finds the phone
in only **20.7%** of frames from our own `phone_*` clips, which is why the Phase 1
`phone` model learned head-down *posture* as a proxy instead.

- [x] Survey public datasets, with licences and caveats — [`docs/DATASETS.md`](docs/DATASETS.md),
      registry in `src/detection/sources.py`
- [x] Roboflow Universe importer keyed off `ROBOFLOW_API_KEY`, remapping their
      class lists onto ours and dropping out-of-scope classes
- [x] Dataset pipeline: COCO→YOLO conversion, merge public + our own frames,
      clip-grouped split (`src/detection/prepare_dataset.py`)
- [x] Targeted frame extraction — keep the frames the baseline *missed*, so
      annotation effort goes to the blind spot (`src/detection/extract_frames.py`)
- [x] AP@0.5 / IoU implemented from scratch + unit tested, so the baseline
      comparison is apples-to-apples across differing class ids
- [x] Baseline measured: **AP@0.5 0.432, precision 0.761, recall 0.406**
- [x] **Fine-tuned and reported the delta:** F1 **0.193 → 0.923** on held-out
      proctoring images, both models scored by the same from-scratch code
- [x] Threshold calibrated from the sweep rather than guessed (conf **0.35**),
      and the calibrated head-yaw rule (**30°**, F1 0.869) wired into the live
      looking-away decision — replacing the hand-picked gaze cut-points
- [x] Measured the fine-tune on our OWN footage and reported the bad news:
      **56.4%** false-positive rate on 741 phone-free frames, all of them one
      piece of background furniture. Public precision did not transfer.
- [ ] _Deferred:_ annotate own frames as hard negatives to close that gap —
      see [`docs/ANNOTATION_GUIDE.md`](docs/ANNOTATION_GUIDE.md)

**Skills shown:** transfer learning, mAP/F1 evaluation, threshold calibration,
leak-free splitting, and diagnosing domain shift in your own model.

### 🔜 Phase 3 — Ship it

- [ ] FastAPI inference service, containerized with Docker
- [ ] Streamlit / Gradio demo (upload a clip -> get a report) — a live link
- [ ] Experiment tracking with Weights & Biases or MLflow; model + dataset cards

**Skills shown:** serving, containerization, experiment tracking, MLOps.

### 🔭 Phase 4 — Differentiate

- [ ] Real-time optimization (ONNX / quantization) with FPS-vs-accuracy analysis
- [ ] LLM-generated natural-language incident summaries from the event log

---

See [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) for the reasoning and tradeoffs
behind the current design.
