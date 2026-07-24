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

- [ ] Fine-tune YOLOv8 on a custom exam dataset (phones, earbuds, notes, faces)
      annotated in Roboflow / CVAT
- [ ] Report mAP@0.5 vs. the pre-trained baseline

**Skills shown:** annotation, augmentation, transfer learning, mAP evaluation.

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
