# Roadmap

The goal of this project is to grow from *wiring together pre-trained models*
into *training, evaluating, and shipping models* — the skills an AI/ML engineer
role screens for. Each phase adds a concrete, resume-worthy capability.

---

### ✅ Phase 0 — Make it credible

Foundations that let every later phase be measured instead of guessed.

- [x] Unit tests for the pure logic (geometry, EAR, gaze ratio, scoring engine)
- [x] Evaluation harness with real metrics — precision / recall / F1 per behaviour
- [x] Project packaging (`pyproject.toml`) + dev dependencies

**Skills shown:** evaluation, metrics, testing, reproducibility.

---

### ⚰️ Phase 1 — Learn the behaviour model *(attempted, then retired)*

The original plan was to replace the hand-coded rules in `proctor_analyzer.py`
with a temporal classifier trained on recorded clips. It was built end-to-end —
feature logging, sliding windows, clip-grouped splits, a RandomForest with
GroupKFold — and then **removed**, deliberately.

**Why it was retired, which is the useful part:**

- Its `phone` class ranked the real phone features (`phone_frac`, `phone_conf_*`)
  **dead last** in importance and predicted from `head_pitch` instead. It had
  learned head-down *posture* as a proxy, because the detector of the day found a
  phone in only ~20% of frames — there was almost no real signal to learn from.
- The fix was therefore not a better classifier but a better detector → Phase 2.
- The training set was one person in one room. Any metric from it described that
  person, not the problem.

**Skills shown:** feature engineering, grouped cross-validation, and — more
importantly — reading feature importances instead of trusting an F1 score, then
deleting work that doesn't hold up. Code removed; a repo shouldn't ship a
pipeline that can't run.

---

### ✅ Phase 2 — Train a detector

Motivated by a measured failure, not a hunch.

- [x] Survey public datasets with licences and caveats — [`docs/DATASETS.md`](docs/DATASETS.md),
      registry in `src/detection/sources.py`
- [x] Roboflow Universe importer keyed off `ROBOFLOW_API_KEY`, remapping their
      class lists onto ours and dropping out-of-scope classes
- [x] Dataset pipeline: COCO→YOLO conversion, class remapping, grouped splits
- [x] Leak-free splitting by source video (`src/detection/split_by_source.py`) —
      the export's own split drew 100% of val from one video that was 87% of the data
- [x] AP@0.5 / IoU implemented from scratch + unit tested, so both models are
      scored apples-to-apples across differing class conventions
- [x] **Fine-tuned and reported the delta:** F1 **0.203 → 0.927** on 1,822
      held-out proctoring images
- [x] Thresholds calibrated rather than guessed: phone conf **0.35** (F1 0.927),
      head yaw **30°** (F1 0.869), EAR **0.23** (F1 0.979) — with the two that
      remain guesses labelled as such in `src/thresholds.py`
- [x] Measured the fine-tune on an unseen camera and reported the bad news:
      **56.4%** false positives on 741 phone-free frames, all one piece of
      furniture. Public precision did not transfer.
- [x] Fixed it structurally: an occupancy-grid static-region filter
      (`src/object_detection/static_filter.py`), after three designs were
      rejected on measurements — see the README for that arc
- [x] Published to the Hub with a model card that leads with the limitations —
      [kcosteen/sentinelvision-proctoring-yolov8n](https://huggingface.co/kcosteen/sentinelvision-proctoring-yolov8n)
- [ ] _Deferred:_ hard negatives from the deployment environment, which remains
      the honest fix rather than a filter

**Skills shown:** transfer learning, mAP/F1 evaluation, threshold calibration,
leak-free splitting, licence diligence, and diagnosing domain shift in your own
model.

---

### 🔜 Phase 3 — Ship it

- [x] Model card + dataset documentation, published alongside the weights
- [x] Weights fetched from the Hub at runtime, so a fresh clone gets the real
      detector instead of silently falling back to COCO
- [x] Streamlit demo (upload a clip → get a report), runnable locally
- [ ] Host the demo for a clickable link — Streamlit Community Cloud
      (Hugging Face Spaces needs billing on file; the container is ready in
      `deploy/space/` if that changes)
- [ ] FastAPI inference service, containerized
- [ ] Experiment tracking (Weights & Biases / MLflow) for the fine-tuning runs

**Skills shown:** serving, containerization, experiment tracking, MLOps.

---

### 🔭 Phase 4 — Differentiate

- [ ] Real-time optimization (ONNX / quantization) with an FPS-vs-accuracy analysis
- [ ] Temporal modelling over the signal stream — the honest version of Phase 1,
      once there is footage of more than one person to train it on
- [ ] LLM-generated natural-language incident summaries from the event log

---

See [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) for the reasoning and tradeoffs
behind the current design.
