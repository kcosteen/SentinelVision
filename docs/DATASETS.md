# Dataset Survey — Phase 2 (training a phone detector)

What public data exists for exam proctoring, what survived inspection, and what
we decided to train on. Written up because "where did your training data come
from, and what's wrong with it?" is a question every ML interview asks, and
because re-running this search from scratch in six months would be waste.

The machine-readable version of this survey — download paths, licences, the
rejects — lives in [`src/detection/sources.py`](../src/detection/sources.py) and
prints with `python -m src.detection.sources`.

---

## 1. The problem we were shopping for

Phase 1 trained a behaviour model on our own clips and produced a good-looking
`phone` F1 of 0.84. Inspecting the feature importances showed why that number was
a lie: the model ranked the *actual* phone features (`phone_frac`,
`phone_conf_*`) dead last and was really predicting from `head_pitch_mean` and
`gaze_ratio_std` — head-down-at-lap posture. It had learned a proxy, not a phone.

The root cause is measurable: **pre-trained YOLOv8-nano detected the phone in
only 20.7% of sampled frames from webcam clips** — 192 misses out of 242. A
behaviour model can't use a signal that isn't there.

So Phase 2 needs phone images with bounding boxes, ideally shot through a webcam
at a desk.

---

## 2. What we found

### Hugging Face — thin

Searched ~25 query terms across the datasets API (`proctoring`, `exam`,
`cheating`, `phone detection`, `gaze`, `head pose`, `driver distraction`,
`classroom`, …).

**There is no online-proctoring behaviour dataset on Hugging Face.** `proctoring`,
`exam monitoring`, and `cheating detection video` all return zero results. This
is not surprising: proctoring footage is of students' homes and faces, so it is
consent-bound and rarely publishable. Any project in this space is expected to
bring its own data — which is a point in favour of having recorded our own.

One genuinely usable result:

| Dataset | Licence | Size | Verdict |
|---|---|---|---|
| [`harshdadiya-wappnet/phone_detection`](https://huggingface.co/datasets/harshdadiya-wappnet/phone_detection) | Apache-2.0 | 605 images / 589 boxes / 15.7 MB | **Adopted.** Only HF set found with real phone boxes *and* a permissive licence. |

A quirk worth knowing: its COCO file declares two categories, `mobile_phone` and
a typo'd `mobuile_phonw` that carries no annotations. Emitting the empty one as a
class would give the model a head that can never be right, so the converter drops
categories with no boxes.

### Roboflow Universe — where the domain-matched data actually is

Roboflow Universe hosts several purpose-built proctoring sets that are much
closer to our domain than anything on HF:

| Dataset | Contents |
|---|---|
| [Online Proctoring System](https://universe.roboflow.com/online-exam-cheating-detection-kvdul/online-proctoring-system-x27ou-e7abr) | Faces + exam-room objects, shot to simulate remote exams |
| [Cheating (person/phone/calculator)](https://universe.roboflow.com/online-exam-cheating-detection-kvdul/cheating-faalb-jvigx-jxt99) | ~1,798 images with cheating-behaviour boxes |

Fetch either with:

```bash
# PowerShell:  $env:ROBOFLOW_API_KEY = "your_key"
export ROBOFLOW_API_KEY=your_key

python -m src.detection.roboflow_import --list
python -m src.detection.roboflow_import --source roboflow_online_proctoring
```

`prepare_dataset.py` then picks up anything extracted under
`data/detection/external/` automatically — no extra flag.

**The key comes from the environment, never a flag.** Command-line arguments land
in shell history and in the process list where other local users can read them;
an env var does neither and can't be committed. The importer never prints the key,
including in error messages — the URL it logs is built without the query string.

**Version numbers are a guess.** The registry defaults to `v1`; the real version
is on the dataset page. A wrong one gives a clean 404 telling you to check, and
`--version N` overrides it.

**Their classes are not our classes.** These sets label `person`, `face` and
`calculator` alongside phones. The importer remaps the upstream class list onto
ours via `CLASS_ALIASES` and **drops anything unrecognised** — training a head we
never read would only dilute the gradient and the reported mAP. Getting this
backwards would silently turn every `person` box into a `phone`, so it's the
most-tested function in the pipeline (`tests/test_prepare_dataset.py`).

**Phone-free images are rationed.** An image whose boxes were all dropped becomes
a *background*. Backgrounds are useful — they teach the detector what isn't a
phone — but these datasets contain a lot of them, and a detector trained mostly
on absence learns to predict nothing, which is the exact failure we're fixing.
`--background-ratio` caps them at 10% of the positives by default.

Licences vary per dataset on Universe — check the page before shipping anything.

### Head pose — for validating, not training

[`ETHZurich/biwi_kinect_head_pose`](https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose)
has 24 sequences with ground-truth yaw/pitch/roll. Not detector data, but it
would let `src/features/head_pose.py` report a **mean absolute angular error**
instead of being asserted correct. Licence is research/non-commercial, so it
stays out of anything shipped.

### Rejected

Recorded so nobody re-checks them:

| Dataset | Why not |
|---|---|
| `ybli/yolo-classroom-student-head-up-head-down` | Repo is empty; README points to a Baidu Cloud link + extraction code |
| `ybli/yolo-phone-book-cup-object-detection` | Same — empty repo |
| `ybli/yolo-driver-distraction-detection` | Same — empty repo |
| `lord-reso/inbrowser-proctor-dataset` | Name is misleading: audio/ASR (505 speech clips), not vision |
| `vibrantturtle/phone-detection-data` | 998 images, zero annotation files — unlabelled |
| `lamkser/face_occlusion` | Empty repo |
| `gymprathap/Driver-Distracted-Dataset` | 4.3 GB, no README, classification not detection, car-interior domain |
| `MahekDharod/cellphone-detection-dataset` | MIT and plausible, but README is a licence line only — contents unverified |

---

## 3. The domain gap, and why public data alone won't fix this

This is the central point.

Public phone datasets are **clean**: centred, well-lit, unoccluded, often
product-photo-ish. The adopted HF set is 512×512 crops. Our failure mode is the
opposite — a dim webcam, a phone half-hidden by a hand, held at lap level,
motion-blurred.

Training only on public images would improve a metric measured on public images
and change very little about the thing we actually care about. So the plan is
**public data for "what a phone looks like" + our own annotated frames for "what
a phone looks like *in this camera*"**, and to report the two separately:

```
python -m src.detection.train_detector --eval-source own      # the number that matters
python -m src.detection.train_detector --eval-source public   # the flattering one
```

Averaging them would hide the result behind the easy half of the data.

### Spending annotation effort where it pays

Annotation is human time, so the frames worth labelling are the ones the baseline
detector *missed* in clips already known to contain a phone. Those are false
negatives by construction — the model's blind spot, identified for free by the
clip-level label. A frame the detector already nails teaches it almost nothing.

This is active-learning-style sample selection, and it applies to any deployment
environment where the detector underperforms.

This is the cheap end of active learning (uncertainty sampling with the clip
label as oracle). The honest caveat: training only on hard examples biases the
set and can hurt calibration on easy ones, so `--keep-detected` mixes a thinned
slice of already-detected frames back in.

The per-clip breakdown is itself diagnostic:

```
phone_002.mp4   25/25 missed      phone_004.mp4    1/21 missed
phone_005.mp4   27/27 missed      phone_010.mp4   10/21 missed
```

Some clips the baseline handles fine; in others it is blind end to end. That
spread is a data-collection finding, not just a modelling one — whatever pose
`phone_004` uses is the one COCO already covers.

This currently yields **201 frames** awaiting annotation in
`data/detection/raw_frames/` (gitignored — they're cut from private footage),
alongside a manifest recording each frame's baseline confidence.

Two labelling rules matter more than the tool used: a frame with no visible phone
gets an *empty* label rather than a guess, and frames from one clip must never
straddle the train/val boundary.

---

## 4. Measured baseline

Pre-trained YOLOv8n, scored on the 121-image public validation split by our own
AP implementation (`src/detection/detection_metrics.py`, tested in
`tests/test_detection_metrics.py`):

| Model | AP@0.5 | Precision | Recall |
|---|---|---|---|
| yolov8n (COCO `cell phone`) | 0.432 | 0.761 | 0.406 |

Reproduce with `python -m src.detection.train_detector --baseline-only`.

Note the shape of it: precision 0.76 but recall 0.41. When the baseline says
"phone" it is usually right — it simply misses most of them. That is exactly the
failure a fine-tune on in-domain data should fix, and it is why recall is the
metric to watch rather than a single averaged score.

**Why score it ourselves rather than trust `model.val()`.** The baseline predicts
COCO class 67 (`cell phone`); our fine-tuned model predicts class 0 (`phone`). No
single Ultralytics call scores both, and a comparison is only meaningful if both
sides go through identical matching code.

### Pipeline smoke test (not a result)

A deliberately minimal run — **1 epoch at 320px on CPU** — to prove the path works
end to end:

| Model | AP@0.5 | Precision | Recall |
|---|---|---|---|
| yolov8n baseline | 0.432 | 0.761 | 0.406 |
| fine-tuned (1 epoch, 320px) | 0.529 | 0.574 | 0.526 |

Do **not** quote this as a Phase 2 result. One epoch on 481 clean public images
is nowhere near converged, and it is scored on public images only. Its value is
that the plumbing is verified and the direction is the expected one: recall rises
(0.41 → 0.53) while precision falls, which is what adapting a
conservative-but-blind detector to a narrower domain looks like.

---

## 5. Pipeline

```
sources.py            registry: what exists, licence, caveat
   │
roboflow_import.py    ROBOFLOW_API_KEY ──► YOLO export ──► data/detection/external/
   │
(annotation)          own clips ──► frames the baseline missed ──► label (Roboflow/CVAT/labelImg)
   │
prepare_dataset.py    HF COCO JSON ────┬─► YOLO layout + data.yaml
                      Roboflow export ─┤   (classes remapped, unknowns dropped,
                      our .txt labels ─┘    own frames split by CLIP not by frame)
   │
train_detector.py     fine-tune ──► score vs baseline with our own AP code
```

The clip-grouped split exists because consecutive frames of one clip are
near-duplicates: letting them straddle train/val leaks the answer and reports a
mAP we haven't earned. `split_by_source.py` applies the same rule to the public
set, where one source video was 87% of the data and 100% of validation.

---

## 6. Honest limitations

- **605 public images is small.** Enough to demonstrate the pipeline; not enough
  for a strong detector. The Roboflow sets are the obvious next addition.
- **Our own frames are one person, one room, one camera.** Whatever we train will
  be fitted to that. Multi-person and varied-lighting clips remain the highest-
  value data-collection work, exactly as in Phase 1.
- **The public val split is public images only** until our frames are annotated,
  so the 0.432 baseline is measured on the easy domain. Expect the on-our-footage
  number to be far worse — the 20.7% detection rate says so.
- **Licences differ per source.** Apache-2.0 for the adopted HF set; Roboflow
  varies per dataset; Biwi is research-only. Anything shipped needs this checked.
