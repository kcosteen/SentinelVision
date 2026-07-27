"""Find the Eye Aspect Ratio at which an eye should count as closed.

`EAR_THRESHOLD = 0.20` appears in both `blink_detection.py` and
`feature_extractor.py`. It's the value from the original EAR paper, which is a
reasonable starting point but was not measured on our pipeline, our landmark
detector, or anyone's face in particular. Since every blink count downstream
depends on it, it's worth measuring.

    # Download, measure, and sweep (start small -- 2000 images is plenty):
    python -m src.calibration.calibrate_ear --limit 2000

    # Bigger sample, finer sweep:
    python -m src.calibration.calibrate_ear --limit 20000 --step 0.01

Data: `MichalMlodawski/closed-open-eyes` (ODC-BY, 126,560 labelled eye images).

**Why this is measurable at all.** EAR is computed from MediaPipe face-mesh
landmarks, so we can run the real `calculate_ear` over labelled open/closed eyes
and see exactly where the two distributions separate. That makes the threshold an
empirical question rather than a citation.

**The honest limitation.** These are cropped eye images from a different capture
setup than our webcam. EAR is a ratio of distances, so it's more robust to scale
and resolution than a raw pixel measure would be -- but head rotation squashes it,
and a dataset of frontal crops under-represents that. Treat the result as a much
better-grounded prior than 0.20-from-a-paper, not as the final word for our
camera. The distribution overlap printed at the end is the thing to look at: if
the two classes overlap heavily, no single threshold saves you and the honest
answer is that EAR alone is a weak signal.
"""

import argparse
import io
import os

import numpy as np

from src.calibration.sweep import (
    best_threshold,
    candidate_thresholds,
    summarise_choice,
)

DATASET_REPO = "MichalMlodawski/closed-open-eyes"

# The value both modules use today.
CURRENT_EAR_THRESHOLD = 0.20

# MediaPipe face-mesh indices, matching blink_detection.py / feature_extractor.py.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def load_eye_dataframe(limit, cache_dir):
    """Download parquet shards and return a DataFrame of up to `limit` rows.

    Read with pandas directly rather than the `datasets` library: the files are
    plain parquet, and this avoids adding a heavy dependency for one table.

    Shards are sampled EVENLY across the repo, never as a head slice. This
    dataset is ordered by class -- the first twenty shards are 100% `closed_eyes`
    -- so taking the first N gives a single-class sample from which no threshold
    can be derived at all. The same striding trick calibrate_phone_conf uses to
    avoid sampling a few seconds of one video.
    """
    import pandas as pd
    from huggingface_hub import list_repo_files, hf_hub_download

    files = sorted(
        f for f in list_repo_files(DATASET_REPO, repo_type="dataset")
        if f.endswith(".parquet")
    )
    if not files:
        raise SystemExit(f"No parquet files found in {DATASET_REPO}.")

    # Roughly 100 rows per shard, so aim for the shard count that covers `limit`
    # and spread that many picks over the whole repo.
    wanted = max(2, min(len(files), (limit // 100) + 1))
    stride = max(1, len(files) // wanted)
    picks = files[::stride][:wanted]
    print(f"  {len(files)} shards available; sampling {len(picks)} evenly "
          f"(stride {stride}) to cover both classes")

    frames = []
    rows = 0
    for name in picks:
        path = hf_hub_download(
            DATASET_REPO, name, repo_type="dataset", local_dir=cache_dir
        )
        frame = pd.read_parquet(path)
        frames.append(frame)
        rows += len(frame)
        print(f"  {name}: {len(frame)} rows (total {rows})")

    combined = pd.concat(frames, ignore_index=True)
    if len(combined) > limit:
        # Evenly again, so trimming can't undo the spread we just bought.
        combined = combined.iloc[:: max(1, len(combined) // limit)].head(limit)
    return combined.reset_index(drop=True)


def label_to_closed(value):
    """Normalise the dataset's label into 1 = closed, 0 = open, None = unusable.

    This repo's actual vocabulary is `closed_eyes` / `open_eyes`; the rest are
    spellings other eye datasets use, kept so the script survives a source swap.
    """
    text = str(value).strip().lower()
    if text in {"closed", "close", "1", "closed_eye", "closed_eyes", "closedeyes"}:
        return 1
    if text in {"open", "opened", "0", "open_eye", "open_eyes", "openeyes"}:
        return 0
    return None


def measure_ear(image_bytes, face_mesh):
    """EAR for one eye image, or None if no face mesh could be fitted.

    Returns the mean of both eyes when the mesh is found. Cropped eye images
    often fail mesh detection -- that's expected, and those rows are skipped
    rather than counted as a measurement of zero.
    """
    import cv2
    from src.features.eye_analysis import calculate_ear

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return None

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mesh = face_mesh.process(rgb)
    if not mesh.multi_face_landmarks:
        return None

    landmarks = mesh.multi_face_landmarks[0].landmark
    left = calculate_ear(LEFT_EYE, landmarks, width, height)
    right = calculate_ear(RIGHT_EYE, landmarks, width, height)
    return (left + right) / 2


def describe_separation(scores, labels):
    """How far apart are the open and closed distributions?

    If they overlap heavily, no threshold performs well and the honest conclusion
    is that the feature is weak -- not that the threshold needs more tuning.
    """
    closed = np.array([s for s, l in zip(scores, labels) if l == 1])
    opened = np.array([s for s, l in zip(scores, labels) if l == 0])
    if not len(closed) or not len(opened):
        return

    print("\nEAR distribution by class")
    print(f"{'class':<10}{'n':>7}{'mean':>9}{'std':>9}{'p10':>9}{'p50':>9}{'p90':>9}")
    print("-" * 62)
    for name, values in (("closed", closed), ("open", opened)):
        print(f"{name:<10}{len(values):>7}{values.mean():>9.3f}{values.std():>9.3f}"
              f"{np.percentile(values, 10):>9.3f}{np.percentile(values, 50):>9.3f}"
              f"{np.percentile(values, 90):>9.3f}")

    # A crude but readable separation measure: how many standard deviations
    # apart the means are (Cohen's d).
    pooled = np.sqrt((closed.std() ** 2 + opened.std() ** 2) / 2)
    if pooled > 0:
        d = abs(opened.mean() - closed.mean()) / pooled
        verdict = ("well separated" if d > 2 else
                   "usable but overlapping" if d > 1 else
                   "heavily overlapping -- EAR alone is a weak signal here")
        print(f"\nSeparation (Cohen's d): {d:.2f} -- {verdict}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=2000,
                        help="how many labelled images to measure (default 2000)")
    parser.add_argument("--cache-dir",
                        default=os.path.join("data", "calibration", "closed_open_eyes"))
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.45)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--metric", choices=["f1", "precision", "recall"], default="f1")
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--current", type=float, default=CURRENT_EAR_THRESHOLD)
    args = parser.parse_args()

    print(f"Loading {DATASET_REPO} (limit {args.limit})")
    frame = load_eye_dataframe(args.limit, args.cache_dir)
    print(f"  columns: {list(frame.columns)}")

    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True, static_image_mode=True
    )

    scores, labels = [], []
    no_mesh = unusable_label = 0

    for _, row in frame.iterrows():
        closed = label_to_closed(row.get("Label"))
        if closed is None:
            unusable_label += 1
            continue

        payload = row.get("Image_data")
        if isinstance(payload, dict):
            # This repo stores {'filename': str, 'file': bytes}; the usual HF
            # image struct uses 'bytes'. Accept either.
            payload = payload.get("file") or payload.get("bytes")
        if isinstance(payload, (bytes, bytearray)):
            ear = measure_ear(bytes(payload), face_mesh)
        else:
            ear = None

        if ear is None:
            no_mesh += 1
            continue

        scores.append(ear)
        # We're detecting the CLOSED state, so closed is the positive class.
        labels.append(closed)

    print(f"\nMeasured {len(scores)} images "
          f"({no_mesh} no face mesh, {unusable_label} unreadable label)")
    if not scores:
        # Name the actual cause rather than guessing at one -- the two failures
        # need completely different fixes.
        if unusable_label:
            raise SystemExit(
                f"No usable measurements: {unusable_label} rows had a label this "
                "script does not recognise. Print frame['Label'].unique() and add "
                "the spellings to label_to_closed()."
            )
        raise SystemExit(
            "No usable measurements: MediaPipe fitted a face mesh to none of the "
            "images -- they may be tight eye crops rather than faces. Inspect the "
            "cached parquet before trusting any threshold."
        )

    positives = sum(labels)
    print(f"  closed: {positives}   open: {len(labels) - positives}")

    if positives == 0 or positives == len(labels):
        raise SystemExit(
            "Only one class survived measurement, so no threshold is derivable "
            "from this sample -- any cut-point scores identically. The shards are "
            "ordered by class; widen --limit so the even sampling reaches both."
        )

    describe_separation(scores, labels)

    from src.calibration.sweep import sweep_threshold
    # EAR runs the other way round to a confidence: LOW means closed. Negate both
    # sides so the shared ">= threshold" sweep still means "predicted closed".
    rows = sweep_threshold(
        [-s for s in scores], labels,
        [-t for t in candidate_thresholds(args.start, args.stop, args.step)],
    )
    for row in rows:
        row["threshold"] = -row["threshold"]
    rows.sort(key=lambda r: r["threshold"])

    chosen = best_threshold(rows, metric=args.metric, min_recall=args.min_recall)
    print()
    summarise_choice(rows, chosen, args.current, "EAR", metric=args.metric)

    print("\nCAVEAT: measured on cropped eye images from a different capture setup "
          "than\nour webcam. EAR is scale-invariant, which helps, but head rotation "
          "squashes\nit and frontal crops under-represent that. Treat this as a "
          "grounded prior,\nnot the final word for our camera.")


if __name__ == "__main__":
    main()
