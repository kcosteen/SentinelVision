"""Pull frames out of our own clips so they can be annotated for Phase 2.

Public phone datasets are clean: centred, well-lit, unoccluded. Our failure mode
is the opposite -- a dim webcam, a phone half-hidden by a hand at lap level,
motion-blurred. Training only on public data therefore does *not* fix the thing
we measured (baseline YOLOv8n finds the phone in ~18% of frames in our own
`phone_*` clips). The fix is to teach the detector our camera, which means
annotating our own frames.

    # Which frames does the baseline MISS? (the ones worth annotating)
    python -m src.detection.extract_frames --strategy missed

    # Plain every-Nth-frame sampling, all clips:
    python -m src.detection.extract_frames --strategy uniform --every 15 --clips "*"

    # See the counts without writing any images:
    python -m src.detection.extract_frames --strategy missed --dry-run

**Why `missed` is the default.** Annotation is the expensive, human-time step, so
spend it where it buys the most. A frame the baseline already detects confidently
teaches the model almost nothing; a frame from a clip we *know* contains a phone
but where the detector saw nothing is a labelled-by-construction false negative.
Sampling those concentrates the annotation budget on the model's actual blind
spot. This is the cheap end of active learning -- uncertainty sampling, using the
clip-level label we already have as the oracle.

The catch worth stating out loud: training only on missed frames biases the set
toward hard examples and can hurt calibration on easy ones. `--keep-detected`
mixes a fraction of already-detected frames back in as a counterweight.

Frames are flipped to match `FeatureExtractor`, which mirrors every frame before
inference -- so what we annotate is exactly what the live pipeline will see.

Next step after this: annotate the JPEGs (Roboflow, CVAT, or labelImg all export
YOLO .txt), drop the labels next to the images, then run prepare_dataset.py.
"""

import argparse
import csv
import glob
import os

import cv2

from src.detection.class_ids import phone_class_index

# Only these clips are assumed to contain a phone. The clip-level label is what
# makes a no-detection frame a *known* miss rather than merely an empty frame.
DEFAULT_CLIP_GLOB = "phone_*"

MANIFEST_FIELDS = [
    "frame_file", "clip", "frame_index", "time_sec", "baseline_detected", "baseline_conf",
]


def find_clips(clips_dir, pattern):
    """Video paths in `clips_dir` matching a filename glob (without extension)."""
    extensions = ("mp4", "avi", "mov", "mkv")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(clips_dir, f"{pattern}.{ext}")))
    return sorted(paths)


def baseline_phone_conf(yolo, frame, conf_threshold, phone_class):
    """Highest phone confidence this detector gives the frame.

    Returns 0.0 when it sees no phone -- which, in a clip we labelled `phone`,
    is precisely the miss we want to annotate.
    """
    # Restricting to the one class we care about also keeps this fast.
    results = yolo(frame, classes=[phone_class], verbose=False)
    best = 0.0
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf >= conf_threshold:
            best = max(best, conf)
    return best


def sample_clip(path, yolo, args, phone_class=None):
    """Walk one clip; return the rows (frame, metadata) selected for annotation.

    Returns (selected_rows, total_considered, missed_count) so the caller can
    report the baseline's real miss rate rather than assert it.
    """
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    clip = os.path.splitext(os.path.basename(path))[0]

    selected = []
    considered = missed = detected_count = 0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        # Sample every Nth frame: neighbouring frames are near-duplicates, so
        # annotating all of them costs human time and adds almost no information.
        if frame_index % args.every != 0:
            frame_index += 1
            continue

        # Match FeatureExtractor, which mirrors before inference.
        frame = cv2.flip(frame, 1)
        considered += 1

        conf = 0.0
        if yolo is not None:
            conf = baseline_phone_conf(yolo, frame, args.conf, phone_class)
        detected = conf > 0
        if detected:
            detected_count += 1
        else:
            missed += 1

        keep = True
        if args.strategy == "missed":
            # Keep every miss, plus every Nth *detected* frame as a counterweight.
            # The counter must be over detected frames specifically -- keying it on
            # the miss count would keep every hit until the first miss occurred.
            keep = (not detected) or bool(
                args.keep_detected and detected_count % args.keep_detected == 0
            )

        if keep:
            selected.append({
                "frame": frame,
                "frame_file": f"{clip}_f{frame_index:05d}.jpg",
                "clip": clip,
                "frame_index": frame_index,
                "time_sec": round(frame_index / fps, 3),
                "baseline_detected": int(detected),
                "baseline_conf": round(conf, 4),
            })

        frame_index += 1

    capture.release()
    return selected, considered, missed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clips-dir", default="clips")
    parser.add_argument("--clips", default=DEFAULT_CLIP_GLOB,
                        help=f"filename glob without extension (default {DEFAULT_CLIP_GLOB!r})")
    parser.add_argument("--out-dir", default=os.path.join("data", "detection", "raw_frames"))
    parser.add_argument("--strategy", choices=["missed", "uniform"], default="missed",
                        help="'missed' keeps frames the baseline detector failed on")
    parser.add_argument("--every", type=int, default=10,
                        help="consider every Nth frame (default 10)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="confidence at/above which the baseline counts as a detection")
    parser.add_argument("--keep-detected", type=int, default=5,
                        help="in 'missed' mode also keep 1 in N detected frames (0 = none)")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--dry-run", action="store_true",
                        help="report counts without writing images")
    args = parser.parse_args()

    clips = find_clips(args.clips_dir, args.clips)
    if not clips:
        raise SystemExit(f"No clips matching '{args.clips}' in {args.clips_dir}/")

    # 'uniform' needs no model, so don't pay to load one.
    yolo = None
    phone_class = None
    if args.strategy == "missed":
        from ultralytics import YOLO
        yolo = YOLO(args.weights)
        phone_class = phone_class_index(yolo)
        print(f"{args.weights}: scoring class {phone_class} "
              f"({yolo.names[phone_class]!r}) of {len(yolo.names)}\n")

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    manifest = []
    total_considered = total_missed = 0

    for path in clips:
        selected, considered, missed = sample_clip(path, yolo, args, phone_class)
        total_considered += considered
        total_missed += missed

        for row in selected:
            if not args.dry_run:
                cv2.imwrite(os.path.join(args.out_dir, row["frame_file"]), row["frame"])
            manifest.append({field: row[field] for field in MANIFEST_FIELDS})

        rate = f"{missed}/{considered} missed" if yolo is not None else f"{considered} sampled"
        print(f"{os.path.basename(path):<24} {rate:>18}  -> {len(selected)} kept")

    if not args.dry_run:
        manifest_path = os.path.join(args.out_dir, "manifest.csv")
        with open(manifest_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(manifest)

    print(f"\nClips: {len(clips)}   frames considered: {total_considered}")
    if yolo is not None and total_considered:
        # The number that justifies Phase 2 existing at all -- measured, not assumed.
        pct = 100 * (total_considered - total_missed) / total_considered
        print(f"Baseline yolov8n detected a phone in {pct:.1f}% of them "
              f"({total_missed} misses).")
    print(f"Frames selected for annotation: {len(manifest)}")

    if args.dry_run:
        print("\nDry run -- nothing written. Drop --dry-run to extract.")
    else:
        print(f"Wrote -> {args.out_dir}/  (+ manifest.csv)")
        print("\nNext: annotate these as YOLO .txt (Roboflow / CVAT / labelImg),\n"
              "      then: python -m src.detection.prepare_dataset")


if __name__ == "__main__":
    main()
