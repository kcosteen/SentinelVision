"""Find the confidence threshold at which the phone detector should fire.

`feature_extractor.py` treats a detection as real when its confidence clears 0.5;
`object_tracker.py` uses 0.6. Neither number came from anywhere, and they
disagree with each other. This measures both against labelled data.

    # Sweep on the held-out split of the built dataset:
    python -m src.calibration.calibrate_phone_conf

    # Prefer precision -- fewer false accusations -- but still catch half:
    python -m src.calibration.calibrate_phone_conf --metric precision --min-recall 0.5

    # Use every labelled image, not just val:
    python -m src.calibration.calibrate_phone_conf --split all

Requires a built dataset:  python -m src.detection.prepare_dataset

**Why detection-level and not image-level.** "Does this image contain a phone?"
would need phone-free images, and the public set is all-positive. Counting at the
box level sidesteps that: a false positive is a box predicted where no phone is,
which an all-positive dataset supplies perfectly well.

**Why the model runs once.** Predictions are gathered at a floor of 0.01 and
matched to ground truth a single time, then filtered per threshold. Matching is
greedy in descending confidence, so dropping low-confidence predictions cannot
change the verdict on higher-confidence ones -- the filtered result is identical
to re-running the match, for a fraction of the compute.
"""

import argparse
import os

import cv2

from src.calibration.sweep import best_threshold, candidate_thresholds, summarise_choice
from src.detection.class_ids import phone_class_index
from src.detection.detection_metrics import match_image
from src.detection.train_detector import read_yolo_labels

# The value feature_extractor.py currently uses -- what we're checking.
CURRENT_PHONE_CONF = 0.5

# Low enough to capture the whole curve; anything below is noise, not signal.
CONF_FLOOR = 0.01


def resolve_layout(args, split):
    """Work out where the images/labels are, and which class id is the phone.

    Two layouts are supported: the single-class set built by prepare_dataset
    (images/<split>/) and a raw multi-class Roboflow export (<split>/images/).
    Reading the phone class out of the export's data.yaml rather than assuming
    an index means a reordered class list can't silently score the wrong object.
    """
    if not args.export_root:
        return (os.path.join(args.dataset_root, "images", split),
                os.path.join(args.dataset_root, "labels", split),
                None)

    data_yaml = os.path.join(args.export_root, "data.yaml")
    if not os.path.exists(data_yaml):
        raise SystemExit(f"No data.yaml in {args.export_root}")

    import yaml
    names = (yaml.safe_load(open(data_yaml)) or {}).get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    phone_class = phone_class_index(names)

    print(f"  classes in export: {names}  ->  using index {phone_class}")
    return (os.path.join(args.export_root, split, "images"),
            os.path.join(args.export_root, split, "labels"),
            phone_class)


def collect_matches(model, image_dir, label_dir, iou_threshold, class_filter,
                    only_class, limit):
    """Match predictions to ground truth once. Returns (matches, n_truths).

    `matches` is [(confidence, is_true_positive)] pooled over every image.
    """
    if not os.path.isdir(image_dir):
        raise SystemExit(f"No images at {image_dir}")

    matches = []
    n_truths = 0
    n_images = 0
    n_negative_images = 0   # images with no phone at all -- see the caveat below

    names = sorted(
        n for n in os.listdir(image_dir)
        if n.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if limit:
        # Evenly spaced rather than the first N: consecutive frames from one
        # source video are near-duplicates, so a head slice would sample a few
        # seconds of footage and call it a dataset.
        stride = max(1, len(names) // limit)
        names = names[::stride][:limit]

    for name in names:
        image_path = os.path.join(image_dir, name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]
        n_images += 1

        stem = os.path.splitext(name)[0]
        truths = read_yolo_labels(
            os.path.join(label_dir, stem + ".txt"), width, height, only_class
        )
        n_truths += len(truths)
        if not truths:
            n_negative_images += 1

        # Without the class filter every COCO class (person, chair, ...) would be
        # counted as a phone candidate and inflate false positives enormously.
        kwargs = {"conf": CONF_FLOOR, "verbose": False}
        if class_filter is not None:
            kwargs["classes"] = class_filter
        results = model(image, **kwargs)
        predictions = [
            ((*(float(v) for v in box.xyxy[0]),), float(box.conf[0]))
            for box in results[0].boxes
        ]
        matches.extend(match_image(predictions, truths, iou_threshold))

    return matches, n_truths, n_images, n_negative_images


def sweep_detection_threshold(matches, n_truths, thresholds):
    """Precision/recall/F1 at each confidence threshold, counted over boxes.

    Deliberately not `sweep.sweep_threshold`: that one classifies a fixed set of
    samples, whereas here a threshold changes *how many predictions exist at all*,
    and false negatives come from unmatched ground truth rather than from rows.
    """
    rows = []
    for threshold in thresholds:
        kept = [(conf, is_tp) for conf, is_tp in matches if conf >= threshold]
        tp = sum(is_tp for _, is_tp in kept)
        fp = len(kept) - tp
        fn = n_truths - tp

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n_truths if n_truths else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": 0,
            "precision": prec, "recall": rec, "f1": f1,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-root", default=os.path.join("data", "detection", "dataset"))
    parser.add_argument("--export-root", default=None,
                        help="a raw multi-class Roboflow YOLO export to calibrate "
                             "against instead of the built dataset")
    parser.add_argument("--limit", type=int, default=0,
                        help="evenly sample at most N images (0 = all)")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--split", default="val",
                        help="val/train/all, or a split name in the export "
                             "(train/valid/test)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU at which a prediction counts as correct")
    parser.add_argument("--metric", choices=["f1", "precision", "recall"], default="f1")
    parser.add_argument("--min-recall", type=float, default=0.0,
                        help="ignore thresholds that catch less than this share")
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--current", type=float, default=CURRENT_PHONE_CONF,
                        help="the value in the code today, for comparison")
    parser.add_argument("--finetuned", action="store_true",
                        help="weights are our single-class model (predicts class 0) "
                             "rather than the COCO baseline (class 67)")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    # Ask the model which of ITS classes is the phone, rather than assuming.
    # This has to work for three different models: the 80-class COCO baseline
    # (index 67), a single-class fine-tune (index 0), and a multi-class
    # proctoring fine-tune (index 1 of six). Hard-coding any of them -- or
    # disabling the filter entirely -- silently counts every `person` and
    # `laptop` box as a phone prediction and reports nonsense.
    wanted = {"cell phone", "phone", "mobile_phone", "mobile phone"}
    model_names = model.names or {}
    phone_index = next(
        (i for i, n in model_names.items() if str(n).strip().lower() in wanted), None
    )
    if phone_index is None:
        raise SystemExit(
            f"No phone-like class among the model's classes: {list(model_names.values())}"
        )
    class_filter = [phone_index]

    splits = ["train", "val"] if args.split == "all" else [args.split]
    matches, n_truths, n_images, n_negatives = [], 0, 0, 0
    for split in splits:
        image_dir, label_dir, only_class = resolve_layout(args, split)
        split_matches, split_truths, split_images, split_negatives = collect_matches(
            model, image_dir, label_dir, args.iou, class_filter, only_class, args.limit
        )
        matches.extend(split_matches)
        n_truths += split_truths
        n_negatives += split_negatives
        n_images += split_images

    print(f"Model: {args.weights}   split: {args.split}")
    print(f"  model classes: {list(model_names.values())}")
    print(f"  scoring class {phone_index} ({model_names[phone_index]!r}) only")
    print(f"{n_images} images, {n_truths} ground-truth boxes, "
          f"{len(matches)} raw predictions at conf>={CONF_FLOOR}\n")

    rows = sweep_detection_threshold(
        matches, n_truths,
        candidate_thresholds(start=args.start, stop=args.stop, step=args.step),
    )
    chosen = best_threshold(rows, metric=args.metric, min_recall=args.min_recall)
    summarise_choice(rows, chosen, args.current, "phone conf", metric=args.metric)

    print("\nCAVEATS -- read before changing the constant:")
    print("  * The right threshold moves with both the model and the domain;")
    print("    re-run after fine-tuning and on our own frames (--finetuned).")

    # Whether precision here is trustworthy depends entirely on the data, so
    # measure it rather than asserting it. An all-positive set cannot show a
    # phone hallucinated on an empty desk, and flatters precision accordingly.
    share = n_negatives / n_images if n_images else 0
    if share < 0.05:
        print(f"  * This set is effectively ALL-POSITIVE ({n_negatives}/{n_images}")
        print("    images have no phone), so a false positive can only be a")
        print("    mislocated box, never a hallucination on an empty desk. Real")
        print("    footage is mostly phone-free, so the precision above is")
        print("    OPTIMISTIC and a low threshold would fire far more in practice.")
    else:
        print(f"  * {n_negatives}/{n_images} images ({share:.0%}) contain no phone,")
        print("    so false positives here are genuine hallucinations and the")
        print("    precision above is a fair estimate rather than a flattering one.")

    print("  * Proctoring is asymmetric -- a false accusation costs more than a")
    print("    missed phone. Prefer --metric precision --min-recall N over raw F1.")


if __name__ == "__main__":
    main()
