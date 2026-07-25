"""Fine-tune YOLOv8 for phone detection and score it against the baseline.

Phase 2's claim is not "we trained a detector" -- it's "we trained a detector that
beats the pre-trained one on our footage, by this much". So this script always
reports both models on the *same* held-out images, using our own AP
implementation (`detection_metrics.py`) rather than each model's self-reported
number.

    # Evaluate the baseline only -- run this BEFORE training to get the "before".
    python -m src.detection.train_detector --baseline-only

    # Fine-tune, then compare both on the val split:
    python -m src.detection.train_detector --epochs 50

    # The number that actually matters: performance on OUR frames only.
    python -m src.detection.train_detector --eval-source own

Design choices worth defending in an interview:

* **Both models scored by the same code.** The baseline emits COCO class 67
  ('cell phone'); ours emits class 0 ('phone'). Scoring them through one matcher
  removes the class-id mismatch and any doubt about differing conventions.
* **Predictions are gathered at a very low confidence floor.** AP integrates the
  whole precision/recall curve, so throwing away low-confidence boxes up front
  would truncate the curve and silently inflate the score.
* **`--eval-source own` exists because a mixed average flatters us.** Public
  images are clean and easy; averaging over them hides how we do on the dim,
  occluded webcam frames that are the actual product. Report both, lead with ours.
* **Freeze the backbone (`--freeze`) when the dataset is small.** With a few
  hundred images, updating all layers mostly memorises. The early layers already
  encode generic edges/textures worth keeping.
"""

import argparse
import os

import cv2

from src.detection.detection_metrics import evaluate_detections

# COCO class id for 'cell phone' -- what the pre-trained model calls our target.
COCO_CELL_PHONE = 67

# Deliberately near-zero: AP needs the full curve, not a pre-filtered top slice.
AP_CONF_FLOOR = 0.001


def read_yolo_labels(label_path, width, height):
    """YOLO '<cls> <cx> <cy> <w> <h>' (normalised) -> [(x1, y1, x2, y2)] in pixels."""
    if not os.path.exists(label_path):
        return []

    boxes = []
    with open(label_path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = (float(p) for p in parts[:5])
            # Centre form -> corner form, scaled back up to pixels.
            boxes.append((
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ))
    return boxes


def list_val_images(dataset_root, source):
    """Val image paths, optionally filtered to one source via the filename prefix."""
    image_dir = os.path.join(dataset_root, "images", "val")
    if not os.path.isdir(image_dir):
        raise SystemExit(
            f"No val images at {image_dir}. Run: python -m src.detection.prepare_dataset"
        )

    names = sorted(
        n for n in os.listdir(image_dir)
        if n.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if source != "all":
        # prepare_dataset.py prefixes every file with its source ("own__", "public__").
        names = [n for n in names if n.startswith(f"{source}__")]
    return [os.path.join(image_dir, n) for n in names]


def score_model(model, image_paths, dataset_root, class_filter=None):
    """Run a model over the val images and score it with our own AP code."""
    label_dir = os.path.join(dataset_root, "labels", "val")
    per_image = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]

        stem = os.path.splitext(os.path.basename(image_path))[0]
        truths = read_yolo_labels(os.path.join(label_dir, stem + ".txt"), width, height)

        kwargs = {"conf": AP_CONF_FLOOR, "verbose": False}
        if class_filter is not None:
            kwargs["classes"] = class_filter
        results = model(image, **kwargs)

        predictions = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            predictions.append(((x1, y1, x2, y2), float(box.conf[0])))

        per_image.append((predictions, truths))

    return evaluate_detections(per_image)


def print_comparison(rows):
    """Side-by-side table; the delta is the headline Phase 2 result."""
    header = f"{'model':<28}{'AP@0.5':>9}{'precision':>11}{'recall':>9}"
    print("\n" + header)
    print("-" * len(header))
    for name, metrics in rows:
        print(f"{name:<28}{metrics['ap50']:>9.3f}"
              f"{metrics['precision']:>11.3f}{metrics['recall']:>9.3f}")

    if len(rows) == 2:
        before, after = rows[0][1]["ap50"], rows[1][1]["ap50"]
        delta = after - before
        relative = f" ({delta / before:+.0%})" if before > 0 else ""
        print("-" * len(header))
        print(f"{'improvement':<28}{delta:>+9.3f}{relative}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_root = os.path.join("data", "detection", "dataset")
    parser.add_argument("--dataset-root", default=default_root)
    parser.add_argument("--data", default=None,
                        help="path to data.yaml (default: <dataset-root>/data.yaml)")
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="starting weights; also the baseline being beaten")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--freeze", type=int, default=10,
                        help="freeze the first N layers (0 = train everything)")
    parser.add_argument("--device", default=None, help="'cpu', '0', ... (default: auto)")
    # Absolute, so Ultralytics saves here instead of nesting under its runs_dir.
    parser.add_argument("--project", default=os.path.abspath(os.path.join("models", "detection")))
    parser.add_argument("--name", default="phone_yolov8n")
    parser.add_argument("--eval-source", choices=["all", "own", "public"], default="all",
                        help="score on all val images, or only ours / only public")
    parser.add_argument("--baseline-only", action="store_true",
                        help="score the pre-trained model without training")
    args = parser.parse_args()

    data_yaml = args.data or os.path.join(args.dataset_root, "data.yaml")
    if not os.path.exists(data_yaml):
        raise SystemExit(
            f"{data_yaml} not found. Run: python -m src.detection.prepare_dataset"
        )

    from ultralytics import YOLO

    image_paths = list_val_images(args.dataset_root, args.eval_source)
    if not image_paths:
        raise SystemExit(f"No val images for --eval-source {args.eval_source}.")
    print(f"Val images ({args.eval_source}): {len(image_paths)}")

    print(f"\nScoring baseline: {args.weights} (COCO 'cell phone')")
    baseline = YOLO(args.weights)
    baseline_metrics = score_model(baseline, image_paths, args.dataset_root,
                                   class_filter=[COCO_CELL_PHONE])
    print(f"  {baseline_metrics['n_predictions']} predictions over "
          f"{baseline_metrics['n_truths']} ground-truth boxes")

    if args.baseline_only:
        print_comparison([("baseline yolov8n", baseline_metrics)])
        print("\nThis is the 'before' number. Now train: drop --baseline-only.")
        return

    print(f"\nFine-tuning for {args.epochs} epochs (freeze={args.freeze})")
    model = YOLO(args.weights)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    # Ultralytics nests a *relative* `project` under its own configured runs_dir
    # (so 'models/detection' becomes 'runs/detect/models/detection'). Ask the
    # trainer where it actually saved rather than reconstructing the path.
    best = getattr(model.trainer, "best", None)
    if not best or not os.path.exists(best):
        best = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    if not os.path.exists(best):
        raise SystemExit(f"Training finished but no weights at {best}")

    print(f"\nScoring fine-tuned: {best}")
    tuned_metrics = score_model(YOLO(best), image_paths, args.dataset_root)

    print_comparison([
        ("baseline yolov8n (COCO)", baseline_metrics),
        ("fine-tuned", tuned_metrics),
    ])
    print(f"\nWeights -> {best}")
    if args.eval_source == "all":
        print("Now re-check on our footage only:\n"
              "  python -m src.detection.train_detector --baseline-only --eval-source own")


if __name__ == "__main__":
    main()
