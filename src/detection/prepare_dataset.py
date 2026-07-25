"""Build a YOLO-format phone-detection dataset from public data + our own frames.

Takes the sources in `sources.py` (COCO JSON, downloaded from the Hub) plus any
frames we annotated ourselves (already YOLO .txt) and merges them into the
directory layout Ultralytics expects, with a `data.yaml` pointing at it.

    # Fetch the public set and build a dataset from it:
    python -m src.detection.prepare_dataset

    # Public data + our own annotated frames (the combination that matters):
    python -m src.detection.prepare_dataset --own-dir data/detection/raw_frames

    # Inspect the plan without downloading or writing:
    python -m src.detection.prepare_dataset --dry-run

Two flavours of public source are read automatically: COCO-JSON datasets from the
Hugging Face Hub (downloaded on demand) and any Roboflow YOLO export already
extracted under `--cache-dir` by `roboflow_import.py`. Roboflow sets label people
and calculators as well as phones, so their classes are remapped onto ours and
anything unrecognised is dropped.

Design choices worth defending in an interview:

* **One class, `phone`.** The measured problem is phone recall; a `calculator` or
  `book` head we never read would only dilute the gradient and the mAP. Classes
  live in `CLASSES` so adding one later is a single edit, not a refactor.
* **Our own frames split by CLIP, never by frame.** Identical to the grouped
  split in `build_dataset.py`, and for the same reason: consecutive frames from
  one clip are near-duplicates, so letting them straddle train/val leaks the
  answer and reports a mAP we haven't earned. Public images have no clip, so they
  split randomly under a fixed seed.
* **`source` is recorded per image.** Knowing which images came from where lets us
  evaluate on our-own-frames only -- the number that actually predicts live
  performance. A mAP averaged over clean public crops would flatter us.
* **Empty COCO categories are dropped.** The public set carries a typo'd
  `mobuile_phonw` category holding no boxes; emitting it as a class would give
  the model a head that can never be right.
"""

import argparse
import csv
import json
import os
import random
import shutil

import yaml

from src.detection.sources import SOURCES

# The classes the detector learns. Index = YOLO class id.
CLASSES = ["phone"]

# Public datasets name the same thing differently; map their labels onto ours.
# Anything not in here is skipped rather than silently folded into class 0.
# The Roboflow sets carry 'person', 'face' and 'calculator' too -- deliberately
# absent, so those boxes are dropped instead of becoming phantom phones.
CLASS_ALIASES = {
    "mobile_phone": "phone",
    "cell phone": "phone",
    "cellphone": "phone",
    "phone": "phone",
    "mobile": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",
    "handphone": "phone",
    "hp": "phone",
}

# Roboflow exports name their splits like this.
YOLO_SPLIT_DIRS = ("train", "valid", "test")


def coco_to_yolo_boxes(coco_path, image_dir):
    """Convert one COCO JSON into {image_filename: [yolo label lines]}.

    COCO stores boxes as absolute [x_min, y_min, width, height]; YOLO wants
    class-id plus centre-x, centre-y, width, height each normalised to [0, 1].
    """
    with open(coco_path) as handle:
        coco = json.load(handle)

    # Which category ids actually carry boxes -- drops the empty typo category.
    used_category_ids = {ann["category_id"] for ann in coco.get("annotations", [])}

    category_to_class = {}
    for category in coco.get("categories", []):
        if category["id"] not in used_category_ids:
            continue
        our_name = CLASS_ALIASES.get(category["name"].strip().lower())
        if our_name in CLASSES:
            category_to_class[category["id"]] = CLASSES.index(our_name)

    images = {img["id"]: img for img in coco.get("images", [])}
    labels = {img["file_name"]: [] for img in coco.get("images", [])}

    skipped = 0
    for ann in coco.get("annotations", []):
        class_id = category_to_class.get(ann["category_id"])
        if class_id is None:
            skipped += 1
            continue

        image = images.get(ann["image_id"])
        if image is None:
            skipped += 1
            continue

        img_w, img_h = image["width"], image["height"]
        x, y, w, h = ann["bbox"]

        # Normalise to centre form, clamping so a box that pokes over the edge
        # (common in Roboflow exports after augmentation) can't produce an
        # out-of-range coordinate that Ultralytics rejects at load time.
        cx = min(max((x + w / 2) / img_w, 0.0), 1.0)
        cy = min(max((y + h / 2) / img_h, 0.0), 1.0)
        nw = min(max(w / img_w, 0.0), 1.0)
        nh = min(max(h / img_h, 0.0), 1.0)
        if nw <= 0 or nh <= 0:
            skipped += 1
            continue

        labels[image["file_name"]].append(
            f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )

    # Only keep images that exist on disk and carry at least one box. Images with
    # no phone are valid background examples, but the public set is all-positive;
    # our own frames are where deliberate negatives come from.
    present = {
        name: lines
        for name, lines in labels.items()
        if lines and os.path.exists(os.path.join(image_dir, name))
    }
    return present, skipped


def collect_public(source_key, cache_dir, dry_run):
    """Download a Hub COCO dataset and convert it. Returns [(img_path, lines)]."""
    source = SOURCES[source_key]
    if dry_run:
        print(f"  would download {source['repo_id']} ({source['size']})")
        return []

    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=source["repo_id"], repo_type="dataset",
        local_dir=os.path.join(cache_dir, source_key),
    )

    items = []
    total_skipped = 0
    # The export ships train/valid/test subfolders, each with its own COCO file.
    # We re-split ourselves, so the original split boundaries don't matter here.
    for split in ("train", "valid", "test"):
        image_dir = os.path.join(root, split)
        coco_path = os.path.join(image_dir, "_annotations.coco.json")
        if not os.path.exists(coco_path):
            continue
        labels, skipped = coco_to_yolo_boxes(coco_path, image_dir)
        total_skipped += skipped
        for name, lines in labels.items():
            items.append((os.path.join(image_dir, name), lines))

    print(f"  {source_key}: {len(items)} annotated images "
          f"({total_skipped} annotations skipped)")
    return items


def remap_label_lines(lines, source_names):
    """Rewrite YOLO label lines from a source's class list onto ours.

    `source_names` is the upstream `names:` list from its data.yaml, so index 2
    might mean 'phone' there and must become our index 0. Any class we don't
    recognise (person, face, calculator) is dropped -- training a head we never
    read would only dilute the gradient and the reported mAP.

    Returns (kept_lines, n_dropped).
    """
    # Upstream index -> our class id, computed once per dataset.
    index_map = {}
    for index, name in enumerate(source_names):
        our_name = CLASS_ALIASES.get(str(name).strip().lower())
        if our_name in CLASSES:
            index_map[index] = CLASSES.index(our_name)

    kept = []
    dropped = 0
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            source_index = int(float(parts[0]))
        except ValueError:
            continue

        our_id = index_map.get(source_index)
        if our_id is None:
            dropped += 1
            continue
        kept.append(" ".join([str(our_id)] + parts[1:5]))

    return kept, dropped


def read_export_class_names(export_root):
    """The upstream `names:` list from a YOLO export's data.yaml."""
    data_yaml = os.path.join(export_root, "data.yaml")
    if not os.path.exists(data_yaml):
        raise SystemExit(
            f"No data.yaml in {export_root} -- cannot tell which class index means "
            f"'phone'. Re-download with: python -m src.detection.roboflow_import"
        )
    with open(data_yaml) as handle:
        config = yaml.safe_load(handle) or {}

    names = config.get("names")
    if isinstance(names, dict):  # some exports use {0: 'phone', 1: ...}
        names = [names[key] for key in sorted(names)]
    if not names:
        raise SystemExit(f"data.yaml in {export_root} has no 'names' list.")
    return list(names)


def collect_yolo_export(export_root):
    """Read an extracted Roboflow YOLO export.

    Returns (positives, backgrounds), each [(image_path, our_label_lines)].
    An image whose boxes were all dropped becomes a *background* rather than
    being discarded: negatives teach the detector what isn't a phone, but they
    have to be rationed (see cap_backgrounds) or they swamp the positives.
    """
    names = read_export_class_names(export_root)

    positives, backgrounds = [], []
    total_dropped = 0

    for split in YOLO_SPLIT_DIRS:
        image_dir = os.path.join(export_root, split, "images")
        label_dir = os.path.join(export_root, split, "labels")
        if not os.path.isdir(image_dir):
            continue

        for name in sorted(os.listdir(image_dir)):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            label_path = os.path.join(label_dir, os.path.splitext(name)[0] + ".txt")
            raw = []
            if os.path.exists(label_path):
                with open(label_path) as handle:
                    raw = [line.strip() for line in handle if line.strip()]

            kept, dropped = remap_label_lines(raw, names)
            total_dropped += dropped
            entry = (os.path.join(image_dir, name), kept)
            (positives if kept else backgrounds).append(entry)

    print(f"    upstream classes {names} -> kept {len(positives)} images with "
          f"phones, {len(backgrounds)} background, {total_dropped} boxes dropped")
    return positives, backgrounds


def cap_backgrounds(positives, backgrounds, ratio, seed):
    """Keep at most `ratio` backgrounds per positive image.

    Roboflow's proctoring sets label people and calculators too, so many images
    contain no phone at all. Including all of them would leave the detector
    trained mostly on absence and bias it toward predicting nothing -- the exact
    failure we are trying to fix. A ~10% sprinkle is the usual guidance.
    """
    limit = int(len(positives) * ratio)
    if len(backgrounds) <= limit:
        return backgrounds
    sampled = list(backgrounds)
    random.Random(seed).shuffle(sampled)
    return sampled[:limit]


def clip_of(filename):
    """'phone_001_f00030.jpg' -> 'phone_001', so a clip stays on one side of the split."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    marker = stem.rfind("_f")
    return stem[:marker] if marker != -1 else stem


def collect_own(own_dir, labels_dir):
    """Our annotated frames: [(img_path, lines, clip_id)].

    A .txt that exists but is empty is a deliberate background frame (phone
    present in the clip but not visible here) and is kept -- backgrounds teach the
    detector what *isn't* a phone. An image with no .txt at all is simply not yet
    annotated, and is skipped.
    """
    labels_dir = labels_dir or own_dir
    items = []
    unannotated = 0

    for name in sorted(os.listdir(own_dir)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        label_path = os.path.join(labels_dir, os.path.splitext(name)[0] + ".txt")
        if not os.path.exists(label_path):
            unannotated += 1
            continue
        with open(label_path) as handle:
            lines = [line.strip() for line in handle if line.strip()]
        items.append((os.path.join(own_dir, name), lines, clip_of(name)))

    print(f"  own frames: {len(items)} annotated, {unannotated} not yet annotated")
    return items


def split_public(items, val_fraction, seed):
    """Random split -- public images have no clip structure to respect."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - val_fraction))
    return shuffled[:cut], shuffled[cut:]


def split_own_by_clip(items, val_fraction, seed):
    """Group split: every frame of a clip lands in the same side. See module docstring."""
    clips = sorted({clip for _, _, clip in items})
    shuffled = list(clips)
    random.Random(seed).shuffle(shuffled)

    # At least one clip held out whenever there's more than one to hold out.
    n_val = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    val_clips = set(shuffled[:n_val])

    train = [(p, l) for p, l, c in items if c not in val_clips]
    val = [(p, l) for p, l, c in items if c in val_clips]
    return train, val, sorted(val_clips)


def write_split(items, out_root, split, source_tag, manifest):
    """Copy images and write their .txt labels into the YOLO layout."""
    image_dir = os.path.join(out_root, "images", split)
    label_dir = os.path.join(out_root, "labels", split)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for image_path, lines in items:
        # Prefix with the source so a public filename can never collide with ours.
        name = f"{source_tag}__{os.path.basename(image_path)}"
        shutil.copy2(image_path, os.path.join(image_dir, name))

        stem = os.path.splitext(name)[0]
        with open(os.path.join(label_dir, stem + ".txt"), "w") as handle:
            handle.write("\n".join(lines) + ("\n" if lines else ""))

        manifest.append({"file": name, "split": split, "source": source_tag,
                         "boxes": len(lines)})


def write_data_yaml(out_root, path):
    """The file Ultralytics reads. Absolute path avoids cwd surprises on Windows."""
    with open(path, "w") as handle:
        handle.write("# Generated by src/detection/prepare_dataset.py -- do not edit by hand.\n")
        handle.write(f"path: {os.path.abspath(out_root)}\n")
        handle.write("train: images/train\n")
        handle.write("val: images/val\n\n")
        handle.write(f"nc: {len(CLASSES)}\n")
        handle.write(f"names: {CLASSES}\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-root", default=os.path.join("data", "detection", "dataset"))
    parser.add_argument("--cache-dir", default=os.path.join("data", "detection", "external"))
    parser.add_argument("--own-dir", default=None,
                        help="folder of our own annotated frames (images + .txt)")
    parser.add_argument("--own-labels-dir", default=None,
                        help="where the .txt files are, if not alongside the images")
    parser.add_argument("--no-public", action="store_true",
                        help="build from our own frames only")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--background-ratio", type=float, default=0.1,
                        help="max phone-free images per positive, from Roboflow "
                             "exports (default 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Collecting sources")
    public_items = []
    if not args.no_public:
        for key, source in SOURCES.items():
            if source["kind"] == "hf_coco":
                public_items.extend(collect_public(key, args.cache_dir, args.dry_run))

            elif source["kind"] == "roboflow":
                # Only present if the user has fetched it with roboflow_import.
                export_root = os.path.join(args.cache_dir, key)
                if not os.path.isdir(export_root):
                    continue
                print(f"  {key}:")
                positives, backgrounds = collect_yolo_export(export_root)
                kept_backgrounds = cap_backgrounds(
                    positives, backgrounds, args.background_ratio, args.seed
                )
                if len(kept_backgrounds) < len(backgrounds):
                    print(f"    capped backgrounds {len(backgrounds)} -> "
                          f"{len(kept_backgrounds)} ({args.background_ratio:.0%} of positives)")
                public_items.extend(positives + kept_backgrounds)

    own_items = []
    if args.own_dir:
        if not os.path.isdir(args.own_dir):
            raise SystemExit(f"--own-dir not found: {args.own_dir}")
        own_items = collect_own(args.own_dir, args.own_labels_dir)

    if not public_items and not own_items and not args.dry_run:
        raise SystemExit(
            "Nothing to build. Either allow public data (drop --no-public) or "
            "annotate frames first (python -m src.detection.extract_frames)."
        )

    public_train, public_val = split_public(public_items, args.val_fraction, args.seed)
    own_train, own_val, val_clips = split_own_by_clip(own_items, args.val_fraction, args.seed)

    print("\nPlanned split")
    print(f"  public : {len(public_train):>5} train  {len(public_val):>5} val")
    print(f"  own    : {len(own_train):>5} train  {len(own_val):>5} val", end="")
    print(f"   (val clips: {', '.join(val_clips)})" if val_clips else "")
    print(f"  TOTAL  : {len(public_train) + len(own_train):>5} train  "
          f"{len(public_val) + len(own_val):>5} val")

    if args.dry_run:
        print("\nDry run -- nothing written.")
        return

    if os.path.exists(args.out_root):
        shutil.rmtree(args.out_root)  # rebuild clean; stale files silently skew mAP

    manifest = []
    write_split(public_train, args.out_root, "train", "public", manifest)
    write_split(public_val, args.out_root, "val", "public", manifest)
    write_split(own_train, args.out_root, "train", "own", manifest)
    write_split(own_val, args.out_root, "val", "own", manifest)

    manifest_path = os.path.join(args.out_root, "manifest.csv")
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "split", "source", "boxes"])
        writer.writeheader()
        writer.writerows(manifest)

    yaml_path = os.path.join(args.out_root, "data.yaml")
    write_data_yaml(args.out_root, yaml_path)

    print(f"\nWrote {len(manifest)} images -> {args.out_root}/")
    print(f"  {yaml_path}")
    if not own_items:
        print("\nNOTE: public images only. Expect this to underperform on our webcam\n"
              "      footage -- see the domain-gap caveat in src/detection/sources.py.")
    print(f"\nNext: python -m src.detection.train_detector --data {yaml_path}")


if __name__ == "__main__":
    main()
