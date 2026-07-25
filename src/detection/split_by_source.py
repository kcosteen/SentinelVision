"""Re-split a Roboflow export so train and val don't share a source video.

The `online-proctoring-system` export ships train/valid/test splits that are
badly leaky: one source video (`final-2_mp4`) makes up ~87% of the data AND
supplies 100% of both valid and test. Training on it and validating on frames
from the same video of the same person in the same room would report a mAP that
says nothing about whether the model works on anyone else.

    # See the plan without writing anything:
    python -m src.detection.split_by_source --dry-run

    # Write the file lists + data.yaml:
    python -m src.detection.split_by_source

So this regroups by *source*: every frame of the dominant video goes to train,
and the remaining sources -- different people, different rooms -- become val.
The question the val score then answers is the one worth asking: "trained on one
person, does it detect phones for someone it has never seen?"

**Why file lists instead of copying.** Ultralytics accepts a .txt of image paths
wherever it accepts a directory, and finds labels by swapping `/images/` for
`/labels/`. That avoids duplicating 1.4 GB of JPEGs to express a different split.

**The honest caveat this cannot fix.** Val is ~3.3k images from a handful of
sources, and train is essentially one person. A good score here is real evidence
of generalisation; a bad one may just mean the val sources are unusually hard.
Neither is a substitute for more diverse training data.
"""

import argparse
import os
import re

import yaml

# Strip the Roboflow hash and trailing frame number to recover a source id:
# "final-2_mp4-6049_jpg.rf.<hash>.jpg" -> "final-2_mp4"
SOURCE_PATTERN = re.compile(r"^(.*?)[-_]?(\d+)_jpg\.rf\.")


def source_of(filename):
    """Best-effort source id for one exported frame."""
    match = SOURCE_PATTERN.match(filename)
    if match and match.group(1):
        return match.group(1)
    # Files that don't fit the pattern are lumped together rather than each
    # becoming its own "source" and silently scattering across the split.
    return "_misc"


def scan_export(export_root, splits):
    """{source: [absolute image paths]} across every original split."""
    by_source = {}
    for split in splits:
        image_dir = os.path.join(export_root, split, "images")
        if not os.path.isdir(image_dir):
            continue
        with os.scandir(image_dir) as entries:
            for entry in entries:
                if not entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                by_source.setdefault(source_of(entry.name), []).append(
                    os.path.abspath(entry.path)
                )
    return by_source


def choose_val_sources(by_source, val_sources, max_train_share):
    """Pick which sources become validation.

    Default rule: whichever single source dominates the dataset becomes the
    training set, everything else validates. That is the split that tests
    generalisation rather than memorisation.
    """
    if val_sources:
        return set(val_sources)

    largest = max(by_source, key=lambda s: len(by_source[s]))
    total = sum(len(v) for v in by_source.values())
    share = len(by_source[largest]) / total

    if share < max_train_share:
        raise SystemExit(
            f"No single source dominates ({largest} is only {share:.0%}). "
            f"This heuristic assumes one does -- pass --val-sources explicitly."
        )
    return set(by_source) - {largest}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--export-root",
                        default=os.path.join("data", "detection", "external",
                                             "roboflow_online_proctoring"))
    parser.add_argument("--out-dir", default=None,
                        help="where to write the lists (default: the export root)")
    parser.add_argument("--val-sources", nargs="*", default=None,
                        help="source ids for val (default: everything but the largest)")
    parser.add_argument("--max-train-share", type=float, default=0.5,
                        help="require the dominant source to exceed this share")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or args.export_root
    data_yaml = os.path.join(args.export_root, "data.yaml")
    if not os.path.exists(data_yaml):
        raise SystemExit(f"No data.yaml in {args.export_root}")

    names = (yaml.safe_load(open(data_yaml)) or {}).get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    by_source = scan_export(args.export_root, ("train", "valid", "test"))
    if not by_source:
        raise SystemExit(f"No images found under {args.export_root}")

    val_sources = choose_val_sources(by_source, args.val_sources, args.max_train_share)

    train_paths, val_paths = [], []
    for source, paths in sorted(by_source.items()):
        (val_paths if source in val_sources else train_paths).extend(paths)

    print(f"{'source':<38}{'images':>8}  split")
    print("-" * 56)
    for source, paths in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
        where = "val" if source in val_sources else "TRAIN"
        print(f"{source or '(blank)':<38}{len(paths):>8}  {where}")

    print(f"\ntrain: {len(train_paths)} images")
    print(f"val  : {len(val_paths)} images from {len(val_sources)} sources")
    print(f"classes ({len(names)}): {names}")

    if args.dry_run:
        print("\nDry run -- nothing written.")
        return

    train_list = os.path.join(out_dir, "train_by_source.txt")
    val_list = os.path.join(out_dir, "val_by_source.txt")
    for path, entries in ((train_list, train_paths), (val_list, val_paths)):
        with open(path, "w") as handle:
            handle.write("\n".join(entries) + "\n")

    out_yaml = os.path.join(out_dir, "data_by_source.yaml")
    with open(out_yaml, "w") as handle:
        handle.write("# Generated by src/detection/split_by_source.py\n")
        handle.write("# Train and val share no source video -- see the module docstring.\n")
        handle.write(f"path: {os.path.abspath(args.export_root)}\n")
        handle.write(f"train: {os.path.abspath(train_list)}\n")
        handle.write(f"val: {os.path.abspath(val_list)}\n\n")
        handle.write(f"nc: {len(names)}\n")
        handle.write(f"names: {list(names)}\n")

    print(f"\nWrote:\n  {train_list}\n  {val_list}\n  {out_yaml}")
    print(f"\nNext: python -m src.detection.train_detector --data {out_yaml}")


if __name__ == "__main__":
    main()
