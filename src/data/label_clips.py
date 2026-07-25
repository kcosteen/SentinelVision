"""
Batch-label every clip in clips/ from its filename, extracting features to CSV.

Single-behavior clips are named `<behavior>_<NNN>.<ext>` -- e.g. looking_away_001.mp4,
normal_003.mp4, phone_002.mp4. This reads the `<behavior>` prefix, turns it into
the 0/1 `label_*` columns (via the SAME parse_labels the CLI uses), and runs
feature extraction once per clip into data/sessions/<clip>.csv.

    # See the label each clip WOULD get -- fast, no models loaded:
    python -m src.data.label_clips --dry-run

    # Actually extract features for every clip (skips ones already done):
    python -m src.data.label_clips

    # Re-do clips whose CSV already exists:
    python -m src.data.label_clips --overwrite

Why filename-based? Right now each clip is a SINGLE behavior end-to-end, so the
name fully determines the label -- no per-frame timeline editing needed. The
prefix must be a known behavior (or 'normal'); an unrecognized name fails loudly
so a mis-named file can't silently poison the dataset. When you later record
clips with co-occurring behaviors, label those by hand with `record_session`
(`--labels looking_away,phone`) instead of this batch tool.
"""

import argparse
import os
import re

from src.data.labels import LABEL_COLUMNS, parse_labels

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Strip a trailing _<digits> so multi-word behaviors survive intact:
# "looking_away_001" -> "looking_away", "normal_003" -> "normal".
_PREFIX = re.compile(r"^(.*)_\d+$")


def behavior_from_filename(stem):
    """Clip stem -> behavior name (everything before the trailing _NNN), or None."""
    match = _PREFIX.match(stem)
    return match.group(1) if match else None


def find_clips(clips_dir):
    """Sorted list of video filenames in `clips_dir`."""
    return [
        name
        for name in sorted(os.listdir(clips_dir))
        if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS
    ]


def build_plan(clips):
    """[(clip_name, stem, label_columns)], failing loudly on an unreadable name."""
    plan = []
    for name in clips:
        stem = os.path.splitext(name)[0]
        behavior = behavior_from_filename(stem)
        if behavior is None:
            raise SystemExit(
                f"Cannot read a behavior from '{name}'. "
                f"Expected <behavior>_<NNN>.<ext>, e.g. looking_away_001.mp4."
            )
        try:
            label_columns = parse_labels(behavior)
        except ValueError as error:
            raise SystemExit(f"{name}: {error}")
        plan.append((name, stem, label_columns))
    return plan


def print_table(plan):
    """Print each clip with the 0/1 it gets in every label column."""
    behaviors = [col.replace("label_", "") for col in LABEL_COLUMNS]
    clip_width = max([len("clip")] + [len(name) for name, _, _ in plan])

    header = f"{'clip':<{clip_width}}  " + "  ".join(behaviors)
    print(header)
    print("-" * len(header))
    for name, _, label_columns in plan:
        cells = "  ".join(
            f"{label_columns[col]:<{len(behavior)}}"
            for col, behavior in zip(LABEL_COLUMNS, behaviors)
        )
        print(f"{name:<{clip_width}}  {cells}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clips-dir", default="clips", help="folder of raw clips")
    parser.add_argument(
        "--out-dir", default=os.path.join("data", "sessions"),
        help="where the per-clip feature CSVs are written",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the label each clip would get, without extracting features",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="re-extract clips whose CSV already exists (default: skip them)",
    )
    args = parser.parse_args()

    clips = find_clips(args.clips_dir)
    if not clips:
        raise SystemExit(f"No video clips found in {args.clips_dir}/")

    # Resolve every label first so a mis-named file fails before any slow work.
    plan = build_plan(clips)
    print_table(plan)

    if args.dry_run:
        print("\nDry run -- no features extracted. Drop --dry-run to process.")
        return

    # Imported here (not at top) so --dry-run stays instant: this pulls in the
    # heavy YOLO / MediaPipe stack that FeatureExtractor needs.
    from src.data.record_session import record_features

    os.makedirs(args.out_dir, exist_ok=True)
    processed = skipped = 0
    for name, stem, label_columns in plan:
        out_path = os.path.join(args.out_dir, f"{stem}.csv")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"skip  {name}  (exists: {out_path})")
            skipped += 1
            continue

        source = os.path.join(args.clips_dir, name)
        print(f"label {name} -> {stem}.csv")
        rows = record_features(source, out_path, stem, label_columns)
        print(f"      {rows} rows")
        processed += 1

    print(f"\nDone. {processed} extracted, {skipped} skipped, {len(plan)} total.")


if __name__ == "__main__":
    main()
