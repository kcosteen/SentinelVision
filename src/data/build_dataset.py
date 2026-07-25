"""
Turn the per-frame session CSVs into a windowed training table (Phase 1, step 2-3).

A single frame barely tells you anything; a *pattern over a few seconds* does
(eyes darting around, head swinging, a phone visible in half the frames). So we
slide a fixed-length window over each clip and summarise the frames inside it into
one row of aggregate features. That row -- not the raw frame -- is what the
behaviour model learns from.

    # Default: 2s windows, 1s step (50% overlap) -> data/dataset.csv
    python -m src.data.build_dataset

    # Longer context, no overlap:
    python -m src.data.build_dataset --window-sec 3 --step-sec 3

Design choices worth defending in an interview:

* **Aggregate mean AND std.** The mean says *where* a signal sat; the std says how
  much it *moved*. "Looking away" often shows up as high gaze/head std, not an
  unusual mean -- exactly the temporal pattern the hand-coded rules in
  `proctor_analyzer.py` cannot see.
* **`clip_id` travels with every row.** When you split into train/test later,
  split *by clip_id*, never by row. Two windows from the same clip are highly
  correlated; letting them straddle the split leaks the answer and inflates your
  scores. Grouping is the honest way.
* **No imputation here.** Windows with no detected face leave face-mesh features
  blank (NaN -> empty cell). Filling them in is a *training-time* decision that
  must be fit on the training split only (or it leaks test statistics), so it
  lives in the model pipeline, not in this dataset builder.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from src.data.labels import LABEL_COLUMNS

# Aggregate feature columns produced per window (order defines CSV order).
FEATURE_FIELDS = [
    "gaze_ratio_mean", "gaze_ratio_std",
    "frac_gaze_offcenter",
    "ear_mean", "ear_std",
    "eyes_closed_frac",
    "blink_rate",
    "head_pitch_mean", "head_pitch_std",
    "head_yaw_mean", "head_yaw_std",
    "head_roll_mean", "head_roll_std",
    "face_count_mean", "frac_no_face",
    "person_count_mean", "person_count_max",
    "phone_frac", "phone_conf_mean", "phone_conf_max",
    "book_frac",
]

META_FIELDS = ["clip_id", "window_index", "window_start_sec", "window_end_sec", "n_frames"]

OUTPUT_FIELDS = META_FIELDS + FEATURE_FIELDS + LABEL_COLUMNS


def read_sessions(sessions_dir):
    """Concatenate every session CSV, tagging each row with its clip_id.

    Empty cells (frames where no face was detected) are read as NaN, so the
    NaN-skipping pandas aggregates below simply ignore them.
    """
    paths = sorted(glob.glob(os.path.join(sessions_dir, "*.csv")))
    if not paths:
        raise SystemExit(f"No session CSVs found in {sessions_dir}/. Run label_clips first.")

    df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    df["clip_id"] = df["session_id"]
    return df


def window_starts(t_min, t_max, window_sec, step_sec):
    """Start times for full windows within [t_min, t_max].

    A clip shorter than one window yields a single window covering all of it, so
    short clips aren't silently dropped.
    """
    if t_max - t_min < window_sec:
        return [t_min]

    starts = []
    start = t_min
    while start + window_sec <= t_max + 1e-9:
        starts.append(start)
        start += step_sec
    return starts


def aggregate_window(win, clip_id, window_index, start, end):
    """Summarise the frames in one window (a DataFrame slice) into one row."""
    # Eye gaze off-centre: fraction of *measured* frames not labelled CENTER.
    directions = win["gaze_direction"].dropna()
    frac_offcenter = (directions != "CENTER").mean() if len(directions) else np.nan

    # Blinks are logged as a running total, so blinks *in this window* is the rise
    # across it; divide by the window span to get a rate (blinks/second).
    blinks = win["blink_total"].dropna()
    duration = end - start
    blink_rate = (blinks.max() - blinks.min()) / duration if len(blinks) and duration > 0 else 0.0

    row = {
        "clip_id": clip_id,
        "window_index": window_index,
        "window_start_sec": round(start, 3),
        "window_end_sec": round(end, 3),
        "n_frames": len(win),
        "gaze_ratio_mean": win["gaze_ratio"].mean(),
        "gaze_ratio_std": win["gaze_ratio"].std(ddof=0),
        "frac_gaze_offcenter": frac_offcenter,
        "ear_mean": win["ear"].mean(),
        "ear_std": win["ear"].std(ddof=0),
        "eyes_closed_frac": win["eyes_closed"].mean(),
        "blink_rate": blink_rate,
        "head_pitch_mean": win["head_pitch"].mean(),
        "head_pitch_std": win["head_pitch"].std(ddof=0),
        "head_yaw_mean": win["head_yaw"].mean(),
        "head_yaw_std": win["head_yaw"].std(ddof=0),
        "head_roll_mean": win["head_roll"].mean(),
        "head_roll_std": win["head_roll"].std(ddof=0),
        "face_count_mean": win["face_count"].mean(),
        "frac_no_face": (win["face_count"] == 0).mean(),
        "person_count_mean": win["person_count"].mean(),
        "person_count_max": int(win["person_count"].max()),
        "phone_frac": win["phone_detected"].mean(),
        "phone_conf_mean": win["phone_conf"].mean(),
        "phone_conf_max": win["phone_conf"].max(),
        "book_frac": win["book_detected"].mean(),
    }
    # A single-behaviour clip stamps every frame identically; max keeps the right
    # multi-label meaning ("this behaviour occurred somewhere in the window")
    # even once you start labelling mixed timelines.
    for col in LABEL_COLUMNS:
        row[col] = int(win[col].max())

    return row


def window_clip(clip_df, window_sec, step_sec, min_frames):
    """Slide over one clip's frames, yielding an aggregate row per window."""
    if clip_df.empty:
        return []

    clip_df = clip_df.sort_values("time_sec")
    clip_id = clip_df["clip_id"].iloc[0]
    times = clip_df["time_sec"]
    t_min, t_max = times.iloc[0], times.iloc[-1]

    rows = []
    for start in window_starts(t_min, t_max, window_sec, step_sec):
        end = start + window_sec
        # Half-open [start, end) so overlapping windows don't double-count a frame.
        win = clip_df[(times >= start) & (times < end)]
        if len(win) < min_frames:
            continue
        rows.append(aggregate_window(win, clip_id, len(rows), start, end))
    return rows


def build(sessions_dir, window_sec, step_sec, min_frames):
    """Read every session CSV; return (windowed DataFrame, list of clip_ids)."""
    df = read_sessions(sessions_dir)

    rows = []
    clip_ids = []
    for clip_id, clip_df in df.groupby("clip_id", sort=True):
        clip_ids.append(clip_id)
        rows.extend(window_clip(clip_df, window_sec, step_sec, min_frames))

    result = pd.DataFrame(rows, columns=OUTPUT_FIELDS).round(4)
    return result, clip_ids


def print_summary(result, clip_ids):
    print(f"Clips:   {len(clip_ids)}")
    print(f"Windows: {len(result)}")
    print("Positive windows per label:")
    for col in LABEL_COLUMNS:
        positives = int((result[col] == 1).sum())
        print(f"  {col:<26} {positives:>4} / {len(result)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sessions-dir", default=os.path.join("data", "sessions"))
    parser.add_argument("--out", default=os.path.join("data", "dataset.csv"))
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=1.0,
                        help="gap between window starts (default 1.0 = 50%% overlap)")
    parser.add_argument("--min-frames", type=int, default=1,
                        help="drop windows with fewer measured frames than this")
    args = parser.parse_args()

    result, clip_ids = build(args.sessions_dir, args.window_sec, args.step_sec, args.min_frames)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # na_rep="" keeps blank cells blank (no face -> no gaze/head reading).
    result.to_csv(args.out, index=False, na_rep="")

    print_summary(result, clip_ids)
    print(f"\nWrote {len(result)} windows -> {args.out}")


if __name__ == "__main__":
    main()
