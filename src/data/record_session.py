"""
Record a session's per-frame features to CSV (Phase 1 data collection).

Point it at your webcam or a recorded clip and it writes one row per frame using
FeatureExtractor. Because proctoring is multi-label (behaviors co-occur), stamp
the clip with `--labels` listing every behavior that is true for it — one or
several. Leave it blank (or "normal") for a clean, no-abnormal-behavior clip.

Examples
--------
    # Live webcam, show the window, clip is normal behavior
    python -m src.data.record_session --source 0 --show --labels normal

    # A clip where the person is looking away AND on their phone (two labels!)
    python -m src.data.record_session --source clips/phone_002.mp4 --labels looking_away,phone
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2

from src.data.feature_extractor import FeatureExtractor, MEASUREMENT_FIELDS
from src.data.labels import BEHAVIOR_LABELS, LABEL_COLUMNS, parse_labels

CSV_FIELDS = ["session_id", "frame_index", "time_sec"] + MEASUREMENT_FIELDS + LABEL_COLUMNS


def parse_source(source):
    """"0" -> webcam index 0; anything else is treated as a file path."""
    return int(source) if source.isdigit() else source


def record_features(source, out_path, session_id, label_columns, show=False):
    """Extract per-frame features from one source into a labeled CSV.

    Runs FeatureExtractor over every frame of `source` (a webcam index string
    like "0", or a video path) and writes one row per frame to `out_path`,
    stamping each row with `label_columns` ({label_x: 0/1, ...}). Returns the
    number of rows written.

    This is the shared core used by both the CLI (`main`) and the batch labeler
    (`src.data.label_clips`), so both paths produce byte-identical CSVs.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(parse_source(source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    extractor = FeatureExtractor()

    rows_written = 0
    start = time.time()

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            measurements = extractor.extract(frame)

            # For a video file, derive time from the frame index and FPS; for a
            # live webcam (FPS often reported as 0), fall back to wall-clock time.
            if fps and fps > 0:
                time_sec = round(frame_index / fps, 3)
            else:
                time_sec = round(time.time() - start, 3)

            row = {
                "session_id": session_id,
                "frame_index": frame_index,
                "time_sec": time_sec,
            }
            row.update(measurements)
            row.update(label_columns)  # same label(s) stamped on every frame
            writer.writerow(row)
            rows_written += 1

            if show:
                display = cv2.flip(frame, 1)
                cv2.putText(
                    display,
                    f"frame {frame_index}  faces {measurements['face_count']}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.imshow("Recording features", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    return rows_written


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", default="0",
        help="webcam index (e.g. 0) or path to a video file",
    )
    parser.add_argument(
        "--out", default=None,
        help="output CSV path (default: data/sessions/<session_id>.csv)",
    )
    parser.add_argument(
        "--session-id", default=None,
        help="identifier stored in each row (default: a timestamp)",
    )
    parser.add_argument(
        "--labels", default="",
        help="comma-separated behaviors true for THIS clip (they can co-occur), "
             "e.g. 'looking_away,phone'. Use 'normal' or leave empty for none. "
             f"Valid: {BEHAVIOR_LABELS}",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="display the video while recording",
    )
    args = parser.parse_args()

    # Resolve labels up front so a typo fails before we open the camera.
    try:
        label_columns = parse_labels(args.labels)
    except ValueError as error:
        raise SystemExit(str(error))

    session_id = args.session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join("data", "sessions", f"{session_id}.csv")

    print(f"Recording features -> {out_path}  (press 'q' to stop)")
    rows_written = record_features(
        args.source, out_path, session_id, label_columns, show=args.show,
    )
    print(f"Done. Wrote {rows_written} rows to {out_path}")


if __name__ == "__main__":
    main()
