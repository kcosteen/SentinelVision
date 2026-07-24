# Feature logging (Phase 1 data collection)

Turns recorded sessions into a **training dataset** by writing per-frame features
to CSV. These are the same signals the live app reacts to — captured as numbers a
model can learn from, instead of hard-coded rules.

## Workflow

1. **Record** short clips (see [`../../evaluation/README.md`](../../evaluation/README.md)
   for clip-length guidance) — or run live.
2. **Extract features** to CSV:

   ```bash
   # From a recorded clip that is one behavior end-to-end:
   python -m src.data.record_session --source clips/clip_007.mp4 --label phone

   # Or live from the webcam, with a preview window:
   python -m src.data.record_session --source 0 --show --label normal
   ```

3. Each run writes one CSV to `data/sessions/`. Collect many, then concatenate
   them into a single training table.
4. **Label**: a single-behavior clip is labeled in one shot with `--label`. For a
   mixed session, leave it blank and fill the `label` column by time segment
   afterwards (use `time_sec` / `frame_index`).
5. **Train (Phase 1)**: load the combined CSV, aggregate rows into sliding time
   windows (e.g. blink rate and gaze variance over the last 2 seconds), and train
   a classifier on those windows.

## Columns (schema)

See [`../../data/sample_features.csv`](../../data/sample_features.csv) for a
runnable example of the format.

| Column | Meaning |
|---|---|
| `session_id`, `frame_index`, `time_sec` | Which session, which frame, and when. |
| `face_count` | Faces detected (0 = absent, >1 = multiple people). |
| `gaze_ratio` | Continuous 0–1 horizontal gaze (0 = left corner, 1 = right). |
| `gaze_direction` | LEFT / RIGHT / CENTER derived from `gaze_ratio`. |
| `ear` | Eye Aspect Ratio (eye openness). |
| `eyes_closed` | 1 if `ear` < 0.20. |
| `blink_total` | Cumulative blink count so far in the session. |
| `head_pitch/yaw/roll` | Head orientation in degrees. |
| `person_count` | People detected by YOLO. |
| `phone_detected`, `phone_conf` | Phone present (1/0) and its confidence. |
| `book_detected` | Book present (1/0). |
| `label` | Ground-truth behavior (you fill this in). |

## Notes

- Features are computed on a **mirror-flipped** frame, matching the live app.
- **Empty cells** mean no face was detected that frame, so the face-mesh features
  (gaze, EAR, head pose) are unavailable — that absence is itself a useful signal.
- `data/` is **gitignored**: recordings are your own, potentially private, and can
  be large. Only the tiny `sample_features.csv` schema example is committed.
