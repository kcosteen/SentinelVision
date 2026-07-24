# Feature logging (Phase 1 data collection)

Turns recorded sessions into a **training dataset** by writing per-frame features
to CSV. These are the same signals the live app reacts to — captured as numbers a
model can learn from, instead of hard-coded rules.

## Workflow

1. **Record** short clips (see [`../../evaluation/README.md`](../../evaluation/README.md)
   for clip-length guidance) — or run live.
2. **Extract features** to CSV:

   ```bash
   # A clip that is one behavior end-to-end:
   python -m src.data.record_session --source clips/phone_002.mp4 --labels phone

   # Behaviors co-occur — list several (looking away AND on a phone):
   python -m src.data.record_session --source clips/multi_001.mp4 --labels looking_away,phone

   # Or live from the webcam, normal behavior, with a preview window:
   python -m src.data.record_session --source 0 --show --labels normal
   ```

3. Each run writes one CSV to `data/sessions/`. Collect many, then concatenate
   them into a single training table.
4. **Label (multi-label)**: proctoring behaviors co-occur, so each behavior has
   its own 0/1 column and a clip can switch on several via `--labels`. A
   single-behavior clip is labeled in one shot; for a mixed *timeline*, leave
   `--labels` blank and edit the `label_*` columns by time segment afterwards
   (use `time_sec` / `frame_index`).
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
| `label_looking_away`, `label_phone`, `label_multiple_people`, `label_absent` | Ground-truth behaviors (0/1). **Several can be 1 at once.** You set these via `--labels`. |

> **Features vs. labels:** columns like `phone_detected` are *features* — what the
> models measured. The `label_*` columns are the *targets* — the human-confirmed
> truth the classifier will learn to predict. They are deliberately kept separate.

## Notes

- Features are computed on a **mirror-flipped** frame, matching the live app.
- **Empty cells** mean no face was detected that frame, so the face-mesh features
  (gaze, EAR, head pose) are unavailable — that absence is itself a useful signal.
- `data/` is **gitignored**: recordings are your own, potentially private, and can
  be large. Only the tiny `sample_features.csv` schema example is committed.
