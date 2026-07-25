# clips/

Drop your **raw recorded video clips** here (`.mp4`, `.avi`, ...) for feature
extraction. One clip = one behavior, ~10 seconds each (see
[`../evaluation/README.md`](../evaluation/README.md) for guidance).

Suggested naming — put the behavior in the filename so it's easy to track:

```
clips/
├── normal_001.mp4
├── looking_away_003.mp4
├── phone_002.mp4
└── multiple_people_001.mp4
```

Then turn a clip into a labeled feature CSV (lands in `data/sessions/`):

```bash
python -m src.data.record_session --source clips/phone_002.mp4 --labels phone
```

Because each clip here is a **single behavior**, the filename prefix already
determines the label — so you can label the whole folder in one shot instead of
running the command per clip:

```bash
python -m src.data.label_clips --dry-run   # preview the 0/1 each clip will get
python -m src.data.label_clips             # extract features + labels for all clips
```

> The video files themselves are **gitignored** (they're your own footage —
> private and large). Only this README is tracked so the folder stays in the repo.
