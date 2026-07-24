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
python -m src.data.record_session --source clips/phone_002.mp4 --label phone
```

> The video files themselves are **gitignored** (they're your own footage —
> private and large). Only this README is tracked so the folder stays in the repo.
