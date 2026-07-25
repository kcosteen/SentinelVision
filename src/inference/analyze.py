"""
Run the full proctoring pipeline on a video and predict behaviours.

This is the glue the demo app sits on: raw video -> per-frame features
(FeatureExtractor) -> sliding windows (build_dataset) -> per-behaviour predictions
from the trained model (models/behavior_model.joblib). It reuses the exact same
feature code the model was trained on, so what the demo scores matches training.

    from src.inference.analyze import analyze_video
    windows, summary = analyze_video("clips/looking_away_001.mp4")
"""

import os

import cv2
import joblib
import numpy as np
import pandas as pd

from src.data.build_dataset import FEATURE_FIELDS, window_clip
from src.data.feature_extractor import FeatureExtractor
from src.data.labels import LABEL_COLUMNS

DEFAULT_MODEL_PATH = os.path.join("models", "behavior_model.joblib")


def load_model(path=DEFAULT_MODEL_PATH):
    """Load the {behaviour: sklearn pipeline} dict trained by src.models.train."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.models.train` first."
        )
    return joblib.load(path)


def extract_frame_features(video_path, target_fps=10.0, progress=None):
    """Per-frame features for a video, sampled down to ~target_fps for speed.

    Behaviours last seconds, so scoring ~10 frames/second instead of every frame
    keeps the demo responsive with no meaningful loss. `time_sec` is still derived
    from the true frame index, so the windows line up with real time.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = max(1, round(fps / target_fps)) if fps > 0 else 1

    extractor = FeatureExtractor()  # fresh per video: blink state must not carry over
    rows = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride == 0:
            measurements = extractor.extract(frame)
            time_sec = frame_index / fps if fps > 0 else frame_index / target_fps
            row = {"clip_id": "uploaded", "time_sec": round(time_sec, 3)}
            # Empty cells (no face that frame) -> NaN so the aggregates skip them.
            row.update({k: (np.nan if v == "" else v) for k, v in measurements.items()})
            # window_clip's aggregator reads label columns; inference has none, so 0.
            for col in LABEL_COLUMNS:
                row[col] = 0
            rows.append(row)
            if progress and total_frames:
                progress(min(frame_index / total_frames, 1.0))
        frame_index += 1

    cap.release()
    if progress:
        progress(1.0)
    return pd.DataFrame(rows)


def _flagged_intervals(windows, behaviour):
    """Merge consecutive flagged windows into (start_sec, end_sec) time ranges."""
    intervals = []
    flagged = windows[windows[f"{behaviour}_flag"] == 1]
    for _, row in flagged.iterrows():
        start, end = row["window_start_sec"], row["window_end_sec"]
        if intervals and start <= intervals[-1][1] + 1e-6:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return intervals


def summarize(windows, model):
    """Per-behaviour rollup: how often it fired, peak confidence, when."""
    n = len(windows)
    summary = {}
    for behaviour in model:
        flags = windows[f"{behaviour}_flag"]
        probs = windows[f"{behaviour}_prob"]
        flagged = int(flags.sum())
        summary[behaviour] = {
            "flagged_windows": flagged,
            "total_windows": n,
            "flagged_fraction": flagged / n if n else 0.0,
            "max_prob": float(probs.max()) if n else 0.0,
            "intervals": _flagged_intervals(windows, behaviour),
        }
    return summary


def overall_verdict(summary):
    """A single headline from the per-behaviour fractions (demo-friendly, honest)."""
    worst = max((b["flagged_fraction"] for b in summary.values()), default=0.0)
    if worst >= 0.5:
        return "High risk", "A behaviour was flagged across most of the clip."
    if worst >= 0.15:
        return "Suspicious", "One or more behaviours were flagged intermittently."
    return "Appears normal", "No behaviour was flagged for a sustained stretch."


def analyze_video(video_path, window_sec=2.0, step_sec=1.0, target_fps=10.0,
                  model=None, progress=None):
    """Score a video. Returns (windows DataFrame with predictions, summary dict)."""
    model = model if model is not None else load_model()

    frames = extract_frame_features(video_path, target_fps=target_fps, progress=progress)
    if frames.empty:
        raise ValueError("No frames could be read from the video.")

    windows = pd.DataFrame(window_clip(frames, window_sec, step_sec, min_frames=1))
    if windows.empty:
        raise ValueError("Video too short to form a single analysis window.")

    features = windows[FEATURE_FIELDS]
    for behaviour, pipeline in model.items():
        windows[f"{behaviour}_prob"] = pipeline.predict_proba(features)[:, 1]
        windows[f"{behaviour}_flag"] = pipeline.predict(features).astype(int)

    return windows, summarize(windows, model)
