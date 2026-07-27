"""
Run the full proctoring pipeline on a video and flag behaviours.

This is the glue the demo app sits on: raw video -> per-frame features
(FeatureExtractor) -> per-frame events (ProctorAnalyzer) -> time windows -> a
summary of what was flagged and when.

    from src.inference.analyze import analyze_video
    windows, summary = analyze_video("clips/looking_away_001.mp4")

**Everything here runs on models and thresholds derived from PUBLIC data**: the
YOLOv8n fine-tuned on the open proctoring dataset, plus the head-yaw and EAR
thresholds swept against the Gourier and open/closed-eyes sets. Nothing depends
on privately recorded footage, so a fresh clone reproduces the demo exactly.

It used to score with a scikit-learn behaviour model (`behavior_model.joblib`)
trained on locally recorded clips. That made the demo unreproducible for anyone
else, and gave the phone class almost nothing real to learn from -- the detector
of the day found a phone in ~20% of frames, so the model leaned on head-down
posture as a proxy. Deciding with the same rules engine the live app uses keeps
one source of truth for "what counts as suspicious" across both entry points.
"""

import cv2
import numpy as np
import pandas as pd

from src.behavior.proctor_analyzer import ProctorAnalyzer
from src.data.feature_extractor import FeatureExtractor

# The analyzer speaks in human-readable event names; the demo reports the label
# vocabulary from src/data/labels.py. One map, so neither side drifts.
EVENT_TO_BEHAVIOUR = {
    "Looking away": "looking_away",
    "Phone detected": "phone",
    "Multiple people detected": "multiple_people",
    "No face detected": "absent",
}

BEHAVIOURS = list(EVENT_TO_BEHAVIOUR.values())


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
            row = {"time_sec": round(time_sec, 3)}
            # Empty cells (no face that frame) -> NaN so the aggregates skip them.
            row.update({k: (np.nan if v == "" else v) for k, v in measurements.items()})
            rows.append(row)
            if progress and total_frames:
                progress(min(frame_index / total_frames, 1.0))
        frame_index += 1

    cap.release()
    if progress:
        progress(1.0)
    return pd.DataFrame(rows)


def _events_for_frame(row, analyzer):
    """The events the live rules engine would raise for one frame of features.

    A fresh analyzer is used per frame deliberately: ProctorAnalyzer.analyze is
    pure, but add_score carries a 10-second cooldown meant for a live session.
    Here we want the raw per-frame verdict, and the cooldown is not applied.
    """
    phone_conf = row.get("phone_conf")
    objects = []
    if row.get("phone_detected") == 1:
        objects.append({
            "label": "cell phone",
            "confidence": 0.0 if pd.isna(phone_conf) else float(phone_conf),
        })

    face_count = row.get("face_count")
    face_count = 0 if pd.isna(face_count) else int(face_count)

    head_yaw = row.get("head_yaw")
    head_yaw = None if pd.isna(head_yaw) else float(head_yaw)

    gaze = row.get("gaze_direction")
    gaze = None if not isinstance(gaze, str) else gaze

    return analyzer.analyze(objects, face_count, gaze, head_yaw)


def flag_frames(frames):
    """Add a 0/1 column per behaviour, using the live app's own decision rules."""
    analyzer = ProctorAnalyzer()
    flags = {behaviour: [] for behaviour in BEHAVIOURS}

    for _, row in frames.iterrows():
        fired = {
            EVENT_TO_BEHAVIOUR[event]
            for event in _events_for_frame(row, analyzer)
            if event in EVENT_TO_BEHAVIOUR
        }
        for behaviour in BEHAVIOURS:
            flags[behaviour].append(int(behaviour in fired))

    out = frames.copy()
    for behaviour, values in flags.items():
        out[f"{behaviour}_fired"] = values
    return out


def to_windows(frames, window_sec=2.0, step_sec=1.0):
    """Roll per-frame flags into overlapping time windows.

    A window is flagged when ANY frame inside it fired, and its "probability" is
    the fraction of frames that did. That fraction is an honest confidence -- it
    says how much of the window the behaviour was actually present for -- not a
    calibrated posterior, and the demo labels it as such.
    """
    if frames.empty:
        return pd.DataFrame()

    start, end = float(frames["time_sec"].min()), float(frames["time_sec"].max())
    rows = []
    edge = start
    while edge <= max(end - window_sec, start):
        stop = edge + window_sec
        inside = frames[(frames["time_sec"] >= edge) & (frames["time_sec"] < stop)]
        if len(inside):
            row = {"window_start_sec": round(edge, 3), "window_end_sec": round(stop, 3)}
            for behaviour in BEHAVIOURS:
                fired = inside[f"{behaviour}_fired"]
                row[f"{behaviour}_prob"] = float(fired.mean())
                row[f"{behaviour}_flag"] = int(fired.any())
            rows.append(row)
        edge += step_sec

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
                  progress=None):
    """Score a video. Returns (windows DataFrame with flags, summary dict)."""
    frames = extract_frame_features(video_path, target_fps=target_fps, progress=progress)
    if frames.empty:
        raise ValueError("No frames could be read from the video.")

    windows = to_windows(flag_frames(frames), window_sec, step_sec)
    if windows.empty:
        raise ValueError("Video too short to form a single analysis window.")

    return windows, summarize(windows, BEHAVIOURS)
