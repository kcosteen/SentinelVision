"""
SentinelVision -- demo web app.

Upload an exam webcam clip; the app runs the full vision pipeline and shows what
it flagged and when. Run it with:

    streamlit run app.py

It scores frames through src/inference/analyze.py, which applies the same rules
engine the live webcam app uses -- so the demo and the product agree on what
counts as suspicious. Everything it depends on (the fine-tuned detector, the
calibrated thresholds) is derived from public data, so a fresh clone reproduces
this exactly.
"""

import os
import tempfile

import pandas as pd
import streamlit as st

from src.inference.analyze import BEHAVIOURS, analyze_video, overall_verdict
from src.thresholds import (
    EAR_CLOSED,
    HEAD_YAW_LOOKING_AWAY,
    PHONE_CONF,
    using_finetuned,
)

st.set_page_config(page_title="SentinelVision -- Exam Proctor Demo", page_icon="🎓", layout="wide")


@st.cache_resource
def warm_up():
    """Load the detector once and keep it warm across reruns."""
    from src.data.feature_extractor import FeatureExtractor
    return FeatureExtractor().describe()


def fmt_time(seconds):
    """Seconds -> M:SS for human-readable timelines."""
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"


def nice(behaviour):
    return behaviour.replace("_", " ").title()


def save_upload(uploaded):
    """Persist the uploaded video to a temp path OpenCV can read."""
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    return tmp.name


def render_results(windows, summary):
    verdict, blurb = overall_verdict(summary)
    banner = {"High risk": st.error, "Suspicious": st.warning, "Appears normal": st.success}
    banner[verdict](f"**{verdict}** — {blurb}")

    st.subheader("Per-behaviour findings")
    cols = st.columns(len(summary))
    for col, (behaviour, stats) in zip(cols, summary.items()):
        col.metric(
            nice(behaviour),
            f"{stats['flagged_fraction']:.0%} of clip",
            help=f"Flagged in {stats['flagged_windows']} of {stats['total_windows']} windows.",
        )
        col.caption(f"Peak confidence {stats['max_prob']:.0%}")

    st.subheader("Confidence over time")
    st.caption("Model confidence per 2-second window. Above the 0.5 line = flagged.")
    chart = windows[["window_start_sec"]].copy()
    for behaviour in summary:
        chart[nice(behaviour)] = windows[f"{behaviour}_prob"]
    chart["flag threshold"] = 0.5
    st.line_chart(chart.set_index("window_start_sec"))

    flagged_rows = [
        {"Behaviour": nice(behaviour), "From": fmt_time(start), "To": fmt_time(end)}
        for behaviour, stats in summary.items()
        for start, end in stats["intervals"]
    ]
    st.subheader("Flagged moments")
    if flagged_rows:
        st.dataframe(pd.DataFrame(flagged_rows), hide_index=True, use_container_width=True)
    else:
        st.write("Nothing flagged — the clip looks clean.")


def main():
    st.title("🎓 SentinelVision — Exam Proctor Demo")
    st.write(
        "Upload a webcam clip of someone taking an exam. The app measures gaze, head "
        "pose, blinks and objects on every frame with a fine-tuned YOLOv8 detector, "
        "then flags suspicious behaviour over time using calibrated thresholds."
    )

    with st.sidebar:
        st.header("About")
        if using_finetuned():
            st.success("Fine-tuned detector loaded.")
        else:
            st.warning(
                "Fine-tuned weights missing — falling back to stock COCO YOLOv8n, "
                "which is far weaker on phones."
            )
        st.caption(warm_up())

        st.write("**Behaviours it can flag:**")
        for behaviour in BEHAVIOURS:
            st.write(f"- {nice(behaviour)}")

        with st.expander("How it works & limitations"):
            st.markdown(
                "- **Pipeline:** MediaPipe (face / gaze / head pose) + a **fine-tuned "
                "YOLOv8n** for objects, scored per frame by the same rules engine "
                "the live app uses, then rolled into 2s windows.
"
                f"- **Calibrated thresholds:** phone confidence `{PHONE_CONF}` "
                f"(F1 0.923), head yaw `{HEAD_YAW_LOOKING_AWAY:.0f}°` (F1 0.869), "
                f"EAR `{EAR_CLOSED}` (F1 0.979) — each swept against public "
                "labelled data, not picked by hand.
"
                "- **Fine-tuning gain:** phone F1 **0.193 → 0.923** against the "
                "stock COCO model on held-out proctoring images.
"
                "- **Honest caveat:** that precision was measured on the public "
                "dataset. On unfamiliar rooms the detector still raises false "
                "phone alarms on dark rectangular background objects — measured at "
                "56% of phone-free frames on one held-out camera.
"
                "- The window score is the *fraction of frames* that fired, not a "
                "calibrated probability."
            )

    uploaded = st.file_uploader("Exam clip", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        st.info("👆 Upload a short clip (a few seconds is enough) to see the analysis.")
        return

    st.video(uploaded)
    if not st.button("Analyze clip", type="primary"):
        return

    path = save_upload(uploaded)
    try:
        bar = st.progress(0.0, text="Analyzing frames…")
        windows, summary = analyze_video(path, progress=lambda f: bar.progress(f))
        bar.empty()
        render_results(windows, summary)
    except ValueError as error:
        st.error(str(error))
    finally:
        os.remove(path)


if __name__ == "__main__":
    main()
