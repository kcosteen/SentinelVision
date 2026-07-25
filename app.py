"""
SentinelVision -- demo web app.

Upload an exam webcam clip; the app runs the full vision pipeline and the trained
behaviour model, then shows what it flagged and when. Run it with:

    streamlit run app.py

It reuses the same feature code the model trained on (src/inference/analyze.py),
so what you see here matches the training pipeline exactly.
"""

import os
import tempfile

import pandas as pd
import streamlit as st

from src.inference.analyze import analyze_video, load_model, overall_verdict

st.set_page_config(page_title="SentinelVision -- Exam Proctor Demo", page_icon="🎓", layout="wide")


@st.cache_resource
def get_model():
    """Load the trained model once and keep it warm across reruns."""
    return load_model()


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
        "Upload a webcam clip of someone taking an exam. The app detects gaze, head "
        "pose, blinks and objects per frame, then a trained model predicts suspicious "
        "behaviours over time."
    )

    with st.sidebar:
        st.header("About")
        try:
            model = get_model()
            st.success("Trained model loaded.")
            st.write("**Behaviours it can flag:**")
            for behaviour in model:
                st.write(f"- {nice(behaviour)}")
        except FileNotFoundError as error:
            model = None
            st.error(str(error))

        with st.expander("How it works & limitations"):
            st.markdown(
                "- **Pipeline:** MediaPipe (face/gaze/head) + YOLOv8 (objects) → "
                "2s feature windows → a random-forest classifier.\n"
                "- **Honest caveat:** trained on a small, single-person dataset, so "
                "numbers are illustrative, not production-grade.\n"
                "- The *phone* signal leans on head-down posture (weak object "
                "detection), so it can miss a phone held at eye level — a motivation "
                "for the next phase (fine-tuning the detector)."
            )

    if model is None:
        st.info("Train a model first: `python -m src.models.train`, then reload.")
        return

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
        windows, summary = analyze_video(path, model=model, progress=lambda f: bar.progress(f))
        bar.empty()
        render_results(windows, summary)
    except ValueError as error:
        st.error(str(error))
    finally:
        os.remove(path)


if __name__ == "__main__":
    main()
