# Hugging Face Spaces no longer offers a `streamlit` SDK -- create_repo accepts
# only gradio, docker or static -- so a Streamlit app ships as a container.
FROM python:3.11-slim

# OpenCV needs these even in its headless build: the wheel links against
# libglib/libgomp regardless of whether any GUI is involved. Without them the
# import fails at runtime with a bare "libgthread-2.0.so.0: cannot open shared
# object file", which reads like a Python problem and is not one.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as uid 1000 and only $HOME is writable. Everything that wants a
# cache must be pointed inside it, or the first write raises PermissionError
# somewhere deep in a library: ultralytics writes settings on import, matplotlib
# builds a font cache, and huggingface_hub caches the detector it downloads.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Dependencies first, so editing the app doesn't re-run the slow layer.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# 7860 is what Spaces expects; app_port in README.md must agree.
EXPOSE 7860

# headless: no browser to open. enableXsrfProtection off because the Space sits
# behind Hugging Face's own proxy, which otherwise breaks file uploads.
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
