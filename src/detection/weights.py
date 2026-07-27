"""Get the fine-tuned detector, downloading it from the Hub if it isn't local.

The weights are 6 MB of binary that changes rarely, so they are gitignored rather
than carried in git history. That leaves a fresh clone -- or a Hugging Face Space
built from one -- with no detector at all, silently falling back to stock COCO
weights whose phone F1 is 0.193 against this model's 0.923.

So fetch them. The model is public, no token needed:
https://huggingface.co/kcosteen/sentinelvision-proctoring-yolov8n

Failure is non-fatal and loud. Offline, the app still runs on the COCO baseline;
it just says so rather than pretending the numbers in the README apply.
"""

import os

from src.thresholds import FINETUNED_WEIGHTS

HF_REPO = "kcosteen/sentinelvision-proctoring-yolov8n"
HF_FILENAME = "proctoring_yolov8n_best.pt"


def ensure_finetuned_weights(repo=HF_REPO, filename=HF_FILENAME, quiet=False):
    """Path to the fine-tuned weights, downloading them if needed.

    Returns the path on success, or None if they could not be obtained -- in
    which case the caller should fall back to the COCO baseline.
    """
    if os.path.exists(FINETUNED_WEIGHTS):
        return FINETUNED_WEIGHTS

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        if not quiet:
            print("[weights] huggingface_hub not installed; cannot fetch the "
                  "fine-tuned detector.")
        return None

    if not quiet:
        print(f"[weights] {FINETUNED_WEIGHTS} not found -- fetching {repo}")

    try:
        os.makedirs(os.path.dirname(FINETUNED_WEIGHTS), exist_ok=True)
        # local_dir puts it exactly where thresholds.py expects to find it, so
        # every later run and every other entry point sees it without asking.
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=os.path.dirname(FINETUNED_WEIGHTS),
        )
        if not quiet:
            print(f"[weights] downloaded -> {path}")
        return path if os.path.exists(path) else None

    except Exception as error:
        # Offline, rate-limited, repo renamed -- all recoverable, none worth
        # crashing over. The caller degrades to the baseline and says so.
        if not quiet:
            print(f"[weights] could not fetch ({type(error).__name__}: {error}); "
                  "falling back to stock COCO weights.")
        return None
