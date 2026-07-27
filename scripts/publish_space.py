"""Deploy the Streamlit demo to a Hugging Face Space.

    # Authenticate once (WRITE-scope token; stores it in your HF config so it
    # never has to appear in a shell command or a chat message):
    hf auth login

    python -m scripts.publish_space --repo YOUR_USERNAME/sentinelvision

    # See exactly what would be uploaded, without uploading:
    python -m scripts.publish_space --repo YOUR_USERNAME/sentinelvision --dry-run

The Space gets `app.py`, the `src/` package, and the two files in deploy/space:
a README carrying the YAML front-matter Spaces needs, and a requirements file
that differs from the repo's in two ways that both matter (headless OpenCV, no
scikit-learn -- see the comments in it).

The 6 MB detector is NOT uploaded. `src/detection/weights.py` pulls it from the
model repo on first run, so the Space stays small and there is exactly one
authoritative copy of the weights rather than two that can drift apart.
"""

import argparse
import os
import sys

SPACE_DIR = os.path.join("deploy", "space")

# app.py plus the whole `src` package. Some of it -- the calibration sweeps, the
# detector-training pipeline -- the Space never imports, but the package is ~170 KB
# of pure Python and shipping all of it removes a whole class of failure where a
# too-clever exclusion list drops a transitive import and the Space dies on boot.
# Notebooks, tests and datasets stay behind; those are genuinely large.
INCLUDE_FILES = ["app.py"]
INCLUDE_TREES = ["src"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def collect(dry_run=False):
    """(local_path, path_in_repo) pairs to upload."""
    pairs = [
        (os.path.join(SPACE_DIR, "README.md"), "README.md"),
        (os.path.join(SPACE_DIR, "requirements.txt"), "requirements.txt"),
        (os.path.join(SPACE_DIR, "Dockerfile"), "Dockerfile"),
    ]
    for name in INCLUDE_FILES:
        pairs.append((name, name))

    for tree in INCLUDE_TREES:
        for dirpath, dirnames, filenames in os.walk(tree):
            dirnames[:] = [d for d in dirnames if d not in SKIP_PARTS]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                local = os.path.join(dirpath, filename)
                pairs.append((local, local.replace(os.sep, "/")))

    missing = [p for p, _ in pairs if not os.path.exists(p)]
    if missing:
        raise SystemExit("Missing:\n  " + "\n  ".join(missing))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", required=True, help="e.g. username/space-name")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true",
                        help="skip the public-deploy confirmation")
    args = parser.parse_args()

    pairs = collect()
    total = sum(os.path.getsize(p) for p, _ in pairs)

    from huggingface_hub import HfApi
    api = HfApi()

    try:
        who = api.whoami()
    except Exception:
        raise SystemExit("Not authenticated. Run `hf auth login` first "
                         "(the token needs WRITE scope).")

    print(f"account : {who.get('name')}")
    print(f"space   : {args.repo}  ({'private' if args.private else 'PUBLIC'})")
    print(f"files   : {len(pairs)}  ({total/1024:.0f} KB)")
    for _, remote in sorted(pairs, key=lambda x: x[1])[:40]:
        print(f"          {remote}")
    if len(pairs) > 40:
        print(f"          ... and {len(pairs) - 40} more")
    print("weights : fetched at runtime from the model repo, not uploaded")

    if args.dry_run:
        print("\nDry run -- nothing uploaded.")
        return

    if not args.private and not args.yes:
        answer = input("\nDeploy PUBLICLY? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            raise SystemExit("Aborted.")

    # Spaces dropped the `streamlit` SDK -- create_repo now accepts only
    # gradio, docker or static -- so the Streamlit app ships as a container.
    api.create_repo(args.repo, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)

    for local, remote in pairs:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=args.repo, repo_type="space")
        print(f"  uploaded {remote}")

    print(f"\nDone -> https://huggingface.co/spaces/{args.repo}")
    print("The first build takes a few minutes -- watch the Logs tab.")


if __name__ == "__main__":
    sys.exit(main())
