"""Publish the fine-tuned detector to the Hugging Face Hub.

The weights are gitignored (6 MB of binary that changes rarely and does not
belong in git history), which means they live on exactly one laptop. The README
quotes their numbers, and they cost a GPU run to produce and cannot be rebuilt on
a CPU-only machine -- so "one laptop" is the whole backup story. This fixes that
and gives the model a citable home at the same time.

    # 1. Authenticate once (opens a browser, or paste a token from
    #    https://huggingface.co/settings/tokens -- needs WRITE scope):
    huggingface-cli login

    # 2. Publish:
    python -m scripts.publish_model --repo YOUR_USERNAME/sentinelvision-proctoring-yolov8n

    # Dry run first if you want to see exactly what would be uploaded:
    python -m scripts.publish_model --repo YOUR_USERNAME/... --dry-run

Uploads two things: the weights, and docs/MODEL_CARD.md as the repo's README --
which is what renders on the model page. A model card is not paperwork; it is
where the limitations live, and these weights have limitations a user needs to
know about before trusting them.
"""

import argparse
import os
import sys

WEIGHTS = os.path.join("models", "detection", "proctoring_yolov8n_best.pt")
CARD = os.path.join("docs", "MODEL_CARD.md")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", required=True,
                        help="target repo id, e.g. username/model-name")
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--card", default=CARD)
    parser.add_argument("--private", action="store_true",
                        help="create the repo private (default: public)")
    parser.add_argument("--dry-run", action="store_true",
                        help="check everything, upload nothing")
    parser.add_argument("--yes", action="store_true",
                        help="skip the public-upload confirmation (for non-interactive use)")
    args = parser.parse_args()

    for path in (args.weights, args.card):
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}")

    size_mb = os.path.getsize(args.weights) / 1e6

    from huggingface_hub import HfApi
    api = HfApi()

    try:
        who = api.whoami()
    except Exception:
        raise SystemExit(
            "Not authenticated. Run `huggingface-cli login` first "
            "(the token needs WRITE scope)."
        )

    print(f"account : {who.get('name')}")
    print(f"repo    : {args.repo}  ({'private' if args.private else 'PUBLIC'})")
    print(f"weights : {args.weights}  ({size_mb:.1f} MB)")
    print(f"card    : {args.card}  -> README.md")

    if args.dry_run:
        print("\nDry run -- nothing uploaded.")
        return

    # Public upload is irreversible in practice: anything published may be
    # fetched and cached by others within seconds, so confirm rather than assume.
    if not args.private and not args.yes:
        answer = input("\nPublish PUBLICLY? Anyone can download it. [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            raise SystemExit("Aborted.")

    api.create_repo(args.repo, repo_type="model",
                    private=args.private, exist_ok=True)

    api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")
    api.upload_file(path_or_fileobj=args.weights,
                    path_in_repo=os.path.basename(args.weights),
                    repo_id=args.repo, repo_type="model")

    print(f"\nDone -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    sys.exit(main())
