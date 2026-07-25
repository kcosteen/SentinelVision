"""Download a Roboflow Universe dataset into data/detection/external/.

Roboflow Universe hosts the only public data that actually matches our domain
(see docs/DATASETS.md), but every download is scoped to a personal API key. This
reads that key from the **environment** and never takes it as a flag:

    # PowerShell (this session only)
    $env:ROBOFLOW_API_KEY = "your_key_here"

    # bash
    export ROBOFLOW_API_KEY=your_key_here

    # then
    python -m src.detection.roboflow_import --source roboflow_online_proctoring
    python -m src.detection.roboflow_import --list

    # if the dataset page shows a different version than the registry default
    python -m src.detection.roboflow_import --source roboflow_cheating_person_phone --version 3

**Why the environment rather than `--api-key`.** Command-line arguments land in
shell history and in the process list, where other users on the machine can read
them. An env var does neither, and it keeps the key out of any file that could be
committed. The key is never printed by this module, including in error messages.

**Why stdlib rather than `pip install roboflow`.** The SDK pulls a large
dependency tree to wrap one documented REST endpoint, and this project has kept
itself dependency-light elsewhere (see evaluation/metrics.py). `urllib` +
`zipfile` are enough. If Roboflow changes the endpoint, the failure is a clear
HTTP error here rather than a silent behaviour change.

After downloading, build the training set as usual -- prepare_dataset.py picks up
any extracted Roboflow export automatically:

    python -m src.detection.prepare_dataset --own-dir data/detection/raw_frames
"""

import argparse
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from src.detection.sources import SOURCES, by_kind

API_KEY_ENV = "ROBOFLOW_API_KEY"

# Roboflow's documented export endpoint. Asking for yolov8 gives us the layout
# prepare_dataset.py already knows how to read.
EXPORT_URL = "https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}"
EXPORT_FORMAT = "yolov8"

# The export zip is the whole dataset, so allow a generous read timeout.
TIMEOUT_SECONDS = 120


def require_api_key():
    """The key from the environment, or a clear explanation of how to set it."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise SystemExit(
            f"{API_KEY_ENV} is not set.\n\n"
            "  1. Make a free account at https://roboflow.com\n"
            "  2. Copy your key from Settings -> API Keys\n"
            "  3. Set it for this session:\n"
            f"       PowerShell:  $env:{API_KEY_ENV} = \"your_key\"\n"
            f"       bash:        export {API_KEY_ENV}=your_key\n\n"
            "It is read from the environment on purpose -- passing it as a flag "
            "would put it in your shell history."
        )
    return key


def request_export_link(source, version, api_key):
    """Ask Roboflow for a download link. Returns the zip URL.

    The endpoint answers with JSON describing the generated export. A dataset
    version that doesn't exist gives a 404, which is the most likely failure --
    the version in the registry is a guess, and the real one is on the page.
    """
    url = EXPORT_URL.format(
        workspace=source["workspace"],
        project=source["project"],
        version=version,
        fmt=EXPORT_FORMAT,
    )
    # The key travels as a query parameter (what the API expects) but is never
    # echoed: every message below prints `url`, not the full request.
    request_url = f"{url}?{urllib.parse.urlencode({'api_key': api_key})}"

    print(f"Requesting export: {url}")
    try:
        with urllib.request.urlopen(request_url, timeout=TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        # Re-raise without the URL, which carries the key in its query string.
        if error.code == 401:
            raise SystemExit(f"Roboflow rejected the key in {API_KEY_ENV} (HTTP 401).")
        if error.code == 404:
            raise SystemExit(
                f"Not found (HTTP 404): {url}\n"
                f"Most likely the version is wrong. Open {source['url']} and use "
                f"the version number shown there:\n"
                f"  python -m src.detection.roboflow_import --source ... --version N"
            )
        raise SystemExit(f"Roboflow returned HTTP {error.code} for {url}")
    except urllib.error.URLError as error:
        raise SystemExit(f"Could not reach Roboflow: {error.reason}")

    link = (payload.get("export") or {}).get("link")
    if not link:
        # Surface the response shape so an API change is diagnosable, not silent.
        raise SystemExit(
            "Roboflow's response had no export link. Keys present: "
            f"{sorted(payload)}. The API may have changed -- see {source['url']}."
        )
    return link


def download_and_extract(link, destination):
    """Stream the export zip to disk and unpack it into `destination`."""
    os.makedirs(destination, exist_ok=True)
    archive = os.path.join(destination, "_export.zip")

    print("Downloading export zip ...")
    try:
        with urllib.request.urlopen(link, timeout=TIMEOUT_SECONDS) as response:
            with open(archive, "wb") as handle:
                shutil.copyfileobj(response, handle)
    except urllib.error.URLError as error:
        raise SystemExit(f"Download failed: {error.reason}")

    size_mb = os.path.getsize(archive) / 1e6
    print(f"  {size_mb:.1f} MB -> extracting")

    with zipfile.ZipFile(archive) as bundle:
        # Guard against a zip whose entries escape the destination directory.
        for member in bundle.namelist():
            resolved = os.path.normpath(os.path.join(destination, member))
            if not resolved.startswith(os.path.normpath(destination) + os.sep):
                raise SystemExit(f"Refusing to extract unsafe path: {member}")
        bundle.extractall(destination)

    os.remove(archive)
    return destination


def summarise(destination):
    """Report what landed, so a bad export is obvious immediately."""
    images = labels = 0
    for root, _, files in os.walk(destination):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                images += 1
            elif name.lower().endswith(".txt") and name != "README.txt":
                labels += 1

    data_yaml = os.path.join(destination, "data.yaml")
    print(f"\n  images: {images}   label files: {labels}")
    if os.path.exists(data_yaml):
        with open(data_yaml) as handle:
            for line in handle:
                if line.startswith(("names:", "nc:")):
                    print(f"  {line.strip()}")
    else:
        print("  WARNING: no data.yaml in the export -- prepare_dataset.py needs it "
              "to read the upstream class list.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", help="registry key (see --list)")
    parser.add_argument("--version", type=int, default=None,
                        help="dataset version; defaults to the registry value")
    parser.add_argument("--out-dir", default=os.path.join("data", "detection", "external"))
    parser.add_argument("--list", action="store_true",
                        help="show the Roboflow sources in the registry")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the destination already exists")
    args = parser.parse_args()

    available = by_kind("roboflow")

    if args.list or not args.source:
        print("Roboflow sources:\n")
        for name, source in available.items():
            print(f"  {name}")
            print(f"    {source['what']}")
            print(f"    {source['workspace']}/{source['project']} "
                  f"v{source['version']}  |  {source['size']}")
            print(f"    licence: {source['licence']}")
            print(f"    {source['url']}\n")
        if not args.source:
            print("Pick one with --source <name>.")
        return

    if args.source not in available:
        raise SystemExit(
            f"Unknown source '{args.source}'. Available: {', '.join(available)}"
        )

    source = available[args.source]
    version = args.version if args.version is not None else source["version"]
    destination = os.path.join(args.out_dir, args.source)

    if os.path.exists(destination) and not args.force:
        print(f"Already present: {destination}")
        summarise(destination)
        print("\nRe-download with --force, or build the dataset:\n"
              "  python -m src.detection.prepare_dataset")
        return

    api_key = require_api_key()

    print(f"{args.source}  ({source['workspace']}/{source['project']} v{version})")
    print(f"Licence: {source['licence']} -- check {source['url']} before shipping.\n")

    link = request_export_link(source, version, api_key)
    if os.path.exists(destination):
        shutil.rmtree(destination)
    download_and_extract(link, destination)

    print(f"\nExtracted -> {destination}")
    summarise(destination)
    print("\nNext: python -m src.detection.prepare_dataset "
          "--own-dir data/detection/raw_frames")


if __name__ == "__main__":
    main()
