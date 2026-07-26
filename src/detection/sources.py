"""Registry of external datasets usable for the Phase 2 phone detector.

Finding training data for *exam proctoring* is genuinely hard -- there is no
public "student cheating on webcam" detection dataset, because the footage is
by nature private and consent-bound. So this registry records what was actually
surveyed, what survived inspection, and what each source is good for. Keeping it
in code (rather than a note somewhere) means the download path and the licence
sit next to each other and can't drift apart.

    # What's available and why:
    python -m src.detection.sources

Only `hf_*` entries download without credentials. Roboflow Universe hosts the
best-matched data by far, but every download goes through an account-scoped API
key, so those entries are documented for a human to fetch rather than automated.

**The domain-gap caveat, which is the whole point of Phase 2.** The baseline
YOLOv8n finds the phone in only ~18% of frames in our own `phone_*` clips. That
is not fixed by adding more *clean* phone photos: public phone datasets are
mostly well-lit, centred, unoccluded product-ish shots, whereas our failure mode
is a dim webcam, a phone half-hidden by a hand, at lap level, motion-blurred.
Mixing in self-labelled frames from our own clips is therefore not a nice-to-have
-- it is the part that closes the gap. See `extract_frames.py`.
"""

# Each entry: what it is, how to get it, and the honest caveat.
#
# `kind` drives what prepare_dataset.py can do automatically:
#   "hf_coco"    -> snapshot_download + COCO-JSON -> YOLO conversion
#   "manual"     -> needs a human (credentials, or a login-walled archive)
SOURCES = {
    "hf_phone_detection": {
        "kind": "hf_coco",
        "repo_id": "harshdadiya-wappnet/phone_detection",
        "licence": "Apache-2.0",
        "size": "605 images / 589 boxes / 15.7 MB",
        "classes": ["mobile_phone"],
        "what": "Roboflow 'Refined-mobile-tablet' export, 512x512, COCO JSON.",
        "caveat": (
            "The only HF dataset found with real phone bounding boxes AND a "
            "permissive licence. Small, and the images are clean 512x512 crops "
            "-- good for teaching 'what a phone looks like', useless on its own "
            "for our webcam domain. Its COCO category 0 is a typo'd "
            "'mobuile_phonw' supercategory carrying no boxes; the converter "
            "drops empty categories rather than emitting a phantom class."
        ),
    },
    "roboflow_online_proctoring": {
        "kind": "roboflow",
        "workspace": "online-exam-cheating-detection-kvdul",
        "project": "online-proctoring-system-x27ou-e7abr",
        "version": 1,
        "url": "https://universe.roboflow.com/online-exam-cheating-detection-kvdul/online-proctoring-system-x27ou-e7abr",
        "licence": "check on the page (Roboflow Universe varies per dataset)",
        "size": "see page",
        "classes": ["face", "phone", "other exam objects"],
        "what": "Purpose-built online-proctoring set, shot to simulate remote exams.",
        "caveat": (
            "Closest match to our actual domain of anything surveyed. Fetch with "
            "`python -m src.detection.roboflow_import --source "
            "roboflow_online_proctoring`, which reads ROBOFLOW_API_KEY from the "
            "environment. Confirm `version` against the dataset page -- the "
            "default of 1 is a guess and a wrong version is a 404."
        ),
    },
    "roboflow_cheating_vfvwa": {
        "kind": "roboflow",
        "workspace": "mahmoud-mohamed-phhz1",
        "project": "cheating-vfvwa",
        "version": 1,
        "url": "https://universe.roboflow.com/mahmoud-mohamed-phhz1/cheating-vfvwa/dataset/1",
        "licence": "CC BY 4.0",
        "size": "1,775 images (4,933 after 3x augmentation)",
        "classes": ["cheating (992)", "normal (783)"],
        "what": "Whole-image cheating/normal CLASSIFICATION -- no bounding boxes.",
        "caveat": (
            "VERIFIED BY INSPECTION. It is the Gourier Head Pose Image Database "
            "relabelled: lab shots against a plain grey background, filenames "
            "encoding tilt/pan. Its 'cheating' label is almost exactly "
            "|head yaw| >= 45 degrees (F1 0.935 predicting the label from pan "
            "alone), and tilt barely matters -- head straight DOWN is labelled "
            "*normal*. So it contains no phones, is not exam footage, and its "
            "notion of cheating disagrees with ours. Genuinely useful for one "
            "thing: an external, labelled reference for where 'looking away' "
            "begins in yaw."
        ),
    },
    "roboflow_cheating_person_phone": {
        "kind": "roboflow",
        "workspace": "online-exam-cheating-detection-kvdul",
        "project": "cheating-faalb-jvigx-jxt99",
        "version": 1,
        "url": "https://universe.roboflow.com/online-exam-cheating-detection-kvdul/cheating-faalb-jvigx-jxt99",
        "licence": "check on the page",
        "size": "~1,798 images",
        "classes": ["person", "phone", "calculator"],
        "what": "Cheating-behaviour detection with person/phone/calculator boxes.",
        "caveat": (
            "Larger than the HF option and already exam-framed. 'person' and "
            "'calculator' are out of scope -- the importer drops any class not in "
            "CLASS_ALIASES rather than training heads we never read. Confirm "
            "`version` against the dataset page."
        ),
    },
    "hf_biwi_head_pose": {
        "kind": "manual",
        "url": "https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose",
        "licence": "other -- research / non-commercial, check before shipping",
        "size": "24 sequences, ~15k frames",
        "classes": ["yaw/pitch/roll ground truth"],
        "what": "Kinect head-pose sequences with ground-truth Euler angles.",
        "caveat": (
            "Not detector data. Its value is *validating* src/features/head_pose.py "
            "numerically -- report mean absolute angular error instead of asserting "
            "the module works. It's a loader script that pulls from ETH, and the "
            "licence is research-only, so it stays out of any shipped artefact."
        ),
    },
}

# Surveyed and rejected -- recorded so the search isn't repeated in six months.
REJECTED = {
    "ybli/yolo-classroom-student-head-up-head-down": "repo is empty; README points to a Baidu Cloud link + extraction code",
    "ybli/yolo-phone-book-cup-object-detection": "same -- empty repo, Baidu link",
    "ybli/yolo-driver-distraction-detection": "same -- empty repo, Baidu link",
    "lord-reso/inbrowser-proctor-dataset": "name is misleading: it's audio/ASR (505 speech clips), not vision",
    "lamkser/face_occlusion": "empty repo",
    "vibrantturtle/phone-detection-data": "998 images but zero annotation files -- unlabelled",
    "gymprathap/Driver-Distracted-Dataset": "4.3 GB, no README, classification not detection, car-interior domain",
    "MahekDharod/cellphone-detection-dataset": "MIT, 74.5 MB zip, but README is only a licence line -- contents unverified",
}


def by_kind(kind):
    """The sources of one kind, in registry order."""
    return {name: src for name, src in SOURCES.items() if src["kind"] == kind}


def print_catalogue():
    """Human-readable dump of what's usable, what needs a key, and what's dead."""
    print("AUTOMATED (no credentials -- prepare_dataset.py fetches these)")
    print("=" * 70)
    for name, src in by_kind("hf_coco").items():
        print(f"\n{name}")
        print(f"  {src['what']}")
        print(f"  {src['repo_id']}  |  {src['licence']}  |  {src['size']}")
        print(f"  caveat: {src['caveat']}")

    print("\n\nROBOFLOW (needs ROBOFLOW_API_KEY in the environment)")
    print("=" * 70)
    for name, src in by_kind("roboflow").items():
        print(f"\n{name}")
        print(f"  {src['what']}")
        print(f"  {src['url']}")
        print(f"  {src['workspace']}/{src['project']} v{src['version']}"
              f"  |  licence: {src['licence']}  |  {src['size']}")
        print(f"  classes upstream: {', '.join(src['classes'])}")
        print(f"  caveat: {src['caveat']}")
    print("\n  Fetch:  python -m src.detection.roboflow_import --source <name>")

    print("\n\nMANUAL (needs a human decision)")
    print("=" * 70)
    for name, src in by_kind("manual").items():
        print(f"\n{name}")
        print(f"  {src['what']}")
        print(f"  {src['url']}")
        print(f"  licence: {src['licence']}  |  {src['size']}")
        print(f"  caveat: {src['caveat']}")

    print("\n\nSURVEYED AND REJECTED")
    print("=" * 70)
    for name, why in REJECTED.items():
        print(f"  {name}\n      {why}")


if __name__ == "__main__":
    print_catalogue()
