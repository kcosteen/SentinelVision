"""Measure how accurate `src/features/head_pose.py` actually is.

Head pose and gaze are NOT learned models -- `calculate_head_pose` solves a PnP
problem from six MediaPipe landmarks, and `gaze_ratio` is iris position over eye
width. There is nothing to train. What they need instead is (a) evidence they
work and (b) a threshold for "looking away" that came from data rather than
taste. This does both.

    # Validate + calibrate against ground-truth angles:
    python -m src.calibration.validate_head_pose

    # Bigger sample:
    python -m src.calibration.validate_head_pose --limit 1200

Ground truth comes from the `cheating-vfvwa` download, which is the Gourier Head
Pose Image Database relabelled: each filename encodes the true tilt and pan in
degrees, on a 15-degree grid.

**The stretch correction.** Roboflow preprocessed these with "Stretch to
640x640", but the originals are 4:3. Squashing 4:3 into 1:1 compresses the
horizontal axis by ~0.75 and would systematically shrink every yaw estimate --
an artifact of the export, not a flaw in our solver. So frames are restored to
4:3 before measuring. `--no-unstretch` shows how much that correction is worth.

**What counts as success.** Not a small error -- a *monotonic* one. For a
threshold to exist at all, estimated yaw has to rise consistently with true yaw.
A constant offset or scale is harmless (we pick the threshold on our own scale);
a non-monotonic response means no threshold can work.
"""

import argparse
import collections
import os
import re

import cv2
import numpy as np

from src.calibration.sweep import best_threshold, format_sweep, sweep_threshold
from src.features.head_pose import calculate_head_pose

DEFAULT_ROOT = os.path.join("data", "detection", "external", "roboflow_cheating_vfvwa")

# "person01111-60-60_jpg.rf.<hash>.jpg" -> tilt -60, pan -60
FILENAME = re.compile(r"person(\d+?)([+-]\d+)([+-]\d+)_jpg")

# The originals are 384x288; only the ratio matters for undoing the stretch.
ORIGINAL_ASPECT = 4 / 3


def parse_angles(filename):
    """(tilt, pan) in degrees from a Gourier filename, or None."""
    match = FILENAME.search(filename)
    if not match:
        return None
    return int(match.group(2)), int(match.group(3))


def collect_images(root, limit):
    """One image per (person, tilt, pan) -- the export has ~3 augmented copies."""
    unique = {}
    for split in ("train", "valid", "test"):
        for label in ("cheating", "normal"):
            directory = os.path.join(root, split, label)
            if not os.path.isdir(directory):
                continue
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    match = FILENAME.search(entry.name)
                    if not match:
                        continue
                    # Same subject+pose across augmentations: keep one.
                    key = (match.group(1), match.group(2), match.group(3))
                    unique.setdefault(key, entry.path)

    items = sorted(unique.items())
    if limit:
        stride = max(1, len(items) // limit)
        items = items[::stride][:limit]
    return items


def measure(path, face_mesh, unstretch):
    """(pitch, yaw, roll) for one image, or None if no face mesh was found."""
    image = cv2.imread(path)
    if image is None:
        return None

    if unstretch:
        # Undo Roboflow's square stretch so the geometry matches a real camera.
        height = image.shape[0]
        image = cv2.resize(image, (int(round(height * ORIGINAL_ASPECT)), height))

    height, width = image.shape[:2]
    mesh = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not mesh.multi_face_landmarks:
        return None
    return calculate_head_pose(mesh.multi_face_landmarks[0].landmark, width, height)


def report_monotonicity(rows):
    """Does estimated yaw track true pan -- in sign, in magnitude, or not at all?

    Both are reported because they fail independently. `cv2.RQDecomp3x3` is prone
    to sign flips and angle wrapping, which can leave the SIGNED estimate looking
    like noise (mean ~0 at every true pan) while |yaw| still rises cleanly with
    |pan|. Only checking the signed mean would call that "unusable" when a
    perfectly good magnitude threshold exists; only checking the magnitude would
    hide that left/right cannot be told apart.
    """
    signed = collections.defaultdict(list)
    absolute = collections.defaultdict(list)
    for true_pan, _, est_yaw, _ in rows:
        signed[true_pan].append(est_yaw)
        absolute[abs(true_pan)].append(abs(est_yaw))

    print(f"\nSIGNED -- can we tell left from right?")
    print(f"{'true pan':>9}{'n':>6}{'est. yaw (mean)':>18}{'std':>8}")
    print("-" * 42)
    signed_means = []
    for pan in sorted(signed):
        values = np.array(signed[pan])
        signed_means.append(values.mean())
        print(f"{pan:>9}{len(values):>6}{values.mean():>18.1f}{values.std():>8.1f}")

    print(f"\nMAGNITUDE -- can we tell 'turned' from 'facing forward'?")
    print(f"{'|true pan|':>11}{'n':>6}{'mean |yaw|':>13}{'std':>8}")
    print("-" * 40)
    abs_means = []
    for pan in sorted(absolute):
        values = np.array(absolute[pan])
        abs_means.append(values.mean())
        print(f"{pan:>11}{len(values):>6}{values.mean():>13.1f}{values.std():>8.1f}")

    def rank_correlation(series):
        """Spearman rho between bin order and the measured means.

        Strict all-pairs monotonicity is the wrong test here: the estimate
        saturates at extreme angles (the landmarks the solver needs become
        occluded), so one small dip in the last bins would fail it while the
        relationship is plainly strong. Rank correlation tolerates that.
        """
        n = len(series)
        if n < 3:
            return 0.0
        order = np.argsort(np.argsort(np.array(series, dtype=float)))
        expected = np.arange(n)
        centred_a = order - order.mean()
        centred_b = expected - expected.mean()
        denominator = np.sqrt((centred_a ** 2).sum() * (centred_b ** 2).sum())
        return float((centred_a * centred_b).sum() / denominator) if denominator else 0.0

    signed_rho = rank_correlation(signed_means)
    abs_rho = rank_correlation(abs_means)

    # Graded rather than a hard pass/fail. Over ~7 bins a single adjacent swap
    # already costs ~0.1 of rho, so a 0.9 gate would fail relationships that are
    # obviously strong. The threshold sweep below is the practical test anyway;
    # this is here to explain *why* it does or doesn't work.
    def verdict(rho):
        magnitude = abs(rho)
        if magnitude > 0.85:
            return "strong"
        if magnitude > 0.6:
            return "weak"
        return "none"

    signed_ok = verdict(signed_rho) == "strong"
    abs_ok = verdict(abs_rho) == "strong"

    print()
    print(f"Signed  yaw vs pan : rho = {signed_rho:+.2f}  ({verdict(signed_rho)})")
    print(f"|yaw| vs |pan|     : rho = {abs_rho:+.2f}  ({verdict(abs_rho)})")
    print()

    if signed_ok:
        print("Signed yaw tracks pan -- left/right direction is usable.")
    else:
        print("Signed yaw does NOT track pan: the mean sits near zero at every")
        print("true angle, so LEFT and RIGHT cannot be distinguished. That is the")
        print("known RQDecomp3x3 sign-flip, not a threshold problem.")

    if abs_ok:
        print("|yaw| rises with |pan| -- a 'looking away' magnitude threshold IS")
        print("meaningful, even though the direction is not.")
        # Flag where the response stops growing: past that, the estimate cannot
        # distinguish "turned a lot" from "turned even more".
        peak = int(np.argmax(abs_means))
        if peak < len(abs_means) - 1:
            print(f"It saturates around |pan| = {sorted(absolute)[peak]} deg, so "
                  f"angles beyond that\nare not separable from one another.")
    else:
        print("|yaw| does not track |pan| either -- the solver output is unusable.")

    return signed_ok, abs_ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--no-unstretch", action="store_true",
                        help="skip the 4:3 correction, to see what it's worth")
    parser.add_argument("--looking-away-deg", type=float, default=45.0,
                        help="true |pan| at which we call it looking away")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(
            f"{args.root} not found. Fetch it with:\n"
            f"  python -m src.detection.roboflow_import --source roboflow_cheating_vfvwa"
        )

    items = collect_images(args.root, args.limit)
    if not items:
        raise SystemExit(f"No Gourier-style filenames under {args.root}")
    print(f"Measuring {len(items)} unique (person, tilt, pan) images "
          f"[unstretch={'off' if args.no_unstretch else 'on'}]")

    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True, static_image_mode=True
    )

    rows = []
    no_face = 0
    for (_, tilt_s, pan_s), path in items:
        angles = measure(path, face_mesh, not args.no_unstretch)
        if angles is None:
            no_face += 1
            continue
        pitch, yaw, _ = angles
        rows.append((int(pan_s), int(tilt_s), yaw, pitch))

    print(f"  {len(rows)} measured, {no_face} no face mesh")
    if not rows:
        raise SystemExit("No measurements -- MediaPipe found no faces at all.")

    signed_ok, magnitude_ok = report_monotonicity(rows)

    # Calibrate the threshold on OUR scale: does |estimated yaw| separate the
    # poses the dataset calls "looking away"? Absolute error against the true
    # degrees matters far less than whether a usable cut point exists.
    scores = [abs(yaw) for _, _, yaw, _ in rows]
    labels = [1 if abs(pan) >= args.looking_away_deg else 0 for pan, _, _, _ in rows]

    print(f"\nCalibrating |estimated yaw| against 'true |pan| >= "
          f"{args.looking_away_deg:.0f} deg'")
    print(f"  positives: {sum(labels)} / {len(labels)}")

    thresholds = [float(t) for t in range(5, 61, 5)]
    sweep = sweep_threshold(scores, labels, thresholds)
    chosen = best_threshold(sweep)
    print()
    print(format_sweep(sweep, highlight=chosen["threshold"] if chosen else None))

    if chosen:
        print(f"\n=> Use |yaw| >= {chosen['threshold']:.0f} deg for 'looking away' "
              f"(precision {chosen['precision']:.3f}, recall {chosen['recall']:.3f}, "
              f"f1 {chosen['f1']:.3f})")

    print("\nNOTE: this validates the SOLVER and picks a threshold. Nothing here is")
    print("trained -- head pose is geometry. The images are also lab portraits on a")
    print("plain background, so treat the threshold as a grounded starting point for")
    print("a real webcam, not a final answer.")
    if not magnitude_ok:
        print("\nThe magnitude result above outweighs any threshold reported.")
    elif not signed_ok:
        print("\nUsable for 'is the head turned?', NOT for 'which way?'. If the")
        print("proctoring logic ever needs direction (e.g. glancing at a neighbour")
        print("on one side), the solver needs fixing first -- see docs.")


if __name__ == "__main__":
    main()
