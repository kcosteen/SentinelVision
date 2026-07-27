"""The on-screen overlay for the live app.

Kept apart from main.py for two reasons. It is the only place allowed to draw,
which keeps the "never paint on the frame the models measure" rule easy to hold
-- everything here takes the DISPLAY frame. And it is pure drawing over plain
arguments, so the layout can be exercised on a blank image without a webcam.

Design notes, since a screen recording of this is the thing most people will
actually look at:

* Text sits on translucent panels rather than bare video. White-on-video is
  unreadable the moment something pale drifts behind it, which on a webcam is
  constantly.
* Colour carries the state: green normal, amber suspicious, red high risk. The
  score meter uses the same colour, so the reading is available peripherally
  without parsing any words.
* Detection boxes are drawn. Without them a viewer sees a verdict with no
  evidence -- showing the box the model actually put around the phone is what
  makes the demo legible.
"""

import cv2
import numpy as np

# --- palette (BGR) ---------------------------------------------------------
INK = (238, 238, 238)
MUTED = (170, 170, 170)
PANEL = (24, 22, 20)

GREEN = (120, 214, 126)
AMBER = (66, 183, 245)
RED = (86, 86, 240)
CYAN = (206, 191, 92)

STATUS_COLOURS = {
    "Normal": GREEN,
    "Suspicious": AMBER,
    "High Risk": RED,
}

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX


def panel(frame, x, y, w, h, alpha=0.72, colour=PANEL, radius=10, accent=None):
    """Translucent rounded backdrop, so text stays readable over any video.

    Rounded because square black boxes read as "debug output"; rounded ones read
    as an interface, and this is the surface a recruiter actually watches.
    `accent` paints a colour bar down the left edge -- the state is then legible
    peripherally, before any word has been read.
    """
    h_frame, w_frame = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(w_frame, x + w), min(h_frame, y + h)
    if x2 <= x or y2 <= y:
        return

    slab = frame[y:y2, x:x2]

    # Build the rounded shape once as a mask, then blend only inside it.
    mask = np.zeros(slab.shape[:2], np.uint8)
    sh, sw = mask.shape
    r = max(0, min(radius, sh // 2, sw // 2))
    cv2.rectangle(mask, (r, 0), (sw - r, sh), 255, -1)
    cv2.rectangle(mask, (0, r), (sw, sh - r), 255, -1)
    for cx, cy in ((r, r), (sw - r, r), (r, sh - r), (sw - r, sh - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1)

    tint = np.empty_like(slab)
    tint[:] = colour
    blended = cv2.addWeighted(tint, alpha, slab, 1 - alpha, 0)
    np.copyto(slab, blended, where=mask[:, :, None].astype(bool))

    if accent is not None:
        bar = np.zeros(slab.shape[:2], np.uint8)
        cv2.rectangle(bar, (0, r // 2), (4, sh - r // 2), 255, -1)
        strip = np.empty_like(slab)
        strip[:] = accent
        np.copyto(slab, strip, where=(bar & mask)[:, :, None].astype(bool))


def draw_border(frame, colour, thickness=4):
    """A thin frame-edge glow in the status colour.

    The single most legible element in a recording: the whole picture changes
    colour the instant the verdict does, which survives compression, small
    playback windows, and a viewer who is not reading the text.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), colour, thickness)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)


def text(frame, message, org, scale=0.6, colour=INK, thickness=1, font=FONT):
    cv2.putText(frame, message, org, font, scale, colour, thickness, cv2.LINE_AA)


# Boxed prominently; everything else is context and drawn faintly. `person` is
# already implied by the face brackets, and a full-height box around the subject
# fights with the one thing the viewer should be looking at.
FLAGGED = {"cell phone", "book", "headphone"}


def draw_detections(frame, detections, colour=CYAN, context=False):
    """Box what the pipeline acted on, with its confidence.

    Only detections that survived filtering are passed in, so what is boxed is
    exactly what the analyzer saw -- the overlay cannot drift from the decision.
    """
    for item in detections:
        box = item.get("box")
        if not box:
            continue

        flagged = item["label"] in FLAGGED
        if not flagged and not context:
            continue

        x1, y1, x2, y2 = (int(v) for v in box)

        if not flagged:
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
            continue

        # Corner-accented box: reads as a target rather than a plain rectangle,
        # and leaves the middle of the object unobscured.
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
        arm = max(10, int(min(x2 - x1, y2 - y1) * 0.25))
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * arm, cy), RED, 4, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * arm), RED, 4, cv2.LINE_AA)

        label = f"{item['label']}  {item['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT_S, 0.52, 1)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 18, y1 - 2), RED, -1)
        text(frame, label, (x1 + 9, y1 - 9), 0.52, (18, 18, 18), 1, FONT_S)


def draw_faces(frame, face_results, colour=GREEN):
    """Corner brackets around each face, rather than a full box.

    Deliberately lighter than MediaPipe's own `draw_detection`, which paints a
    filled box plus six keypoints -- fine for debugging, cluttered on a demo, and
    it obscures the face the viewer is trying to watch.
    """
    if not getattr(face_results, "detections", None):
        return

    h, w = frame.shape[:2]
    for detection in face_results.detections:
        box = detection.location_data.relative_bounding_box
        x1, y1 = int(box.xmin * w), int(box.ymin * h)
        x2, y2 = int((box.xmin + box.width) * w), int((box.ymin + box.height) * h)

        arm = max(12, int((x2 - x1) * 0.18))
        for (cx, cy, dx, dy) in (
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ):
            cv2.line(frame, (cx, cy), (cx + dx * arm, cy), colour, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * arm), colour, 2, cv2.LINE_AA)


def draw_status(frame, status, score, calibrating=False, high_risk=70):
    """Status badge and score meter, top-left. The primary read."""
    x, y, w, h = 22, 22, 330, 118

    if calibrating:
        colour, headline = CYAN, "CALIBRATING"
    else:
        colour = STATUS_COLOURS.get(status, INK)
        headline = status.upper()

    panel(frame, x, y, w, h, accent=colour)
    draw_border(frame, colour)

    text(frame, headline, (x + 24, y + 46), 0.95, colour, 2)

    if calibrating:
        text(frame, "learning the background", (x + 24, y + 78), 0.52, MUTED, 1, FONT_S)
        # Indeterminate sweep, so it reads as working rather than stuck.
        bar_x, bar_y, bar_w = x + 24, y + 92, w - 48
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 7), (58, 56, 54), -1)
        return

    # The number and its bar share the status colour, so the reading is
    # available peripherally without parsing any digits.
    score_text = f"{score:.0f}"
    text(frame, score_text, (x + 24, y + 84), 0.72, INK, 1)
    (sw, _), _ = cv2.getTextSize(score_text, FONT, 0.72, 1)
    text(frame, f"of {high_risk}", (x + 34 + sw, y + 84), 0.44, MUTED, 1, FONT_S)

    bar_x, bar_y, bar_w = x + 24, y + 96, w - 48
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 7), (58, 56, 54), -1)
    filled = int(bar_w * max(0.0, min(1.0, score / float(high_risk))))
    if filled:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 7), colour, -1)


def draw_signals(frame, gaze, head_yaw, fps=None, detector=None):
    """Live measurement readout, bottom-left -- what the decision is made from.

    Shown because a verdict with no visible inputs is a magic box. These are the
    three numbers the rules actually consume.
    """
    rows = [
        ("gaze", gaze if gaze else "--"),
        ("head yaw", f"{abs(head_yaw):.0f}deg" if head_yaw is not None else "--"),
    ]
    if fps:
        rows.append(("fps", f"{fps:.0f}"))

    w = 250
    h = 30 * len(rows) + 26
    x, y = 22, frame.shape[0] - h - 22
    panel(frame, x, y, w, h)

    text(frame, "SIGNALS", (x + 20, y + 24), 0.42, MUTED, 1, FONT_S)
    for i, (name, value) in enumerate(rows):
        line = y + 50 + i * 30
        text(frame, name, (x + 20, line), 0.5, MUTED, 1, FONT_S)
        text(frame, str(value), (x + 140, line), 0.56, INK, 1, FONT_S)

    if detector:
        # Its own chip: which detector is loaded changes what every number above
        # means, so it should not be a faint line floating on the video.
        cw = cv2.getTextSize(detector, FONT_S, 0.42, 1)[0][0] + 28
        panel(frame, x, y - 34, cw, 26, alpha=0.66, radius=8)
        text(frame, detector, (x + 14, y - 16), 0.42, CYAN, 1, FONT_S)


def draw_events(frame, events, weights=None):
    """Active alerts, top-right -- one row each, so they cannot collide."""
    if not events:
        return

    w = 322
    x = frame.shape[1] - w - 22
    y = 22
    h = 38 * len(events) + 34
    panel(frame, x, y, w, h, accent=RED)

    text(frame, "ALERTS", (x + 22, y + 26), 0.42, MUTED, 1, FONT_S)

    for i, event in enumerate(events):
        top = y + 40 + i * 38
        cv2.circle(frame, (x + 28, top + 12), 4, RED, -1, cv2.LINE_AA)
        text(frame, event, (x + 44, top + 17), 0.56, INK, 1, FONT_S)
        if weights and event in weights:
            badge = f"+{weights[event]}"
            (tw, _), _ = cv2.getTextSize(badge, FONT_S, 0.52, 1)
            text(frame, badge, (x + w - tw - 20, top + 17), 0.52, RED, 1, FONT_S)


def draw_title(frame, title="SentinelVision", subtitle=None):
    """Wordmark, bottom-right. A recording gets shared stripped of its context."""
    (tw, _), _ = cv2.getTextSize(title, FONT, 0.62, 1)
    x = frame.shape[1] - tw - 26
    y = frame.shape[0] - 26
    text(frame, title, (x, y), 0.62, INK, 1)
    if subtitle:
        (sw, _), _ = cv2.getTextSize(subtitle, FONT_S, 0.4, 1)
        text(frame, subtitle, (frame.shape[1] - sw - 26, y - 22), 0.4, (198, 198, 198), 1, FONT_S)
