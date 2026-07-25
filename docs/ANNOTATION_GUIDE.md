# Annotation Guide — labelling the Phase 2 phone frames

How to turn the 201 frames in `data/detection/raw_frames/` into training labels,
and — more importantly — how to draw them *consistently*. Inconsistent boxes put
a ceiling on mAP that no amount of training removes, because the model is being
asked to fit two contradictory definitions of "phone" at once.

Budget roughly 45–75 minutes for 201 frames once you're warmed up.

---

## 1. The one rule that matters most

**Label what is visible in the frame, not what you know is in the clip.**

These frames were selected precisely because the baseline detector found nothing
(`--strategy missed`). There are two very different reasons a frame ends up in
that pile:

1. The phone **is** visible and the detector failed — this is the gold. Draw a box.
2. The phone **isn't** visible in this frame at all — it's out of shot, behind a
   hand, below the desk edge, or the moment happens to fall between gestures.

Case 2 is real and common. The clip is labelled `phone` end-to-end, but any
individual frame may simply not show one. If you draw a box where no phone is
visible — guessing at where it "must" be — you are teaching the detector to
hallucinate phones from posture. **That is the exact bug Phase 2 exists to fix.**
Case 2 frames get an *empty* label file, not a guess (see §5).

If you can't see it, it isn't there.

---

## 2. The label spec

One class only: **`phone`** → class id **`0`**.

Draw the **tightest box that contains every visible pixel of the phone**, and
nothing more.

| Situation | What to do |
|---|---|
| Phone fully visible | Tight box around the phone body |
| Partly behind a hand/finger | Box the **whole phone including the hidden part**, inferred from its visible outline |
| Partly out of frame | Box only the part **inside** the frame — clip at the edge |
| Only a sliver visible (< ~10% of the phone) | Skip the frame entirely (§6) |
| Screen glow visible but body not distinguishable | No box — you're seeing light, not a phone |
| A second phone on the desk | Box it too — every phone gets a box |
| Tablet / laptop / TV remote | **No box.** Class is phones, not screens |
| Phone in a case | Box the phone-plus-case as one object |

**Include or exclude the hand?** Exclude. The box is the phone, not the grip. A
hand wrapped around a phone still gets a box around the phone's outline only.

**Occluded-part rule, stated precisely.** For a phone half-hidden behind fingers,
box the full phone as if the fingers were transparent — this is the standard
"amodal-ish" convention COCO uses for partial occlusion, and it's what keeps
boxes stable frame-to-frame as fingers shift. But for a phone running off the
edge of the frame, clip at the boundary — a box can't extend outside the image.
Those two rules feel contradictory; they aren't. Occlusion by an object is
inferred, occlusion by the frame edge is clipped.

---

## 3. Choosing a tool

All three export YOLO `.txt`, which is what `prepare_dataset.py` reads.

**A privacy note first.** These frames are your own face, in your own room. The
repo already gitignores them for that reason. Roboflow and other hosted
annotators require **uploading that footage to a third party**, where it may be
retained under their terms and, on the free Universe tier, can default to public
visibility. That's a real decision, not a formality. For personal footage I'd
annotate locally.

| Tool | Install | Trade-off |
|---|---|---|
| **labelImg** (suggested) | `pip install labelImg` | Local, offline, writes YOLO `.txt` directly. Minimal UI, which is a virtue at this size. The upstream project is archived but installs and works. |
| **labelme** | `pip install labelme` | Actively maintained, local. Saves JSON natively — needs a conversion step to YOLO. |
| **Roboflow** | Browser | Best UX and auto-suggestions, but uploads your footage. Check the project is set **private** if you use it. |

For 201 single-class frames, labelImg is the pragmatic pick: no account, no
upload, and the output lands in exactly the format we need.

---

## 4. Workflow with labelImg

```bash
pip install labelImg
labelImg data/detection/raw_frames
```

Then, once:

1. Click **Save format** on the left toolbar until it reads **YOLO** (it defaults
   to PascalVOC — if you miss this you'll produce `.xml` and have to redo it).
2. **View → Auto Save Mode**. Without this, clicking to the next image silently
   discards your box.
3. Set **Change Save Dir** to `data/detection/raw_frames` so `.txt` files land
   beside their images.

Keyboard flow, which is most of the speed:

| Key | Does |
|---|---|
| `W` | Start a new box |
| `D` | Next image |
| `A` | Previous image |
| `Ctrl` + `S` | Save (unnecessary with auto-save on) |

The first time you draw a box it asks for a class name — type `phone` exactly,
lowercase. It's reused from then on.

labelImg writes a `classes.txt` alongside the labels. Harmless — the importer
ignores any file that isn't `<image-stem>.txt`.

---

## 5. Background frames — do not skip this

A frame where **no phone is visible** gets an **empty `.txt` file**: same name as
the image, zero bytes.

This is a deliberate, valuable label, not a missing one. It tells the detector
"this desk, this lighting, this posture, no phone" — which is what stops it
firing on head-down posture alone. `prepare_dataset.py` keeps them:

> A `.txt` that exists but is empty is a deliberate background frame and is kept
> — backgrounds teach the detector what *isn't* a phone. An image with no `.txt`
> at all is simply not yet annotated, and is skipped.

So the distinction is load-bearing:

- **empty `.txt`** → "I looked, there's no phone" → used as a negative
- **no `.txt`** → "I haven't done this one" → silently ignored

labelImg won't create a file for an image you drew nothing on. Create them in one
sweep at the end, for every image still missing a label:

```bash
python -c "
import os, glob
d = 'data/detection/raw_frames'
made = 0
for img in glob.glob(os.path.join(d, '*.jpg')):
    txt = os.path.splitext(img)[0] + '.txt'
    if not os.path.exists(txt):
        open(txt, 'w').close()
        made += 1
print(f'created {made} empty (background) label files')
"
```

Only run that **after** you've been through every frame — it marks everything
untouched as "no phone here", so running it early would label your unannotated
backlog as background.

---

## 6. When to skip a frame entirely

Delete the image (and its `.txt`) rather than labelling it if:

- It's too motion-blurred for **you** to locate the phone with confidence. If you
  can't tell, the label would be noise.
- Only a tiny sliver of phone is visible — under ~10% of the object. Tiny,
  ambiguous boxes hurt more than they help at this dataset size.
- It's a near-duplicate of the previous frame with nothing changed.

Deleting is fine. 180 consistent frames beat 201 with 20 coin-flips in them.

---

## 7. Verify before you train

Check the count `prepare_dataset.py` sees:

```bash
python -m src.detection.prepare_dataset --own-dir data/detection/raw_frames --dry-run
```

You want `own frames: <N> annotated, 0 not yet annotated`. Any non-zero
"not yet annotated" means images without a `.txt` — either you missed them or you
haven't run the background sweep in §5.

Then sanity-check the labels themselves:

```bash
python -c "
import glob, os, collections
d = 'data/detection/raw_frames'
cls = collections.Counter(); boxes = empty = bad = 0
for f in glob.glob(os.path.join(d, '*.txt')):
    if os.path.basename(f) == 'classes.txt': continue
    lines = [l for l in open(f).read().split('\n') if l.strip()]
    if not lines: empty += 1
    for l in lines:
        p = l.split(); boxes += 1; cls[p[0]] += 1
        c = [float(v) for v in p[1:5]]
        if any(v < 0 or v > 1 for v in c) or c[2] <= 0 or c[3] <= 0: bad += 1
print('class ids:', dict(cls), '(must be only 0)')
print('boxes:', boxes, '| background files:', empty, '| malformed:', bad)
"
```

Three things to confirm:

- **class ids is `{'0': N}` and nothing else.** A stray `1` means labelImg picked
  up a second class name — a typo like `Phone` or `phone ` creates one silently.
- **malformed is 0.** Coordinates must be normalised to `[0, 1]`.
- **background count looks plausible** — some backgrounds are expected and
  healthy, but if it's most of your frames, re-read §1; you may be being too
  strict about what counts as visible.

Then build and train:

```bash
python -m src.detection.prepare_dataset --own-dir data/detection/raw_frames
python -m src.detection.train_detector --epochs 50
python -m src.detection.train_detector --eval-source own
```

---

## 8. Why consistency beats volume

The model can only learn a boundary you drew the same way twice. If you box
phone-plus-hand in the morning and phone-only after lunch, those two conventions
disagree on maybe 30% of each box's area — and at IoU 0.5 that's enough to turn
correct detections into scored misses. The measured mAP drops even though the
model learned fine.

Two habits that prevent it:

1. **Do all 201 in as few sittings as possible.** Convention drifts across days.
2. **When you hit a genuinely ambiguous frame, write the decision down** in a
   scratch note and apply it every time after. Then add it to §2 as a row, so the
   rule outlives your memory of it.

Being able to say *"here is my labelling protocol, here is the ambiguous case I
hit, here is the rule I adopted and why"* is a stronger interview answer than any
metric. Annotation policy is where most real ML projects quietly lose accuracy,
and knowing that is the point.

---

## 9. What this feeds

Annotated frames flow into the pipeline in `docs/DATASETS.md` §5:

```
raw_frames/ + .txt ──► prepare_dataset.py ──► train_detector.py ──► --eval-source own
```

Your frames are split **by clip**, never by frame, so all 20-odd frames of
`phone_003` land on one side of the train/val line. That's why the val score is
honest — and why annotating every frame of one clip and none of another would
skew it. Spread the effort across all ten clips.
