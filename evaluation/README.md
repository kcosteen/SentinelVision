# Evaluation

This folder measures **how good the proctor's decisions actually are**, instead
of just trusting that they look right on screen.

## Why this exists

A demo that "looks like it works" is not evidence. Interviewers (and users) want
numbers: *When the system says "phone", how often is it right? How many real
phones does it miss?* Those are **precision** and **recall**, and this harness
computes them.

## The metrics (plain English)

For each behavior we compare the human label (ground truth) with the system's
prediction across many clips, then count four outcomes:

| | predicted **yes** | predicted **no** |
|---|---|---|
| **actually yes** | true positive (TP) | false negative (FN) — *missed it* |
| **actually no** | false positive (FP) — *false alarm* | true negative (TN) |

- **Precision** = TP / (TP + FP) — *of the alerts we raised, how many were real?*
- **Recall** = TP / (TP + FN) — *of the real events, how many did we catch?*
- **F1** = harmonic mean of the two (a single number that punishes lopsided models).

Precision and recall trade off: a system that flags *everything* has perfect
recall but terrible precision (constant false alarms). A good proctor needs both.

## How to run

```bash
python -m evaluation.evaluate                 # uses the bundled sample data
python -m evaluation.evaluate --truth my_truth.csv --pred my_pred.csv
```

## How to make it real (your next data-collection task)

1. **Record clips.** Capture ~30–50 short webcam clips covering each behavior
   (looking away, phone visible, a second person, and plenty of "normal").
2. **Label them.** Fill in `ground_truth.csv` by hand — one row per clip, `1`/`0`
   per behavior. This *is* your test set.
3. **Predict.** Run the pipeline over the same clips and write its calls into
   `predictions.csv` with the identical `clip_id`s.
4. **Score.** Run the command above and read the report.
5. **Iterate.** Adjust thresholds, look at the failure cases, and watch the
   numbers move. Record the results in the README — that's your headline metric.

The `sample_ground_truth.csv` / `sample_predictions.csv` files show the exact
format and let the harness run out of the box.
