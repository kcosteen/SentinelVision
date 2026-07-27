import csv
import os
from datetime import datetime


log_file = os.path.join("logs", "events.csv")


# The logs directory is NOT in version control -- it holds session output, which
# is nobody else's business -- so a fresh clone has no logs/ at all. Create it
# rather than assuming it, or importing this module is an instant crash.
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Create file if it doesn't exist
if not os.path.exists(log_file):

    with open(log_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "timestamp",
                "event",
                "score_added",
                "total_score",
                "details"
            ]
        )


def log_event(event, score_added, total_score, details=""):

    with open(log_file, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                datetime.now(),
                event,
                score_added,
                total_score,
                details
            ]
        )