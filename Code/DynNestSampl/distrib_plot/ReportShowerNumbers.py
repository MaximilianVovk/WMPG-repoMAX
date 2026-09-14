from pathlib import Path
from collections import Counter, defaultdict
import re
import csv


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\Sporadic_final")

# Save CSV outputs here
OUTPUT_DIR = ROOT / "IAU_shower_summary"


# ============================================================
# REGEX
# ============================================================

# Meteor ID such as:
# 20220505_033342
# 20220505_033343
EVENT_ID_RE = re.compile(r"(\d{8}_\d{6})")

# Examples:
# IAU No.  =   -1
# IAU No.  =  7
IAU_NO_RE = re.compile(
    r"IAU\s+No\.\s*=\s*([+-]?\d+)",
    re.IGNORECASE
)

# Examples:
# IAU code =  ...
# IAU code =  PER
IAU_CODE_RE = re.compile(
    r"IAU\s+code\s*=\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE
)


# ============================================================
# FUNCTIONS
# ============================================================

def get_event_id(report_file):
    """
    Extract YYYYMMDD_HHMMSS from the filename first.
    If not found, try the parent folder.
    """

    match = EVENT_ID_RE.search(report_file.name)

    if match:
        return match.group(1)

    match = EVENT_ID_RE.search(report_file.parent.name)

    if match:
        return match.group(1)

    return report_file.parent.name


def get_group(report_file, event_id):
    """
    Determine the group from the directory structure.

    Example:

    ROOT
      uniform_Sporadic
        Stony
          Slow
            SlowSporad_uniqueEMCCD-rho-new
              20220505_033343
                20220505_033343_report.txt

    becomes:

    uniform_Sporadic\\Stony\\Slow\\SlowSporad_uniqueEMCCD-rho-new
    """

    parent = report_file.parent

    # If the immediate parent is the meteor folder, remove it
    if EVENT_ID_RE.search(parent.name):
        group_dir = parent.parent
    else:
        group_dir = parent

    try:
        relative = group_dir.relative_to(ROOT)
        return str(relative)
    except ValueError:
        return str(group_dir)


def read_shower_association(report_file):
    """
    Extract IAU No. and IAU code from one report.
    """

    try:
        text = report_file.read_text(
            encoding="utf-8",
            errors="replace"
        )
    except Exception as exc:
        return None, None, f"READ ERROR: {exc}"

    no_match = IAU_NO_RE.search(text)
    code_match = IAU_CODE_RE.search(text)

    iau_no = no_match.group(1).strip() if no_match else None
    iau_code = code_match.group(1).strip() if code_match else None

    return iau_no, iau_code, None


def shower_status(iau_no, iau_code):
    """
    Classify the shower result.
    """

    # Section could not be read/found
    if iau_no is None and iau_code is None:
        return "MISSING"

    # Normal sporadic/unassociated notation
    if iau_no == "-1":
        return "UNASSOCIATED"

    if iau_code is None:
        return "MISSING"

    clean_code = iau_code.strip()

    if clean_code in ("", "...", "..", "."):
        return "UNASSOCIATED"

    return "ASSOCIATED"


def display_code(iau_no, iau_code):
    """
    Label used for summaries.
    """

    status = shower_status(iau_no, iau_code)

    if status == "UNASSOCIATED":
        return "UNASSOCIATED"

    if status == "MISSING":
        return "MISSING"

    return iau_code.strip()


# ============================================================
# FIND ALL REPORT FILES
# ============================================================

print("=" * 100)
print("SEARCHING FOR *_report.txt FILES")
print("=" * 100)
print(f"Root: {ROOT}")
print()

report_files = sorted(
    path for path in ROOT.rglob("*")
    if path.is_file()
    and path.name.lower().endswith("_report.txt")
)

print(f"Found {len(report_files)} report files.")
print()


# ============================================================
# READ ALL REPORTS
# ============================================================

records = []

for report_file in report_files:

    event_id = get_event_id(report_file)
    group = get_group(report_file, event_id)

    iau_no, iau_code, error = read_shower_association(report_file)

    status = shower_status(iau_no, iau_code)
    summary_code = display_code(iau_no, iau_code)

    records.append({
        "event_id": event_id,
        "iau_no": iau_no,
        "iau_code": iau_code,
        "summary_code": summary_code,
        "status": status,
        "group": group,
        "report_file": str(report_file),
        "error": error,
    })


# ============================================================
# PRINT EVERY METEOR
# ============================================================

print()
print("=" * 100)
print("ALL METEORS")
print("=" * 100)

for rec in records:

    iau_no = rec["iau_no"] if rec["iau_no"] is not None else "NOT FOUND"
    iau_code = rec["iau_code"] if rec["iau_code"] is not None else "NOT FOUND"

    print(
        f"{rec['event_id']:20s} | "
        f"IAU No. = {iau_no:>5s} | "
        f"IAU code = {iau_code:<10s} | "
        f"{rec['status']:<12s} | "
        f"{rec['group']}"
    )


# ============================================================
# OVERALL SUMMARY
# ============================================================

print()
print()
print("=" * 100)
print("OVERALL SUMMARY")
print("=" * 100)

total_reports = len(records)

unique_events = set(rec["event_id"] for rec in records)

associated = [
    rec for rec in records
    if rec["status"] == "ASSOCIATED"
]

unassociated = [
    rec for rec in records
    if rec["status"] == "UNASSOCIATED"
]

missing = [
    rec for rec in records
    if rec["status"] == "MISSING"
]

print(f"Total report files     : {total_reports}")
print(f"Unique meteor IDs      : {len(unique_events)}")
print(f"Associated reports     : {len(associated)}")
print(f"Unassociated reports   : {len(unassociated)}")
print(f"Missing shower info    : {len(missing)}")


# ============================================================
# SHOWER TYPES FOUND
# ============================================================

print()
print("=" * 100)
print("SHOWER TYPES FOUND")
print("=" * 100)

overall_counts = Counter(
    rec["summary_code"]
    for rec in records
)

for code, count in sorted(
    overall_counts.items(),
    key=lambda x: (-x[1], x[0])
):
    print(f"{code:20s} : {count:4d}")


# ============================================================
# SHOW ASSOCIATED METEORS BY SHOWER
# ============================================================

print()
print("=" * 100)
print("METEORS GROUPED BY IAU CODE")
print("=" * 100)

events_by_code = defaultdict(list)

for rec in records:
    events_by_code[rec["summary_code"]].append(rec["event_id"])

for code in sorted(events_by_code):

    events = sorted(set(events_by_code[code]))

    print()
    print(f"{code}  ({len(events)} unique meteors)")
    print("-" * 60)

    for event in events:
        print(f"  {event}")


# ============================================================
# SUMMARY FOR EACH DIRECTORY GROUP
# ============================================================

print()
print()
print("=" * 100)
print("SUMMARY BY GROUP")
print("=" * 100)

grouped = defaultdict(list)

for rec in records:
    grouped[rec["group"]].append(rec)


for group in sorted(grouped):

    group_records = grouped[group]

    print()
    print("#" * 100)
    print(group)
    print("#" * 100)

    group_unique_events = set(
        rec["event_id"]
        for rec in group_records
    )

    group_counts = Counter(
        rec["summary_code"]
        for rec in group_records
    )

    print(f"Reports        : {len(group_records)}")
    print(f"Unique meteors : {len(group_unique_events)}")

    print()
    print("IAU codes:")

    for code, count in sorted(
        group_counts.items(),
        key=lambda x: (-x[1], x[0])
    ):
        print(f"  {code:20s} : {count:4d}")

    # --------------------------------------------------------
    # Unassociated
    # --------------------------------------------------------

    group_unassociated = sorted(set(
        rec["event_id"]
        for rec in group_records
        if rec["status"] == "UNASSOCIATED"
    ))

    print()
    print(
        f"UNASSOCIATED (-1 or ...): "
        f"{len(group_unassociated)}"
    )

    for event in group_unassociated:
        print(f"    {event}")

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    group_missing = sorted(set(
        rec["event_id"]
        for rec in group_records
        if rec["status"] == "MISSING"
    ))

    if group_missing:

        print()
        print(
            f"MISSING IAU INFORMATION: "
            f"{len(group_missing)}"
        )

        for event in group_missing:
            print(f"    {event}")


# ============================================================
# FIND METEORS WITH MULTIPLE REPORT FILES
# ============================================================

print()
print()
print("=" * 100)
print("MULTIPLE REPORT FILES / POSSIBLE DUPLICATES")
print("=" * 100)

reports_by_event_group = defaultdict(list)

for rec in records:

    key = (
        rec["group"],
        rec["event_id"]
    )

    reports_by_event_group[key].append(rec)


duplicates_found = False

for (group, event_id), event_records in reports_by_event_group.items():

    if len(event_records) > 1:

        duplicates_found = True

        print()
        print(f"Event: {event_id}")
        print(f"Group: {group}")
        print(f"Number of reports: {len(event_records)}")

        for rec in event_records:

            print(
                f"    {rec['report_file']} "
                f"-> No={rec['iau_no']}, "
                f"code={rec['iau_code']}"
            )


if not duplicates_found:
    print("No meteor has multiple *_report.txt files inside the same group.")


# ============================================================
# SAME METEOR APPEARING IN DIFFERENT GROUPS
# ============================================================

print()
print()
print("=" * 100)
print("METEOR IDs APPEARING IN MULTIPLE GROUPS")
print("=" * 100)

groups_by_event = defaultdict(set)

for rec in records:
    groups_by_event[rec["event_id"]].add(rec["group"])


multi_group_events = {
    event: groups
    for event, groups in groups_by_event.items()
    if len(groups) > 1
}

if multi_group_events:

    for event in sorted(multi_group_events):

        print()
        print(event)

        for group in sorted(multi_group_events[event]):
            print(f"    {group}")

else:
    print("No event ID occurs in multiple groups.")


# ============================================================
# SAVE CSV FILES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# CSV 1: one row per report
# ------------------------------------------------------------

all_csv = OUTPUT_DIR / "all_meteors_IAU_codes.csv"

with all_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "event_id",
            "iau_no",
            "iau_code",
            "status",
            "group",
            "report_file",
            "error",
        ]
    )

    writer.writeheader()

    for rec in records:

        writer.writerow({
            "event_id": rec["event_id"],
            "iau_no": rec["iau_no"],
            "iau_code": rec["iau_code"],
            "status": rec["status"],
            "group": rec["group"],
            "report_file": rec["report_file"],
            "error": rec["error"],
        })


# ------------------------------------------------------------
# CSV 2: overall shower counts
# ------------------------------------------------------------

summary_csv = OUTPUT_DIR / "IAU_code_summary.csv"

with summary_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "IAU_code",
        "count"
    ])

    for code, count in sorted(
        overall_counts.items(),
        key=lambda x: (-x[1], x[0])
    ):
        writer.writerow([
            code,
            count
        ])


# ------------------------------------------------------------
# CSV 3: shower counts per group
# ------------------------------------------------------------

group_csv = OUTPUT_DIR / "IAU_code_summary_by_group.csv"

with group_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "group",
        "IAU_code",
        "count"
    ])

    for group in sorted(grouped):

        counts = Counter(
            rec["summary_code"]
            for rec in grouped[group]
        )

        for code, count in sorted(counts.items()):

            writer.writerow([
                group,
                code,
                count
            ])


# ------------------------------------------------------------
# CSV 4: unassociated meteors
# ------------------------------------------------------------

unassociated_csv = OUTPUT_DIR / "unassociated_meteors.csv"

with unassociated_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "event_id",
        "IAU_no",
        "IAU_code",
        "group",
        "report_file"
    ])

    for rec in records:

        if rec["status"] == "UNASSOCIATED":

            writer.writerow([
                rec["event_id"],
                rec["iau_no"],
                rec["iau_code"],
                rec["group"],
                rec["report_file"],
            ])


# ============================================================
# FINISHED
# ============================================================

print()
print()
print("=" * 100)
print("DONE")
print("=" * 100)

print(f"Results saved to:")
print(f"  {OUTPUT_DIR}")

print()
print("Created:")
print(f"  {all_csv.name}")
print(f"  {summary_csv.name}")
print(f"  {group_csv.name}")
print(f"  {unassociated_csv.name}")