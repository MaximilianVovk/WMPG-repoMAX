from pathlib import Path
import shutil


# ============================================================
# FOLDERS
# ============================================================

# OLD results:
# contains all events and the CORRECT posterior_backup.pkl.gz files
SOURCE_ROOT = Path(
    r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\Uniform-Meteor-results\Faint-meteor\Stony"
)

# NEW results:
# contains only the events you want to keep,
# but currently has the WRONG posterior_backup.pkl.gz files
DEST_ROOT = Path(
    r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\Sporadic_finalnew_12frag\Stony\2_frag"
)


# ============================================================
# STORAGE FOR REPORT
# ============================================================

copied = []
missing_source_folder = []
missing_posterior = []


# ============================================================
# LOOP THROUGH ONLY THE EVENTS PRESENT IN THE NEW DIRECTORY
# ============================================================

for dest_folder in DEST_ROOT.rglob("*"):

    if not dest_folder.is_dir():
        continue

    # We only care about folders that contain a posterior backup
    # in the NEW directory.
    dest_posteriors = list(
        dest_folder.glob("*_posterior_backup.pkl.gz")
    )

    if not dest_posteriors:
        continue

    # Relative path of this event inside DEST_ROOT
    relative_path = dest_folder.relative_to(DEST_ROOT)

    # Expected corresponding folder in the old results
    source_folder = SOURCE_ROOT / relative_path

    print("\n" + "=" * 80)
    print(f"EVENT: {dest_folder.name}")
    print("=" * 80)

    # ========================================================
    # CASE 1: SOURCE FOLDER DOES NOT EXIST
    # ========================================================

    if not source_folder.exists():

        print("MISSING SOURCE FOLDER")
        print()
        print("Destination event folder:")
        print(f"  {dest_folder}")
        print()
        print("Expected source folder:")
        print(f"  {source_folder}")

        missing_source_folder.append(
            (
                str(dest_folder),
                str(source_folder)
            )
        )

        continue

    # ========================================================
    # CASE 2: SOURCE FOLDER EXISTS, BUT ZIP IS MISSING
    # ========================================================

    source_posteriors = list(
        source_folder.glob("*_posterior_backup.pkl.gz")
    )

    if not source_posteriors:

        print("MISSING POSTERIOR ZIP")
        print()
        print("Destination event folder:")
        print(f"  {dest_folder}")
        print()
        print("Source folder exists:")
        print(f"  {source_folder}")
        print()
        print("But no *_posterior_backup.pkl.gz file was found.")

        missing_posterior.append(
            (
                str(dest_folder),
                str(source_folder)
            )
        )

        continue

    # ========================================================
    # CASE 3: COPY THE CORRECT POSTERIOR FILE
    # ========================================================

    for source_file in source_posteriors:

        dest_file = dest_folder / source_file.name

        print("COPYING")
        print()
        print(f"FROM:")
        print(f"  {source_file}")
        print()
        print(f"TO:")
        print(f"  {dest_file}")

        # Overwrites the existing incorrect file
        shutil.copy2(source_file, dest_file)

        copied.append(
            (
                str(source_file),
                str(dest_file)
            )
        )


# ============================================================
# CREATE REPORT FILE
# ============================================================

report_file = DEST_ROOT / "posterior_backup_copy_report.txt"

with open(report_file, "w", encoding="utf-8") as f:

    f.write("POSTERIOR BACKUP COPY REPORT\n")
    f.write("=" * 100 + "\n\n")

    f.write("SOURCE ROOT:\n")
    f.write(f"{SOURCE_ROOT}\n\n")

    f.write("DESTINATION ROOT:\n")
    f.write(f"{DEST_ROOT}\n\n")

    # --------------------------------------------------------
    # COPIED
    # --------------------------------------------------------

    f.write("=" * 100 + "\n")
    f.write(f"COPIED FILES: {len(copied)}\n")
    f.write("=" * 100 + "\n\n")

    for source_file, dest_file in copied:

        f.write("FROM:\n")
        f.write(f"{source_file}\n")

        f.write("TO:\n")
        f.write(f"{dest_file}\n")

        f.write("-" * 100 + "\n")

    # --------------------------------------------------------
    # MISSING SOURCE FOLDERS
    # --------------------------------------------------------

    f.write("\n")
    f.write("=" * 100 + "\n")
    f.write(
        f"MISSING SOURCE FOLDERS: "
        f"{len(missing_source_folder)}\n"
    )
    f.write("=" * 100 + "\n\n")

    for dest_folder, expected_source in missing_source_folder:

        f.write("DESTINATION EVENT FOLDER:\n")
        f.write(f"{dest_folder}\n\n")

        f.write("EXPECTED SOURCE FOLDER:\n")
        f.write(f"{expected_source}\n")

        f.write("-" * 100 + "\n")

    # --------------------------------------------------------
    # SOURCE EXISTS BUT POSTERIOR ZIP IS MISSING
    # --------------------------------------------------------

    f.write("\n")
    f.write("=" * 100 + "\n")
    f.write(
        f"SOURCE FOLDER EXISTS BUT POSTERIOR ZIP IS MISSING: "
        f"{len(missing_posterior)}\n"
    )
    f.write("=" * 100 + "\n\n")

    for dest_folder, source_folder in missing_posterior:

        f.write("DESTINATION EVENT FOLDER:\n")
        f.write(f"{dest_folder}\n\n")

        f.write("SOURCE FOLDER:\n")
        f.write(f"{source_folder}\n\n")

        f.write("MISSING FILE:\n")
        f.write("*_posterior_backup.pkl.gz\n")

        f.write("-" * 100 + "\n")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINISHED")
print("=" * 80)

print(f"Files copied:                         {len(copied)}")
print(f"Missing source folders:               {len(missing_source_folder)}")
print(f"Source exists but zip missing:         {len(missing_posterior)}")

print()
print("Report saved to:")
print(report_file)


# ============================================================
# PRINT MISSING EVENTS AGAIN AT THE END
# ============================================================

if missing_source_folder:

    print("\n")
    print("=" * 80)
    print("EVENTS WITH MISSING SOURCE FOLDER")
    print("=" * 80)

    for dest_folder, expected_source in missing_source_folder:

        print()
        print("DESTINATION:")
        print(dest_folder)

        print("EXPECTED SOURCE:")
        print(expected_source)


if missing_posterior:

    print("\n")
    print("=" * 80)
    print("EVENTS WHERE SOURCE EXISTS BUT ZIP IS MISSING")
    print("=" * 80)

    for dest_folder, source_folder in missing_posterior:

        print()
        print("DESTINATION:")
        print(dest_folder)

        print("SOURCE:")
        print(source_folder)

        print("MISSING:")
        print("*_posterior_backup.pkl.gz")