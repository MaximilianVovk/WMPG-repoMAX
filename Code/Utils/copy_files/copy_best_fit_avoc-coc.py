from pathlib import Path
import pandas as pd
import shutil


# ============================================================
# USER SETTINGS
# ============================================================

# CSV whose FIRST COLUMN contains the meteor IDs
csv_file = Path(
    r"C:\Users\maxiv\Documents\UWO\Papers\0.6)Tah_erc\ReductionEMCCD\TAHmeteor_event_links_notes.xlsx"
)

# Main results directory containing all meteor folders
search_root = Path(
    r"C:\Users\maxiv\Documents\UWO\Papers\0.6)Tah_erc\Results-dynesty\EMCCD\EMCCD-cz"
)

# Put ALL copied plots into this single folder
output_dir = Path(
    r"C:\Users\maxiv\Documents\UWO\Papers\0.6)Tah_erc\Results-dynesty\EMCCD\EMCCD-cz_plot"
)


# ============================================================
# LOAD METEOR IDs
# ============================================================
try:
    df = pd.read_csv(csv_file)

    # First column contains event IDs
    meteor_ids = (
        df.iloc[:, 0]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    print(f"Loaded {len(meteor_ids)} meteor IDs from CSV.")
except Exception as e:
    print(f"Error loading meteor IDs from CSV: {e}")
    meteor_ids = []


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND ALL BEST-FIT LumLag PLOTS ONCE
# ============================================================

print("\nSearching for *_best_fit_vs_height.png files...")

plot_files = []

for path in search_root.rglob("*"):
    if (
        path.is_file()
        and path.parent.name.lower() == "fit_plots"
        and path.name.lower().endswith("_best_fit_vs_height.png")
    ):
        plot_files.append(path)

print(f"Found {len(plot_files)} candidate best-fit plots.\n")

if len(meteor_ids) == 0:
    # add all the plot_files xxxxxxx_xxxxxx numbers in the plot_files to the meteor_ids list
    meteor_ids = []
    for plot_file in plot_files:
        # Extract the meteor ID from the filename (assuming it's the part before the first underscore)
        meteor_id0 = plot_file.name.split("_")[0]
        meteor_id1 = plot_file.name.split("_")[1]
        meteor_ids.append(meteor_id0+"_"+meteor_id1)


# ============================================================
# MATCH METEOR IDs AND COPY
# ============================================================

found = []
missing = []
multiple = []

for meteor_id in meteor_ids:

    # Find files where meteor ID appears either in the filename
    # OR somewhere in the folder path.
    matches = [
        path for path in plot_files
        if (
            meteor_id.lower() in path.name.lower()
            or meteor_id.lower() in str(path.parent).lower()
        )
    ]

    if len(matches) == 0:
        print(f"[MISSING] {meteor_id}")
        missing.append(meteor_id)
        continue

    if len(matches) > 1:
        print(f"[MULTIPLE] {meteor_id}: found {len(matches)} plots")
        multiple.append((meteor_id, matches))

    for n, source in enumerate(matches, start=1):

        # Give copied file a clean name based on meteor ID.
        # If multiple matches exist, number them.
        if len(matches) == 1:
            output_name = f"{meteor_id}_best_fit_vs_height.png"
        else:
            output_name = f"{meteor_id}_best_fit_vs_height_{n}.png"

        destination = output_dir / output_name

        shutil.copy2(source, destination)

        print(f"[COPIED] {meteor_id}")
        print(f"         {source}")
        print(f"      -> {destination}")

        found.append((meteor_id, source, destination))


# ============================================================
# SAVE REPORT
# ============================================================

report_file = output_dir / "copy_report.txt"

with open(report_file, "w", encoding="utf-8") as f:

    f.write("AVOCADO BEST-FIT vs HEIGHT PLOT COPY REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Meteors in CSV: {len(meteor_ids)}\n")
    f.write(f"Plots copied:   {len(found)}\n")
    f.write(f"Missing:        {len(missing)}\n")
    f.write(f"Multiple:       {len(multiple)}\n\n")

    f.write("COPIED\n")
    f.write("-" * 70 + "\n")

    for meteor_id, source, destination in found:
        f.write(f"{meteor_id}\n")
        f.write(f"  Source: {source}\n")
        f.write(f"  Copy:   {destination}\n\n")

    f.write("\nMISSING\n")
    f.write("-" * 70 + "\n")

    for meteor_id in missing:
        f.write(f"{meteor_id}\n")

    f.write("\nMULTIPLE MATCHES\n")
    f.write("-" * 70 + "\n")

    for meteor_id, matches in multiple:
        f.write(f"{meteor_id}\n")
        for match in matches:
            f.write(f"  {match}\n")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(f"Meteors requested : {len(meteor_ids)}")
print(f"Plots copied      : {len(found)}")
print(f"Missing meteors   : {len(missing)}")
print(f"Multiple matches  : {len(multiple)}")

print(f"\nAll plots are in:\n{output_dir}")
print(f"\nReport saved to:\n{report_file}")