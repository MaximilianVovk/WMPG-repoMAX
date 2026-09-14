from pathlib import Path
from collections import defaultdict

# ============================================================
# CHANGE THIS TO THE FOLDER YOU WANT TO SEARCH
# ============================================================
ROOT_FOLDER = Path(r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\Sporadic_finalnew_12frag")

# Output text file
OUTPUT_FILE = ROOT_FOLDER / "duplicate_folder_names.txt"


# ============================================================
# FIND ALL SUBFOLDERS
# ============================================================

folders_by_name = defaultdict(list)

for folder in ROOT_FOLDER.rglob("*"):
    if folder.is_dir():
        folders_by_name[folder.name].append(folder)


# Keep only names appearing 2 or more times
duplicates = {
    name: paths
    for name, paths in folders_by_name.items()
    if len(paths) >= 2
}


# ============================================================
# WRITE RESULTS
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write(f"Root folder searched:\n{ROOT_FOLDER}\n")
    f.write("=" * 100 + "\n\n")

    f.write(f"Number of duplicated folder names: {len(duplicates)}\n\n")

    # Sort alphabetically by folder name
    for name in sorted(duplicates):

        paths = duplicates[name]

        f.write("=" * 100 + "\n")
        f.write(f"{name}   ({len(paths)} copies)\n")
        f.write("=" * 100 + "\n")

        for i, path in enumerate(paths, start=1):
            f.write(f"{i}. {path}\n")

        f.write("\n")


print(f"Done!")
print(f"Found {len(duplicates)} duplicated folder names.")
print(f"Results saved to:")
print(OUTPUT_FILE)