#!/usr/bin/env python3
import os
import re
import shutil
import argparse


def extract_event_id(filename: str) -> str | None:
    """
    Extract the 'YYYYMMDD_hhmmss' part from a filename.
    Example: '20211214_075322_trajectory.pickle' -> '20211214_075322'
    """
    m = re.search(r'(\d{8}_\d{6})', filename)
    return m.group(1) if m else None


def get_unique_dest_dir(output_root: str, event_id: str) -> str:
    """
    Return a destination directory path under output_root that is unique.
    If 'YYYYMMDD_hhmmss' exists, try 'YYYYMMDD_hhmmss_1', '_2', etc.
    """
    base = os.path.join(output_root, event_id)
    if not os.path.exists(base):
        return base

    idx = 1
    while True:
        candidate = f"{base}_{idx}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def collect_files(input_root: str, output_root: str) -> None:
    """
    Walk input_root; for each .pickle file found:
      - determine event_id from filename
      - create an output subfolder named event_id (or event_id_N if needed)
      - copy matching files (same event_id + .pickle/.png/*_report.txt)
    """
    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)

    for dirpath, dirnames, filenames in os.walk(input_root):
        # All pickle files in this directory
        pickle_files = [f for f in filenames if f.lower().endswith(".pickle")]

        for pickle_name in pickle_files:
            event_id = extract_event_id(pickle_name)
            if not event_id:
                # Skip pickle files that don't follow the naming pattern
                continue

            dest_dir = get_unique_dest_dir(output_root, event_id)
            os.makedirs(dest_dir, exist_ok=True)

            # Copy all relevant files that belong to this event_id
            for fname in filenames:
                # Only files that contain the same event_id in the name
                if event_id not in fname:
                    continue

                lower = fname.lower()
                if not (lower.endswith(".pickle") or
                        lower.endswith(".png") or
                        fname.endswith("_report.txt")):
                    continue

                src = os.path.join(dirpath, fname)
                dst = os.path.join(dest_dir, fname)
                shutil.copy2(src, dst)
                print(f"Copied {src} -> {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk a directory tree, find .pickle trajectories, "
                    "and copy them with related PNG and *_report.txt files "
                    "into event-based folders in an output directory."
    )
    parser.add_argument("--input_root", default=r"N:\mvovk\GEM\SKYFIT2",
                        help="Root folder to search in")
    parser.add_argument("--output_root", default=r"C:\Users\maxiv\Documents\UWO\Papers\0.3)Phaethon\GEM\EMCCD",
                        help="Folder where results are stored")
    args = parser.parse_args()

    collect_files(args.input_root, args.output_root)



