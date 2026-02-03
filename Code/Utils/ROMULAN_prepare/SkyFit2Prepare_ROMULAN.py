"""
Romulan Skyfit2 Prepare Script
- Collects and organizes Romulan event .vid.bz2 files for cameras 01R and 02R.
- Decompresses to .vid into per-camera subfolders.
- Copies per-camera .config into same subfolders.
- Copies flats as flat.png (searches same night, otherwise searches previous nights).
- If event file not found at exact second, tries +1s and -1s.

Author: (adapted for Maximilian Vovk workflow)
"""

from __future__ import annotations

import os
import errno
import glob
import shutil
import argparse
import bz2
import datetime
import platform
import tarfile
from typing import Optional, Tuple, List


__version__ = "1.0"

WORK_DIR = os.getcwd()

# Romulan paths (Windows)
DATA_PATH_ROMULAN  = "/srv/meteor/romulan/events"
DATA_PATH_ROMULAN_OLD = "/srv/meteor/romulan/old_results/events"
FLATS_PATH_ROMULAN = "/srv/meteor/romulan/flats"
# Config files (as you specified)
CONFIG_01R = "/srv/meteor/reductions/romulan/2026_JB_remeasure/20090825_035144/01R/01R_2009.config"
CONFIG_02R = "/srv/meteor/reductions/romulan/2026_JB_remeasure/20090825_035144/02R/02R_2009.config"
PLATE_01R = "/srv/meteor/reductions/romulan/2026_JB_remeasure/20090825_035144/01R/platepar_cmn2010.cal"
PLATE_02R = "/srv/meteor/reductions/romulan/2026_JB_remeasure/20090825_035144/02R/platepar_cmn2010.cal"


# check the OS
if platform.system() == 'Windows':
    print('Windows OS detected')
    # Romulan paths (Windows)
    DATA_PATH_ROMULAN  = r"M:\romulan\events"
    DATA_PATH_ROMULAN_OLD = r"M:\romulan\old_results\events"
    FLATS_PATH_ROMULAN = r"M:\romulan\flats"
    # Config files (as you specified)
    CONFIG_01R = r"M:\reductions\romulan\2026_JB_remeasure\20090825_035144\01R\01R_2009.config"
    CONFIG_02R = r"M:\reductions\romulan\2026_JB_remeasure\20090825_035144\02R\02R_2009.config"
elif platform.system() == 'Linux':
    print('Linux OS detected')
else:
    print('OS not recognized, please check the paths in the code')

CAMS = ["01R", "02R"]

FLAT_FILE_NAME = "flat.png"


class ErrorTracker:
    """Indicates if any errors or warnings occurred during runtime."""

    def __init__(self) -> None:
        self.errors: List[str] = []

    def add(self, msg: str) -> None:
        self.errors.append(msg)
        print(msg)


def mkdirP(path: str) -> None:
    """Make a directory if it does not exist."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return
        raise


def adjust_event_time(event_name: str, seconds_add_sub: int = 0) -> str:
    """
    Adjusts event time by adding/subtracting seconds.
    event_name: 'YYYYMMDD_HHMMSS'
    """
    dt = datetime.datetime.strptime(event_name, "%Y%m%d_%H%M%S")
    dt += datetime.timedelta(seconds=seconds_add_sub)
    return dt.strftime("%Y%m%d_%H%M%S")


def formatEventNames(event_list: List[str]) -> List[str]:
    """Formats event names to YYYYMMDD_HHMMSS (supports 'YYYYMMDD HH:MM:SS' style too)."""

    def proper_format(name: str) -> bool:
        if len(name) == 15 and name.count("_") == 1:
            d, t = name.split("_")
            try:
                int(d)
                int(t)
                return True
            except Exception:
                return False
        return False

    out = []
    for raw in event_list:
        ev = raw.strip()

        # Web-like format "YYYYMMDD 03:51:44" or "YYYYMMDD_03:51:44"
        if ":" in ev:
            date = ev[:8]
            parts = ev[8:].strip().replace("_", " ").split(":")
            if len(parts) >= 3:
                ev = date + "_" + parts[0].zfill(2) + parts[1].zfill(2) + parts[2].zfill(2)

        # Clip any extra junk
        if len(ev) > 15:
            ev = ev[:15]

        if proper_format(ev):
            out.append(ev)
        else:
            print(f"!!! Event '{raw}' could not be processed (expected YYYYMMDD_HHMMSS).")

    return out


def night_from_event(event_name: str) -> str:
    return event_name.split("_")[0]


def _candidate_dirs(base: str, night_name: str) -> List[str]:
    """Try both base/night and base, because Romulan layout may differ."""
    return [
        os.path.join(base, night_name),
        base,
    ]


def find_event_file(event_name: str, cam: str, error_track: ErrorTracker) -> Tuple[Optional[str], Optional[str]]:
    """
    Find event file for a given camera.
    Tries: event time, +1s, -1s
    Returns: (filepath, used_event_time)
    """

    night = night_from_event(event_name)
    patterns_for_cam = lambda ev: [
        # Expected style: ev_YYYYMMDD_HHMMSS A _01R.vid.bz2 (A can vary) -> '*' after timestamp catches it.
        f"ev_{ev}*_{cam}.vid.bz2",
        f"ev_{ev}*{cam}.vid.bz2",
    ]

    for sec_adj in [0, 1, -1]:
        ev_try = adjust_event_time(event_name, sec_adj)
        for d in _candidate_dirs(DATA_PATH_ROMULAN, night):
            for pat in patterns_for_cam(ev_try):
                hits = glob.glob(os.path.join(d, pat))
                if hits:
                    # pick most recent if multiple
                    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    if sec_adj != 0:
                        error_track.add(
                            f"{event_name}: WARNING! {cam} event not found at exact second. "
                            f"SOLVED by using {ev_try}."
                        )
                    return hits[0], ev_try

    patterns_for_cam = lambda ev: [
        # Expected style: ev_YYYYMMDD_HHMMSS A _01R.vid.bz2 (A can vary) -> '*' after timestamp catches it.
        f"ev_{ev}*_{cam}.tar",
        f"ev_{ev}*{cam}.tar",
    ]
    for sec_adj in [0, 1, -1]:
        ev_try = adjust_event_time(event_name, sec_adj)
        for d in _candidate_dirs(DATA_PATH_ROMULAN_OLD, night):
            for pat in patterns_for_cam(ev_try):
                hits = glob.glob(os.path.join(d, pat))
                if hits:
                    # pick most recent if multiple
                    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    if sec_adj != 0:
                        error_track.add(
                            f"{event_name}: WARNING! {cam} event not found at exact second. "
                            f"SOLVED by using {ev_try}."
                        )
                    return hits[0], ev_try

    error_track.add(f"{event_name}: WARNING! No Romulan event file found for {cam} (also tried +/- 1s).")
    return None, None


def decompress_bz2_to_vid(bz2_path: str, dest_vid_path: str) -> None:
    """Decompress bz2 file to .vid (overwrite if exists)."""
    mkdirP(os.path.dirname(dest_vid_path))
    with bz2.open(bz2_path, "rb") as fin:
        data = fin.read()
    # Overwrite
    with open(dest_vid_path, "wb") as fout:
        fout.write(data)

def copy_plate(cam: str, dest_dir: str, error_track: ErrorTracker) -> None:
    """Copy the per-camera plate file into dest_dir."""
    if cam == "01R":
        src = PLATE_01R
    elif cam == "02R":
        src = PLATE_02R
    else:
        error_track.add(f"Unknown camera '{cam}' for plate copy.")
        return

    if not os.path.isfile(src):
        error_track.add(f"WARNING! Plate for {cam} not found at: {src}")
        return

    mkdirP(dest_dir)
    dest = os.path.join(dest_dir, os.path.basename(src))
    shutil.copy2(src, dest)

def copy_config(cam: str, dest_dir: str, error_track: ErrorTracker) -> None:
    """Copy the per-camera config file into dest_dir."""
    if cam == "01R":
        src = CONFIG_01R
    elif cam == "02R":
        src = CONFIG_02R
    else:
        error_track.add(f"Unknown camera '{cam}' for config copy.")
        return

    if not os.path.isfile(src):
        error_track.add(f"WARNING! Config for {cam} not found at: {src}")
        return

    mkdirP(dest_dir)
    dest = os.path.join(dest_dir, os.path.basename(src))
    shutil.copy2(src, dest)


def _flat_patterns(night_name: str, cam: str) -> List[str]:
    """
    Robust flat filename patterns.
    You can tweak these if you later confirm the exact naming convention.
    """
    return [
        f"flat*{night_name}*{cam}*.png",
        f"*{night_name}*flat*{cam}*.png",
        f"flat*{cam}*{night_name}*.png",
        f"*{cam}*{night_name}*.png",
        f"flat*{cam}*.png",
        f"*{cam}*flat*.png",
        "flat*.png",
        "*.png",
    ]


def find_flat_file(night_name: str, cam: str) -> Optional[str]:
    """Find the best flat candidate for a given night+camera (or None)."""
    for d in _candidate_dirs(FLATS_PATH_ROMULAN, night_name):
        if not os.path.isdir(d):
            continue
        candidates: List[str] = []
        for pat in _flat_patterns(night_name, cam):
            candidates.extend(glob.glob(os.path.join(d, pat)))

        # Filter out obvious non-flats if any show up (weak heuristic)
        candidates = [c for c in candidates if "dark" not in os.path.basename(c).lower()]

        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]

    return None


def copy_flat_with_lookback(event_name: str, cam: str, dest_dir: str, lookback_days: int, error_track: ErrorTracker) -> None:
    """Copy flat as flat.png (same night or previous nights)."""
    night = night_from_event(event_name)

    flat = find_flat_file(night, cam)
    if flat:
        shutil.copy2(flat, os.path.join(dest_dir, FLAT_FILE_NAME))
        return

    error_track.add(f"{event_name}: WARNING! Flat not found for {cam} on night {night}. Searching previous nights...")

    fmt = "%Y%m%d"
    night_dt = datetime.datetime.strptime(night, fmt)

    for i in range(1, lookback_days + 1):
        prev = (night_dt - datetime.timedelta(days=i)).strftime(fmt)
        flat_prev = find_flat_file(prev, cam)
        if flat_prev:
            shutil.copy2(flat_prev, os.path.join(dest_dir, FLAT_FILE_NAME))
            error_track.add(f"{event_name}: SOLVED! Using flat for {cam} from night {prev} -> {flat_prev}")
            return

    error_track.add(
        f"{event_name}: WARNING! No flat found for {cam} within previous {lookback_days} nights "
        f"(starting from {night})."
    )


def processRomulanEvents(event_list: List[str], out_root: str, lookback_days: int, error_track: ErrorTracker) -> None:
    """Main Romulan processing: create folder, decompress .vid, copy flats and configs."""
    for event_name in event_list:
        event_dir = os.path.join(out_root, event_name)


        for cam in CAMS:
            cam_dir = os.path.join(event_dir, cam)

            # 1) Find event .vid.bz2 (with +/- 1s fallback)
            bz2_path, used_event_time = find_event_file(event_name, cam, error_track)
            if not bz2_path:
                # keep camera dir (maybe you still want flats/config), but log
                error_track.add(f"{event_name}: NOTE: Skipping .vid extraction for {cam} (event file missing).")
                continue
            else:
                mkdirP(event_dir)
                mkdirP(cam_dir)
                print(f"{event_name}: {cam} event file found decompressing : {bz2_path}")
                # 2) Decompress to .vid in cam dir
                # keep original filename, just drop .bz2
                base_name = os.path.basename(bz2_path)
                if base_name.endswith(".bz2"):
                    vid_name = base_name[:-4]
                elif base_name.endswith(".tar"):
                    vid_name = base_name[:-4]  # assuming .tar
                else:
                    vid_name = base_name + ".vid"

                dest_vid = os.path.join(cam_dir, vid_name)
                try:
                    print(f"{event_name}: {cam} decompressing .bz2 to .vid -> {dest_vid}")
                    decompress_bz2_to_vid(bz2_path, dest_vid)
                    print(f"{event_name}: {cam} decompressed -> {dest_vid}")
                except Exception as e:
                    error_track.add(f"{event_name}: ERROR decompressing {cam} file '{bz2_path}': {e}")
                    try:
                        print(f"{event_name}: {cam} trying to extract .tar file")
                        # in case of .tar, try extracting
                        with tarfile.open(bz2_path, 'r') as tar:
                            tar.extractall(path=cam_dir)
                        # look for folders in cam_dir
                        folders = [f for f in os.listdir(cam_dir) if os.path.isdir(os.path.join(cam_dir, f))]
                        if folders:
                            #look if in any of the folders there are .png files
                            for folder in folders:
                                folder_path = os.path.join(cam_dir, folder)
                                png_files = glob.glob(os.path.join(folder_path, "*.png"))
                                if png_files:
                                    for png in png_files:
                                        shutil.move(png, cam_dir)
                                    shutil.rmtree(folder_path)

                        print(f"{event_name}: {cam} extracted .tar -> {cam_dir}")
                    except Exception as e2:
                        error_track.add(f"{event_name}: ERROR extracting {cam} .tar file '{bz2_path}': {e2}")
                        continue

                # If second adjusted, optionally note which timestamp ended up used
                if used_event_time and used_event_time != event_name:
                    # purely informational
                    pass

            # 3) Copy flat (with lookback)
            try:
                copy_flat_with_lookback(event_name, cam, cam_dir, lookback_days, error_track)
            except Exception as e:
                error_track.add(f"{event_name}: ERROR copying flat for {cam}: {e}")

            # 4) Copy config
            try:
                copy_config(cam, cam_dir, error_track)
            except Exception as e:
                error_track.add(f"{event_name}: ERROR copying config for {cam}: {e}")

            # 4) Copy plate
            try:
                copy_plate(cam, cam_dir, error_track)
            except Exception as e:
                error_track.add(f"{event_name}: ERROR copying plate for {cam}: {e}")



def write_errors(error_track: ErrorTracker, out_root: str) -> None:
    if not error_track.errors:
        print("Successfully done!")
        return

    path = os.path.join(out_root, "errors_Skyfit2Prepare_romulan.txt")
    with open(path, "w", encoding="utf-8") as f:
        for msg in error_track.errors:
            f.write(msg + "\n")

    print("---------------------")
    print("Finished with warnings/errors. See:")
    print(path)


if __name__ == "__main__":
    error_track = ErrorTracker()

    ap = argparse.ArgumentParser(
        description=(
            "Prepare Romulan events for SkyFit2: decompress .vid.bz2 into per-camera folders, "
            "copy flats (with previous-night fallback) and copy per-camera configs."
        )
    )
    ap.add_argument(
        "meteors",
        help='Comma-separated events, e.g. "20090825_035144,20090826_012345" or web-like time formats.',
    )
    ap.add_argument(
        "--outdir",
        default=WORK_DIR,
        help="Output root directory (default: current directory). Each event becomes <outdir>/<event>/...",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="How many previous nights to search for flats (default: 30).",
    )

    args = ap.parse_args()

    # Parse and format events
    event_list = formatEventNames(args.meteors.split(","))
    print("Events:", event_list)

    processRomulanEvents(event_list, args.outdir, args.lookback_days, error_track)
    write_errors(error_track, args.outdir)
