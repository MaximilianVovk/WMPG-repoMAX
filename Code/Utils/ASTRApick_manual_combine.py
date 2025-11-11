import os
import re
from io import StringIO
import pandas as pd

# ---------- file discovery ----------
def list_cam_ecsv(folder):
    """Return {basename: path} for files ending with 01G.ecsv or 02G.ecsv."""
    out = {}
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith("01G.ecsv") or f.endswith("02G.ecsv"):
                out[f] = os.path.join(root, f)
    return out

def find_pairs_by_name(folder1, folder2):
    """
    Returns list of (rms_path, astra_path) for files with identical names in both folders.
    The function doesn't assume which folder is RMS or ASTRA—just pairs by identical filename.
    """
    f1 = list_cam_ecsv(folder1)
    f2 = list_cam_ecsv(folder2)
    common = sorted(set(f1.keys()) & set(f2.keys()))
    missing_in_2 = sorted(set(f1.keys()) - set(f2.keys()))
    missing_in_1 = sorted(set(f2.keys()) - set(f1.keys()))

    if missing_in_2:
        print(f"[INFO] {len(missing_in_2)} files found in folder1 but not in folder2 (first 5): {missing_in_2[:5]}")
    if missing_in_1:
        print(f"[INFO] {len(missing_in_1)} files found in folder2 but not in folder1 (first 5): {missing_in_1[:5]}")

    return [(f1[name], f2[name]) for name in common]

# ---------- ECSV helpers (keep header intact) ----------
def read_ecsv_keep_header(path):
    """
    Returns (header_lines, df_as_str) where header lines start with '#'.
    Data are read as strings to preserve formatting; we'll parse datetime separately.
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i_data = 0
    for i, line in enumerate(lines):
        if not line.startswith('#'):
            i_data = i
            break
    header = lines[:i_data]
    df = pd.read_csv(StringIO("".join(lines[i_data:])), dtype=str)
    return header, df

def write_ecsv(header_lines, df, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(header_lines)
        df.to_csv(f, index=False)

# ---------- core merge (copy mag_data only) ----------
def align_and_copy_mag(source_df, target_df, direction_label):
    """
    Copy source_df['mag_data'] into target_df['mag_data'] for rows with matching datetime.
    Datetime matching is done via parsed timestamps to be robust to minor formatting.
    Returns (updated_df, warnings_text).
    """
    if 'datetime' not in source_df.columns or 'datetime' not in target_df.columns:
        raise ValueError(f"{direction_label}: both files must contain a 'datetime' column.")
    if 'mag_data' not in source_df.columns:
        raise ValueError(f"{direction_label}: source file lacks 'mag_data' column.")

    s_dt_str = source_df['datetime'].astype(str)
    t_dt_str = target_df['datetime'].astype(str)
    s_dt = pd.to_datetime(s_dt_str, errors='coerce')
    t_dt = pd.to_datetime(t_dt_str, errors='coerce')

    s = source_df.assign(_dt=s_dt).dropna(subset=['_dt'])
    t = target_df.assign(_dt=t_dt).dropna(subset=['_dt'])

    # Left-join source mag_data onto target by exact timestamp
    merged = t.merge(s[['_dt', 'mag_data']], on='_dt', how='left', suffixes=('', '_SRC'))

    # Diagnostics (missing timestamps across files)
    t_only = t.loc[~t['_dt'].isin(s['_dt']), '_dt']
    s_only = s.loc[~s['_dt'].isin(t['_dt']), '_dt']

    if 'mag_data_SRC' in merged.columns:
        has_src = merged['mag_data_SRC'].notna()
        merged.loc[has_src, 'mag_data'] = merged.loc[has_src, 'mag_data_SRC']

    # Drop helper columns
    updated = merged.drop(columns=[c for c in merged.columns if c.endswith('_SRC') or c == '_dt'])

    # Reinsert any rows that couldn't be parsed as datetime (if any)
    bad_t = target_df.loc[target_df['datetime'].isna() | (pd.to_datetime(target_df['datetime'], errors='coerce').isna())]
    if not bad_t.empty:
        updated = pd.concat([updated, bad_t], ignore_index=True)

    # Build warning text
    warnings = []
    if len(t_only) > 0:
        warnings.append(f"[WARNING] {direction_label}: {len(t_only)} datetimes in TARGET missing from SOURCE (first 5): " +
                        ", ".join(map(str, t_only.sort_values().head(5).tolist())))
    if len(s_only) > 0:
        warnings.append(f"[WARNING] {direction_label}: {len(s_only)} datetimes in SOURCE missing from TARGET (first 5): " +
                        ", ".join(map(str, s_only.sort_values().head(5).tolist())))
    return updated, "\n".join(warnings)

# ---------- high-level ----------
def process_pair_by_name(path1, path2, direction="ASTRA_to_RMS"):
    """
    direction:
      - 'ASTRA_to_RMS': copy mag_data from folder2 → folder1, save in folder1 as *_with_ASTRA_mag.ecsv
      - 'RMS_to_ASTRA': copy mag_data from folder1 → folder2, save in folder2 as *_ASTRA_RMS_mag.ecsv
    """
    hdr1, df1 = read_ecsv_keep_header(path1)
    hdr2, df2 = read_ecsv_keep_header(path2)

    base = os.path.basename(path1)  # same for both
    if direction == "ASTRA_to_RMS":
        # source=path2 (ASTRA), target=path1 (RMS)
        updated, warns = align_and_copy_mag(source_df=df2, target_df=df1, direction_label="ASTRA→RMS")
        out_path = os.path.join(os.path.dirname(path1), base.replace(".ecsv", "_with_ASTRA_mag.ecsv"))
        write_ecsv(hdr1, updated, out_path)
        if warns:
            print(warns)
        print(f"[OK] Wrote: {out_path}")
    elif direction == "RMS_to_ASTRA":
        # source=path1 (RMS), target=path2 (ASTRA)
        updated, warns = align_and_copy_mag(source_df=df1, target_df=df2, direction_label="RMS→ASTRA")
        out_path = os.path.join(os.path.dirname(path2), base.replace(".ecsv", "_ASTRA_RMS_mag.ecsv"))
        write_ecsv(hdr2, updated, out_path)
        if warns:
            print(warns)
        print(f"[OK] Wrote: {out_path}")
    else:
        raise ValueError("direction must be 'ASTRA_to_RMS' or 'RMS_to_ASTRA'.")

def update_all_by_name(folder1_path_RMS_files, folder2_path_ASTRA_files, direction="ASTRA_to_RMS"):
    """
    Pairs files by identical filename (ending in 01G.ecsv/02G.ecsv) across the two folders.
    Copies mag_data in the requested direction and writes outputs alongside the target folder.
    """
    pairs = find_pairs_by_name(folder1_path_RMS_files, folder2_path_ASTRA_files)
    print(f"Found {len(pairs)} matching file pairs by name.")
    for p1, p2 in pairs:
        print(f"Processing: {os.path.basename(p1)}")
        process_pair_by_name(p1, p2, direction=direction)

# ---------- example usage ----------
if __name__ == "__main__":
    # FOLDER 1 = RMS location, FOLDER 2 = ASTRA location (names are identical across folders)
    folder1_path_RMS_files = "/srv/public/mvovk/2ndPaper/reduction_CAMO+EMCCD/ORI/20191026_065838_skyfit2_ZB/ASTRA/manual"
    folder2_path_ASTRA_files = "/srv/public/mvovk/2ndPaper/reduction_CAMO+EMCCD/ORI/20191026_065838_skyfit2_ZB/ASTRA/ASTRA"

    # Copy ASTRA mag_data → RMS and save in RMS folder as *_with_ASTRA_mag.ecsv
    update_all_by_name(folder1_path_RMS_files, folder2_path_ASTRA_files, direction="RMS_to_ASTRA")

    # Or the reverse:
    # update_all_by_name(folder1_path_RMS_files, folder2_path_ASTRA_files, direction="RMS_to_ASTRA")
