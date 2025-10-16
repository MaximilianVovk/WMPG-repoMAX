#!/usr/bin/env python3
import csv, os, sys

def bump_mag(path, delta=0.5):
    tmp = path + ".tmp"

    with open(path, "r", newline="") as f_in, open(tmp, "w", newline="") as f_out:
        # csv.writer ensures one newline per row
        writer = csv.writer(f_out, lineterminator="\n")
        mag_idx = None
        header_seen = False

        for raw in f_in:
            # Preserve comments/meta verbatim
            if raw.lstrip().startswith("#"):
                f_out.write(raw)
                continue

            # Header row (starts with 'datetime,') — write as-is, detect mag column
            if not header_seen and raw.strip().lower().startswith("datetime,"):
                header_seen = True
                f_out.write(raw)  # keep header exactly
                hdr = next(csv.reader([raw], skipinitialspace=True))
                # detect by name (accept 'mag_data' or 'mag')
                for i, name in enumerate(hdr):
                    n = name.strip().strip("'").strip('"').lower()
                    if n in ("mag_data", "mag"):
                        mag_idx = i
                        break
                if mag_idx is None:
                    mag_idx = 10  # fallback
                continue

            # Blank lines pass-through
            if not raw.strip():
                f_out.write(raw)
                continue

            # Data row: bump mag, write with a newline
            row = next(csv.reader([raw], skipinitialspace=True))
            if mag_idx < len(row):
                s = row[mag_idx].strip()
                if s.startswith("+"):
                    s = s[1:]
                if s:
                    row[mag_idx] = f"{float(s) + delta:+.2f}"
            writer.writerow(row)  # always emits a newline

    os.replace(tmp, path)

if __name__ == "__main__":
    # Usage: python bump_mag.py /path/to/file.ecsv [delta]
    infile = r"/srv/public/mvovk/reductions/CAP_spor/20190804_060242_skyfit2_clouds_IAU_goodEMCCD_DONE_NICK/20190804_060242_mir/2019-08-04T06_02_42_Mirfit_02T.ecsv" # sys.argv[1]
    delta = 0.5 # float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    bump_mag(infile, delta)
    print("Done")
