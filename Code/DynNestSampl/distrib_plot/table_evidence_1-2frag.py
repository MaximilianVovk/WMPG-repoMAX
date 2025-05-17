#!/usr/bin/env python3
import os
import re
import math

def get_logz_values(root_dir):
    """
    Walk through root_dir and its subdirectories to find files named log_YYYYMMDD_HHMMSS.txt,
    extract the logz value and its uncertainty, and return a dict mapping timestamps to (value, uncertainty).
    """
    pattern_file = re.compile(r"log_(\d{8}_\d{6})")
    pattern_logz = re.compile(r"logz:\s*([\-\d\.]+)\s*\+/-\s*([\-\d\.]+)")
    values = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            m = pattern_file.match(fname)
            if not m:
                continue
            timestamp = m.group(1)
            file_path = os.path.join(dirpath, fname)
            with open(file_path, 'r') as f:
                for line in f:
                    m2 = pattern_logz.search(line)
                    if m2:
                        val = float(m2.group(1))
                        unc = float(m2.group(2))
                        values[timestamp] = (val, unc)
                        break
    return values


def main():
    # Define the two model directories
    dir1 = r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\CAP-1frag-0417"
    dir2 = r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\CAP-2frag-0417"

    # Extract logz values for each
    vals1 = get_logz_values(dir1)
    vals2 = get_logz_values(dir2)

    # Find common timestamps
    common = sorted(set(vals1.keys()) & set(vals2.keys()))

    # Begin LaTeX table
    header = []
    header.append(r"\begin{table}[htbp]")
    header.append(r"    \centering")
    header.append(r"    \renewcommand{\arraystretch}{1.2}")
    header.append(r"    \setlength{\tabcolsep}{4pt}")
    header.append(r"    \caption{Summary of logz values for 1frag and 2frag models. The Difference column shows 1frag (logz) - 2frag (logz). The last column confirm if a second fragmentation was observed with CAMO narrow-field camera.}")
    header.append(r"    \label{tab:frag_logz_summary}")
    header.append(r"    \resizebox{\textwidth}{!}{%")
    header.append(r"    \begin{tabular}{|l|c|c|c|c|}")
    header.append(r"    \hline")
    header.append(r"    Timestamp & 1frag (logz) & 2frag (logz) & Difference & 2fr \\")
    header.append(r"    \hline")
    rows = []
    for ts in common:
        v1, u1 = vals1[ts]
        v2, u2 = vals2[ts]
        # # Compute difference: ln(e^(v1) - e^(v2))
        # exp1 = math.exp(v1)
        # exp2 = math.exp(v2)
        # if exp1 == exp2:
        #     diff = 0
        # else:
        #     # diff = float('nan')
        #     diff = math.log(abs(exp1 - exp2))
        diff = v1 - v2
        # change from ts the from _ to \_
        ts = ts.replace('_', r'\_')

        col1 = f"${v1:.3f} \pm {u1:.3f}$"
        col2 = f"${v2:.3f} \pm {u2:.3f}$"
        col3 = f"${diff:.3f}$" # \ln(e^{{{v1:.3f}}} - e^{{{v2:.3f}}}) = 
        rows.append(f"    {ts} & {col1} & {col2} & {col3} & No\\\\")
        rows.append(r"    \hline")

    footer = []
    footer.append(r"    \end{tabular}}")
    footer.append(r"\end{table}")

    # Output to stdout or write to a file
    output = '\n'.join(header + rows + footer)
    print(output)

if __name__ == '__main__':
    main()
