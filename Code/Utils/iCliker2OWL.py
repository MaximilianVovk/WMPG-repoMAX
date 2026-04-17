#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def find_email_column(columns):
    # Exact match first
    for c in columns:
        if c.strip().lower() == "email":
            return c
    # Fallback: any column containing "email"
    for c in columns:
        if "email" in c.strip().lower():
            return c
    return None


def transform_email_to_orgdefinedid(df: pd.DataFrame, email_col: str) -> pd.DataFrame:
    s = df[email_col].astype(str).str.strip()

    # exaptions if jamesbarlow26@yahoo.com is present swap it with jbarlo4@uwo.ca
    s = s.replace("jamesbarlow26@yahoo.com", "jbarlo4@uwo.ca")

    # Remove @uwo.ca only if it is at the end (case-insensitive)
    s = s.str.replace(r"(?i)@uwo\.ca$", "", regex=True)

    # delete the Username column if it already exists
    if "Username" in df.columns:
        df = df.drop(columns=["Username"])
    # copy this column, in a new column called Username
    df["Username"] = s

    # Prefix with # (avoid double ##)
    s = s.apply(lambda x: "" if x == "" else (x if x.startswith("#") else "#" + x))

    df[email_col] = s
    df = df.rename(columns={email_col: "OrgDefinedId"})
    # put OrgDefinedId as the first column
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("OrgDefinedId")))
    cols.insert(1, cols.pop(cols.index("Username")))
    df = df[cols]

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert Email column to OrgDefinedId by stripping @uwo.ca and prefixing #."
    )
    parser.add_argument("--input_csv", 
                        default=r"C:\Users\maxiv\Documents\UWO\TA\TA7-W26\Search for Life in the Universe\iClicker_GradesExport_BrightspaceByD2L_04-16-26.csv",
                        type=Path, help="Path to input CSV file")
    parser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=None,
        help="Path to output CSV file (default: <input_stem>_OrgDefinedId.csv)",
    )
    args = parser.parse_args()

    in_path: Path = args.input_csv
    out_path: Path = args.output_csv or in_path.with_name(in_path.stem + "_OrgDefinedId.csv")

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)

    email_col = find_email_column(df.columns)
    if email_col is None:
        raise ValueError(f"Couldn't find an Email column. Columns are: {list(df.columns)}")

    df = transform_email_to_orgdefinedid(df, email_col)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")  # utf-8-sig is Excel-friendly

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
