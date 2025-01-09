import re
from astropy.table import Table, Column
import sys

def parse_line(line):
    """
    Parse a single line from the .met file. This function attempts to identify the record type and extract key-value pairs.
    """
    # Strip comments and whitespace
    line = line.strip()
    if not line or line.startswith('#'):
        return None, {}

    # Identify record type by the first token
    # Common record types in the data snippet: video, scale, exact, frame, mask, point, mark
    tokens = line.split(';')
    if len(tokens) > 1:
        # There are semicolon-delimited sections
        record_type = tokens[0].strip()
        remainder = ';'.join(tokens[1:]).strip()
    else:
        # If no semicolon found, try space delimiting
        parts = line.split()
        if len(parts) > 0:
            record_type = parts[0].strip()
            remainder = ' '.join(parts[1:]).strip()
        else:
            return None, {}

    # record_type might be something like 'video', 'scale', 'exact', 'frame', 'mask', 'point', 'mark'
    # The remainder often includes key-value pairs like: site 1 text 'Tavistock' path 'ev_20230811_082649A_01T.vid'
    # We'll try to parse these pairs. Keys are often word chars, values might be quoted or numeric.
    # We'll attempt a regex-based approach.

    # A regex that matches patterns like: key value pairs where value may be quoted
    # Pattern tries to capture something like: key value pairs including single-quoted strings
    pattern = re.compile(r"(\S+)\s+'([^']*)'|(\S+)\s+(\S+)")
    # The pattern above tries two alternatives:
    # (\S+)\s+'([^']*)' matches key 'value in quotes'
    # (\S+)\s+(\S+) matches key value without quotes

    data = {}
    # We'll need a more careful approach. We'll tokenize by spaces and try to handle quoted strings.
    # Let's do a custom parser:
    # We know that keys and values alternate. Values might be quoted with single quotes.
    # We'll split carefully.

    # A simpler approach: tokenize by spaces, but recombine if we find a quoted value
    parts = re.split(r"\s+", remainder)
    i = 0
    while i < len(parts):
        key = parts[i]
        val = None
        i += 1
        if i >= len(parts):
            # no value for key
            data[key] = None
            break
        # If next token starts with a quote, we combine until end quote
        if parts[i].startswith("'"):
            # Extract a quoted value
            val_tokens = [parts[i]]
            i += 1
            while i < len(parts) and not parts[i].endswith("'"):
                val_tokens.append(parts[i])
                i += 1
            if i < len(parts):
                val_tokens.append(parts[i])
                i += 1
            val_str = " ".join(val_tokens)
            # Strip leading and trailing quotes
            val_str = val_str.strip("'")
            val = val_str
        else:
            # Non-quoted value
            val = parts[i]
            i += 1

        data[key] = val

    return record_type, data

def main(infile, outfile):
    # We will gather all records in a single table with columns for keys that appear.
    # Because .met has many different record types and keys, we can dynamically build the set of keys.

    all_records = []
    all_keys = set()
    with open(infile, 'r') as f:
        for line in f:
            if line.strip() and not line.strip().startswith('#'):
                rtype, rec = parse_line(line)
                if rtype:
                    # Add record_type to rec
                    rec['record_type'] = rtype
                    all_records.append(rec)
                    all_keys.update(rec.keys())

    # Convert all_records to a table
    # ECSV requires us to have columns. We'll create a column for each key in all_keys.
    # Some keys may not appear in all rows, set them to None if missing.

    # Create table
    # Sorting keys so that record_type is first
    keys_list = ['record_type'] + sorted([k for k in all_keys if k != 'record_type'])

    # Prepare rows
    rows = []
    for r in all_records:
        row = []
        for k in keys_list:
            val = r.get(k, None)
            # try convert numeric
            if val is not None:
                # attempt to convert to float or int if numeric
                if re.match(r'^[0-9\.\-]+$', val):
                    # numeric
                    try:
                        if '.' in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except:
                        pass
            row.append(val)
        rows.append(row)

    # Create the Astropy table
    tab = Table(rows=rows, names=keys_list)
    # Write as ECSV
    tab.write(outfile, format='ascii.ecsv', overwrite=True)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python met_to_ecsv.py input.met output.ecsv")
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2]
    main(infile, outfile)
