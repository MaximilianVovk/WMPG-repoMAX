import argparse
import pickle
from pathlib import Path
from typing import Dict, Any


def rename_station_ids(data: Any, mapping: Dict[str, str]) -> None:
    """
    Recursively walk the loaded pickle structure and rename any 'station_id'
    found either:
      - as a dict key, or
      - as an attribute on an object (e.g. ObservedPoints.station_id)
    Modifies 'data' in place.
    """
    print("Renaming station IDs according to mapping:", mapping)

    def _walk(obj: Any, depth: int = 0) -> None:
        indent = "  " * depth
        print(f"{indent}Visiting object of type: {type(obj)}")

        # 1) If it's a dict, check for a 'station_id' key and then recurse into values
        if isinstance(obj, dict):
            print(f"{indent}Dict keys: {list(obj.keys())}")
            if "station_id" in obj:
                old_val = obj["station_id"]
                old_key = str(old_val)
                if old_key in mapping:
                    new_val = mapping[old_key]
                    print(f"{indent}Renaming dict station_id {old_val!r} -> {new_val!r}")
                    obj["station_id"] = new_val
                else:
                    print(f"{indent}No mapping for dict station_id {old_val!r}, leaving as is.")

            for v in obj.values():
                _walk(v, depth + 1)

        # 2) If it's a list/tuple/set, recurse into each element
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                _walk(item, depth + 1)

        # 3) If it's a custom object with attributes, check for a 'station_id' attribute,
        #    then recurse into its attributes
        elif hasattr(obj, "__dict__"):
            attr_dict = vars(obj)
            print(f"{indent}Object has __dict__ with keys: {list(attr_dict.keys())}")

            # Handle station_id as an *attribute* on the object
            if "station_id" in attr_dict:
                old_val = attr_dict["station_id"]
                old_key = str(old_val)
                if old_key in mapping:
                    new_val = mapping[old_key]
                    print(f"{indent}Renaming attr station_id {old_val!r} -> {new_val!r}")
                    attr_dict["station_id"] = new_val
                else:
                    print(f"{indent}No mapping for attr station_id {old_val!r}, leaving as is.")

            # Now recurse into all attributes
            for attr_name, attr_val in attr_dict.items():
                print(f"{indent}  Walking attr '{attr_name}' ({type(attr_val)})")
                _walk(attr_val, depth + 1)

        # 4) Other types: nothing to do
        else:
            return

    _walk(data)



def parse_mapping(mapping_list):
    """
    Parse mappings of the form OLD:NEW from the CLI
    into a dict {OLD: NEW}.
    """
    mapping = {}
    for item in mapping_list:
        print(f"Parsing mapping: {item}")
        # Allow OLD:NEW or OLD=NEW
        if ":" in item:
            old, new = item.split(":", 1)
        elif "=" in item:
            old, new = item.split("=", 1)
        else:
            raise ValueError(f"Invalid mapping '{item}'. Use OLD:NEW or OLD=NEW.")
        mapping[old.strip()] = new.strip()
    return mapping


def main():
    # python rename_stations.py "c:\Users\maxiv\Documents\UWO\Papers\0.3)Phaethon\Base\GEM-Nick\20211214_090300_trajectory_met.pickle" --rename 1=01G 2=02G 4=01K 3=02K --output "c:\Users\maxiv\Documents\UWO\Papers\0.3)Phaethon\Base\GEM-Nick\20211214_090300_trajectory_newName.pickle"
    #  python rename_stations.py "c:\Users\maxiv\Documents\UWO\Papers\0.3)Phaethon\Base\GEM-Nick\20211214_104553_trajectory_met.pickle" --rename 4=01K 2=02K --output "c:\Users\maxiv\Documents\UWO\Papers\0.3)Phaethon\Base\GEM-Nick\20211214_104553_trajectory_newName.pickle"
    parser = argparse.ArgumentParser(
        description="Rename station IDs inside a SkyFit2 pickle file."
    )
    parser.add_argument(
        "input_pickle",
        help="Path to the input .pickle file",
    )
    parser.add_argument(
        "--rename",
        "-r",
        nargs="+",
        required=True,
        help="Station renames in the form OLD:NEW (e.g. 01G:W-GEO1 02G:W-GEO2)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Path for the output .pickle file. "
            "If not given, the input file will be overwritten."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input_pickle)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input pickle not found: {input_path}")

    # Load pickle
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    # Build mapping dict
    mapping = parse_mapping(args.rename)

    # Rename all station_ids wherever they appear
    rename_station_ids(data, mapping)

    # Determine output path
    if args.output is None:
        output_path = input_path  # overwrite
    else:
        output_path = Path(args.output)

    # Save pickle
    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved updated pickle to: {output_path}")


if __name__ == "__main__":
    main()
