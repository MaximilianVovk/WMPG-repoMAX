import numpy as np

def read_prior_to_bounds(file_path=""):
    # Default bounds
    default_bounds = {
        "v_init": (np.nan, 500),
        "zenith_angle": (np.nan, 0.01),
        "m_init": (np.nan, np.nan),
        "rho": (np.log10(10), np.log10(4000)),  # log transformation applied later
        "sigma": (0.008 / 1e6, 0.03 / 1e6),
        "erosion_height_start": (np.nan, np.nan),
        "erosion_coeff": (np.log10(1 / 1e12), np.log10(2 / 1e6)),  # log transformation applied later
        "erosion_mass_index": (1, 3),
        "erosion_mass_min": (np.log10(5e-12), np.log10(1e-9)),  # log transformation applied later
        "erosion_mass_max": (np.log10(1e-10), np.log10(1e-7)),  # log transformation applied later
    }

    default_flags = {
        "v_init": ["norm"],
        "zenith_angle": ["norm"],
        "rho": ["log"],
        "erosion_coeff": ["log"],
        "erosion_mass_min": ["log"],
        "erosion_mass_max": ["log"]
        }

    # Default values if no file path is provided
    if file_path == "":
        # delete from the default_bounds the zenith_angle
        default_bounds.pop("zenith_angle")
        bounds = [default_bounds[key] for key in default_bounds]
        flags_dict = {key: default_flags.get(key, []) for key in default_bounds}
        fixed_values = {
            "zenith_angle": np.nan,
        }

    else:
        bounds = []
        flags_dict = {}
        fixed_values = {}

        def safe_eval(value):
            try:    
                return eval(value, {"__builtins__": {"np": np}}, {})
            except Exception:
                return value  # Return as string if evaluation fails
        
        # Read .prior file, ignoring comment lines
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split('#')[0].strip().split(',')  # Remove inline comments
                name = parts[0].strip()
                # Handle fixed values
                if "fix" in line:
                    fixed_values[name] = safe_eval(parts[1].strip() if len(parts) > 1 else "nan")
                    continue
                min_val = parts[1].strip() if len(parts) > 1 else "nan"
                max_val = parts[2].strip() if len(parts) > 2 else "nan"
                flags = [flag.strip() for flag in parts[3:]] if len(parts) > 3 else []
                
                # Handle NaN values and default replacement
                min_val = safe_eval(min_val) if min_val.lower() != "nan" else default_bounds.get(name, (np.nan, np.nan))[0]
                max_val = safe_eval(max_val) if max_val.lower() != "nan" else default_bounds.get(name, (np.nan, np.nan))[1]

                # check if min_val is greater than max_val and if it is, swap the values
                if min_val > max_val:
                    min_val, max_val = max_val, min_val
                if min_val == max_val and min_val != np.nan:
                    fixed_values[name] = min_val
                    continue

                # check when at least one of the values is nan if the name is in the default_flags and if it is, add the flags to the flags list if not already there
                if np.isnan(min_val) or np.isnan(max_val):
                    if default_flags.get(name, []) and not flags:
                        flags = default_flags.get(name, []) + flags

                # Store flags
                flags_dict[name] = flags
                    
                # Apply log10 transformation if needed
                if "log" in flags:
                    # check if any values is 0 and if it is, replace it with the default value
                    if min_val == 0:
                        min_val = 1/1e12
                    # Apply log10 transformation
                    min_val, max_val = np.log10(min_val), np.log10(max_val)
                
                bounds.append((min_val, max_val))
    
    # check if the bounds the len(bounds) + len(fixed_values) =>10
    if len(bounds) + len(fixed_values) < 10:
        raise ValueError("The number of bounds and fixed values should 10 or above")

    return bounds, flags_dict, fixed_values

# Example usage
file_path = r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\file.prior"  # Update with your actual file path
bounds, flags, fixed = read_prior_to_bounds(file_path)
print("Bounds:", bounds)
print("Flags:", flags)
print("Fixed Values:", fixed)
