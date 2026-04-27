import numpy as np
import matplotlib.pyplot as plt


def read_mem_flux_file(filepath):
    """
    Reads a MEM cube_avg.txt-style flux file.

    Returns:
        speeds: velocity bins in km/s
        data: dictionary with flux arrays for each direction
    """

    directions = [
        "+x ram", "-x wake", "+y port", "-y starboard",
        "+z zenith", "-z nadir", "Earth", "Sun", "anti-Sun",
        "rot (x)", "rot (y)", "rot (z)"
    ]

    rows = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.split()

            # Data rows have 13 numeric columns:
            # speed + 12 flux columns
            if len(parts) == 13:
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    pass

    arr = np.array(rows)

    if arr.shape[1] != 13:
        raise ValueError("Could not read the file correctly.")

    speeds = arr[:, 0]

    data = {}
    for i, direction in enumerate(directions):
        data[direction] = arr[:, i + 1]

    return speeds, data


# -------------------------------------------------------------------
# Input files
# -------------------------------------------------------------------

file1 = "cube_avg_1.txt"
file2 = "cube_avg_2.txt"

speeds1, data1 = read_mem_flux_file(file1)
speeds2, data2 = read_mem_flux_file(file2)

# Check that speed bins match
if not np.allclose(speeds1, speeds2):
    raise ValueError("The speed bins in the two files do not match.")

speeds = speeds1

# -------------------------------------------------------------------
# Directions to plot
# -------------------------------------------------------------------

directions_to_plot = ["+x ram", "-x wake", "+z zenith"]

summed_flux = {}

for direction in directions_to_plot:
    summed_flux[direction] = data1[direction] + data2[direction]

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

plt.figure(figsize=(10, 7))

for direction in directions_to_plot:
    flux = summed_flux[direction]

    # Total flux summed over all speed bins
    total_flux = np.sum(flux)

    # Avoid plotting zero values on log scale
    mask = flux > 0

    plt.plot(
        speeds[mask],
        flux[mask],
        marker="o",
        linewidth=2.0,
        markersize=4,
        label=f"{direction}, total = {total_flux:.6f} #/m²/yr"
    )

plt.yscale("log")
plt.xlabel("Speed (km/s)")
plt.ylabel("Summed flux (#/m$^2$/yr)")
plt.title("Summed MEM flux for selected directions")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

# High-resolution output
plt.savefig("summed_flux_selected_directions_highres.png", dpi=600, bbox_inches="tight")

plt.show()