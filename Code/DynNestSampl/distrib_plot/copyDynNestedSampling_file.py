from pathlib import Path
import shutil


def collect_event_plots(input_folder, output_folder):
    """
    Recursively find and copy:

        *_distrib_plot.png
        *_best_fit_LumLag_plot.png

    All images are copied directly into one output folder.
    """

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    copied_files = 0

    # Find both types of plot
    plot_patterns = [
        "*_distrib_plot.png",
        "*_best_fit_LumLag_plot.png",
    ]

    for pattern in plot_patterns:
        for source_path in input_folder.rglob(pattern):

            # Only accept best-fit plots located inside fit_plots folders
            if (
                source_path.name.endswith("_best_fit_LumLag_plot.png")
                and "fit_plots" not in source_path.parts
            ):
                continue

            destination_path = output_folder / source_path.name

            # Avoid overwriting a file with the same name
            if destination_path.exists():
                parent_name = source_path.parent.name

                if parent_name == "fit_plots":
                    parent_name = source_path.parent.parent.name

                destination_path = (
                    output_folder
                    / f"{parent_name}_{source_path.name}"
                )

            # Add a number if the alternative name also exists
            counter = 2
            original_destination = destination_path

            while destination_path.exists():
                destination_path = (
                    output_folder
                    / f"{original_destination.stem}_{counter}"
                    f"{original_destination.suffix}"
                )
                counter += 1

            shutil.copy2(source_path, destination_path)

            print(f"Copied: {source_path}")
            print(f"     to: {destination_path}\n")

            copied_files += 1

    print("----------------------------------------")
    print(f"Copied {copied_files} images.")
    print(f"Output folder: {output_folder}")


# ==============================================================
# CHANGE THESE PATHS
# ==============================================================

input_folder = r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\Sporadic_final\Stony"
output_folder = r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\test_output"

collect_event_plots(input_folder, output_folder)