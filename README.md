# WMPG Max'repository
Western Meteor Physics Group (WMPG) Maximilian Vovk's personal repository. This repository contains a collection of tools and scripts designed for the analysis and reduction of meteor data. The primary focus is on comparing manual and automated meteor reductions, implementing principal component analysis (PCA) for event classification, and Dynamic Nested Sampling for uncertanty estimation of meteor physical properties.

## Repository Structure

### `Code` Folder
The `Code` folder contains the main scripts used for data processing and analysis.

#### DynNestSampl
- **Description**: Implements Principal Dynamuc Nested Sampling to define uncertanty estimate for the meteor. It reads automatically EMCCD and CAMO .pickle data but it can also work with metsim jons data if path and file name are i the input directory.
- **Usage**: `python "WMPG-repoMAX\Code\DynNestSampl\DynNestSapl_metsim.py" "C:\Users\maxiv\Documents\INPUT-FOLDER" --output_dir "C:\Users\maxiv\Desktop\OUTPUT-FOLDER" --prior "C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\stony_meteoroid.prior"`
- **Dependencies**: WMPG, Dynesty
- 
#### EMCCD_manual-auto_errPlot
- **Description**: This script generates plots to compare the pixel positions of meteors as determined by manual reductions against those obtained from automated methods.
- **Usage**: `python -m Plots_LightCurves`
- **Dependencies**: WMPG

#### Faint_meteor_PhysUnc_RMSD-PCA
- **Description**: Implements Principal Component Analysis (PCA) to identify the closest reduced events to a subset of simulated meteors, using the WMPG's GenerateSimulations.py for simulation.
- **Usage**: `python -m Faint_meteor_PhysUncert.py /home/mvovk/PCA/PER_1000_1milion_manual /home/mvovk/PCA/ 1000`
- **Dependencies**: WMPG

#### SSA
- **Description**: Satellite calibration for EMCCD and LCAM cameras accuracy checks.
- **Usage**: `python Plot_AzAltMag_errCamera.py "C:\Users\maxiv\Documents\INPUT-FOLDER"`
- **Dependencies**: RMS

#### `Utils` Folder
This folder contains utility scripts to assist with the preparation of data for METAL MIFIT and SkyFit2 reductions and conversion between pickle to json file and from.met to .ecsv file for trajectory solution.

### `Thesis` Folder
The `Thesis` folder contains the comprehensive 2 pdf.
