# WMPG Max'repository
Western Meteor Physics Group (WMPG) Maximilian Vovk's personal repository.

- **Meteor Light and Atomic Excitation**: Meteor light is mainly produced by atoms in excited states, often neutrals like iron, sodium, and oxygen, released from the meteor body. These atoms get excited and emit light upon colliding with atmospheric molecules. This process generally does not ionize the atoms except at high speeds.

- **Ionization at High Speeds**: At greater velocities, ionization contributes more to meteor light, exemplified by the triple seven line of oxygen, which requires more energy to become visible.

- **Emission Lines**: Meteor light features individual emission lines from excited states of meteoroid material (such as calcium, magnesium, silicon, iron), different from a blackbody spectrum. The light emission occurs when these excited electrons return to a lower energy state.

This repository contains a collection of tools and scripts designed for the analysis and reduction of meteor data. The primary focus is on comparing manual and automated meteor reductions, implementing principal component analysis (PCA) for event classification, and preparing data for advanced reduction techniques.

## Repository Structure

### `code` Folder
The `code` folder contains the main scripts used for data processing and analysis.

#### ERROR_PLOT
- **Description**: This script generates plots to compare the pixel positions of meteors as determined by manual reductions against those obtained from automated methods.
- **Usage**: `python -m Plots_LightCurves`
- **Dependencies**: WMPG

#### PCA
- **Description**: Implements Principal Component Analysis (PCA) to identify the closest reduced events to a subset of simulated meteors, using the WMPG's GenerateSimulations.py for simulation.
- **Usage**: `python -m wmpl.MetSim.ML.GenerateSimulations_MAX "C:\Users\maxiv\Documents\UWO\Western Meteor Physics Group\Conferences\20230618 - ACM\Code_use\Simulations_PER" ErosionSimParametersEMCCD_PER 100`
`python -m EMCCD_PCA_Shower_PhysProp /home/mvovk/PCA/PER_1000_1milion_manual PER /home/mvovk/PCA/ 1000`
`python -m Plots_LightCurves`
- **Dependencies**: WMPG

#### `Utils` Folder
This folder contains utility scripts to assist with the preparation of data for METAL MIFIT and SkyFit2 reductions.

### `Thesis` Folder
The `Thesis` folder contains the comprehensive 2 pdf.
