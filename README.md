# Population kinetics analysis of B cell activation microscopy tracks

Code to analyze B cell calcium imaging microscopy tracks in terms of the cumulative distribution of active cells, to produce Figure S2 of the manuscript 

> Julianne B. Riggs, Alexander J. Ritter, François X. P. Bourassa*, Alexander R. Meyer*, Erika Kay-Tsumagari, Wei-Yun Wholey, Jeremy Libang, James Mueller, Wen Lu, Ned S. Wingreen, Wei Cheng**, Julie Zikherman**. "Virus-like antigen display delivers a stand–alone danger signal through the BCR that circumvents tolerance", _bioRxiv_, April 24, 2026. [https://doi.org/10.64898/2026.04.23.720482](https://doi.org/10.64898/2026.04.23.720482)

See the Methods section of the manuscript for a summary of the data processing, models and maximum likelihood estimation procedure. See the `models.md` file for a more detailed mathematical description of the cell activation first-passage time distributions and their estimation by maximum likelihood. 

## Code structure

- The module `analysis/tracks.jl` contains data processing, analysis, and distribution fitting functions.

The Jupyter notebook `tracksObsFits.ipynb` fits a one-step model (i.e. exponential distribution) on each dataset to extract an effective activation rate $k_\mathrm{obs}$. It relies on models, log-likelihood functions, and optimization functions in `analysis/kObsML.jl` and on plotting scripts in `analysis/tracksPlots.jl`. 

- The Jupyter notebook `hitModelsFits.ipynb` runs the overall analysis using the complete hit models with a cell activation step, fitted across datasets, and produces the figure panels. It relies on model curves, log-likelihood functions and parameter estimation procedures in `analysis/hitModelsML.jl`, and on plotting scripts in `analysis/tracksPlots.jl`. 

- The Jupyter notebook `onehit_twohits_cartoons.ipynb` makes illustrative cumulative distribution plots for the 1-, 2-, and 3-hit models, as shown in panels D and G. 


## Requirements

This code was run using Julia version 1.12.2 on macOS 26.4 and requires adding the following modules:
- Plots (GR backend, tested with v1.41.2)
- DataFrames (tested with v1.8.1)
- CSV (tested with v0.10.15)
- Optim (tested with v1.13.3)
- Measures (tested with v0.3.3)
- Printf (included in the Julia Standard Library)
- Statistics (included in the Julia Standard Library)
- IJulia (for Jupyter notebooks, tested with v1.33.0)

It also requires Jupyter to run the notebooks, which can be installed through IJulia if not already available on your system (see installation instructions). The code was run on Jupyter notebook version 6.5.7. 


## Installation instructions

Install the Julia language following the instructions on the official website: [Installing Julia](https://julialang.org/install/)

To install Jupyter Notebooks, if not already available on your system, follow the instructions for the IJulia module – [Running IJulia](https://julialang.github.io/IJulia.jl/stable/manual/running/) – or install Jupyter directly – [Installing Jupyter](https://jupyter.org/install). 

The installation of Julia, Jupyter, and the required modules listed above typically takes 30 minutes or less on a personal computer. 


## Reproduction instructions

To reproduce the figure panels, clone or download the repository and run the Jupyter notebooks listed above. The figures should appear in the notebooks and be saved as pdfs in the `figures/` folder. 

Note that once Jupyter and the Julia requirements are installed, the notebooks can be run by launching Jupyter (e.g. with the terminal command `jupyter notebook`) or within an integrated development environment (IDE) supporting notebooks, such as [VSCode](https://code.visualstudio.com/docs/datascience/jupyter-notebooks). 


## Demo

The expected text output when running the cell calling `fit_ml_all_datasets()` in `hitModelsFits.ipynb` is reproduced below. Some figures should also be displayed and saved. Run time: 12.8 s. Running previous cells in the notebook takes ~10 s (loading Julia modules takes some time). 

> fit_ml_all_datasets();
```
MAXIMUM LIKELIHOOD FITTING - JR064 DATASETS

Loading and preprocessing datasets...
Filtering removed 49 tracks out of 122
Filtering removed 46 tracks out of 87
Filtering removed 35 tracks out of 85
Filtering removed 21 tracks out of 80
Filtering removed 25 tracks out of 55
Filtering removed 43 tracks out of 78
Filtering removed 36 tracks out of 96

Loaded 7 datasets:
  [1] 10pM
  [2] 10pM 2
  [3] 1pM
  [4] 0.1pM 2
  [5] 0.1pM
  [6] 0.01pM 2
  [7] 0.01pM
Fitting ONE-HIT model (overall k2, one kc and one t0 per dataset)...

One-hit model results:
  k2        = 0.0562 s^-1
  t0 per dataset:
    Dataset 1 (10pM):  kc = 11.5588 s^-1
    Dataset 1 (10pM): t0 = -3.9821 s
    Dataset 2 (10pM 2):  kc = 200.0000 s^-1
    Dataset 2 (10pM 2): t0 = 1.2806 s
    Dataset 3 (1pM):  kc = 0.0383 s^-1
    Dataset 3 (1pM): t0 = -1.1790 s
    Dataset 4 (0.1pM 2):  kc = 0.0068 s^-1
    Dataset 4 (0.1pM 2): t0 = -9.2385 s
    Dataset 5 (0.1pM):  kc = 0.0116 s^-1
    Dataset 5 (0.1pM): t0 = -0.7886 s
    Dataset 6 (0.01pM 2):  kc = 0.0058 s^-1
    Dataset 6 (0.01pM 2): t0 = -14.2110 s
    Dataset 7 (0.01pM):  kc = 0.0043 s^-1
    Dataset 7 (0.01pM): t0 = 13.8175 s
Fitting TWO-HIT model (overall k2, one kc and one t0 per dataset)...

Two-hit model results:
  k2        = 0.0580 s^-1
  t0 per dataset:
    Dataset 1 (10pM):  kc = 13.6896 s^-1
    Dataset 1 (10pM): t0 = -3.9256 s
    Dataset 2 (10pM 2):  kc = 200.0000 s^-1
    Dataset 2 (10pM 2): t0 = 1.2714 s
    Dataset 3 (1pM):  kc = 0.0625 s^-1
    Dataset 3 (1pM): t0 = -8.1935 s
    Dataset 4 (0.1pM 2):  kc = 0.0138 s^-1
    Dataset 4 (0.1pM 2): t0 = -20.0000 s
    Dataset 5 (0.1pM):  kc = 0.0200 s^-1
    Dataset 5 (0.1pM): t0 = -20.0000 s
    Dataset 6 (0.01pM 2):  kc = 0.0124 s^-1
    Dataset 6 (0.01pM 2): t0 = -20.0000 s
    Dataset 7 (0.01pM):  kc = 0.0085 s^-1
    Dataset 7 (0.01pM): t0 = -16.8898 s
Fitting THREE-HIT model (overall k2, one kc and one t0 per dataset)...

Three-hit model results:
  k2        = 0.0605 s^-1
  t0 per dataset:
    Dataset 1 (10pM):  kc = 15.4569 s^-1
    Dataset 1 (10pM): t0 = -3.8176 s
    Dataset 2 (10pM 2):  kc = 200.0000 s^-1
    Dataset 2 (10pM 2): t0 = 1.2636 s
    Dataset 3 (1pM):  kc = 0.0780 s^-1
    Dataset 3 (1pM): t0 = -14.4546 s
    Dataset 4 (0.1pM 2):  kc = 0.0215 s^-1
    Dataset 4 (0.1pM 2): t0 = -20.0000 s
    Dataset 5 (0.1pM):  kc = 0.0310 s^-1
    Dataset 5 (0.1pM): t0 = -20.0000 s
    Dataset 6 (0.01pM 2):  kc = 0.0192 s^-1
    Dataset 6 (0.01pM 2): t0 = -20.0000 s
    Dataset 7 (0.01pM):  kc = 0.0132 s^-1
    Dataset 7 (0.01pM): t0 = -20.0000 s
RMS ERROR COMPARISON
RMS per dataset:

    Dataset 1 (10pM), one-hit = 9.292696e-02
    Dataset 1 (10pM), two-hit = 9.985267e-02
    Dataset 1 (10pM), three-hit = 1.095229e-01

    Dataset 2 (10pM 2), one-hit = 1.922186e-01
    Dataset 2 (10pM 2), two-hit = 1.856315e-01
    Dataset 2 (10pM 2), three-hit = 1.763906e-01

    Dataset 3 (1pM), one-hit = 6.686562e-02
    Dataset 3 (1pM), two-hit = 7.096059e-02
    Dataset 3 (1pM), three-hit = 7.176354e-02

    Dataset 4 (0.1pM 2), one-hit = 4.092495e-02
    Dataset 4 (0.1pM 2), two-hit = 7.672905e-02
    Dataset 4 (0.1pM 2), three-hit = 1.103855e-01

    Dataset 5 (0.1pM), one-hit = 6.847711e-02
    Dataset 5 (0.1pM), two-hit = 8.156146e-02
    Dataset 5 (0.1pM), three-hit = 1.122771e-01

    Dataset 6 (0.01pM 2), one-hit = 5.370990e-02
    Dataset 6 (0.01pM 2), two-hit = 4.068422e-02
    Dataset 6 (0.01pM 2), three-hit = 6.692768e-02

    Dataset 7 (0.01pM), one-hit = 4.157583e-02
    Dataset 7 (0.01pM), two-hit = 6.021499e-02
    Dataset 7 (0.01pM), three-hit = 8.370835e-02
Generating plots...
  Saved: figures/final_tracks_panels/ml_cdf_all_datasets_comparison.pdf
``` 

## Authors
François X. P. Bourassa, 2026, and all authors of the manuscript. 

## License
CC-BY-SA-4.0

## Zenodo badge
[![DOI](https://zenodo.org/badge/1208515872.svg)](https://doi.org/10.5281/zenodo.19657273)