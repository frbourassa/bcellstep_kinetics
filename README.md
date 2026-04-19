# Population kinetics analysis of B cell activation microscopy tracks

Code to analyze B cell calcium imaging microscopy tracks in terms of the cumulative distribution of active cells, to produce Figure S2 of the manuscript 

> Julianne B. Riggs, Alexander J. Ritter, François X. P. Bourassa*, Alexander R. Meyer*, Erika Kay-Tsumagari, Wei-Yun Wholey, Jeremy Libang, James Mueller, Wen Lu, Ned S. Wingreen, Wei Cheng**, Julie Zikherman**. "Virus-like antigen display delivers a stand–alone danger signal through the BCR that circumvents tolerance", [bioRxiv], 2026. 

See the Methods section of the manuscript for a summary of the data processing, models and maximum likelihood estimation procedure. See the `models.md` file for a more detailed mathematical description of the cell activation first-passage time distributions and their estimation by maximum likelihood. 

## Code structure

- The module `analysis/tracks.jl` contains data processing, analysis, and distribution fitting functions.
The Jupyter notebook `tracksObsFits.ipynb` fits a one-step model (i.e. exponential distribution) on each dataset to extract an effective activation rate $k_\mathrm{obs}$. It relies on models, log-likelihood functions, and optimization functions in `analysis/kObsML.jl` and on plotting scripts in `analysis/tracksPlots.jl`. 
- The Jupyter notebook `hitModelsFits.ipynb` runs the overall analysis using the complete hit models with a cell activation step, fitted across datasets, and produces the figure panels. It relies on model curves, log-likelihood functions and parameter estimation procedures in `analysis/hitModelsML.jl`, and on plotting scripts in `analysis/tracksPlots.jl`. 
- The Jupyter notebook `onehit_twohits_cartoons.ipynb` makes illustrative cumulative distribution plots for the 1-, 2-, and 3-hit models, as shown in panels D and G. 


## Requirements

This code was run using Julia version 1.12.2 and requires installation of the following packages:
- Plots
- DataFrames
- CSV
- Optim
- Measures
- Printf
- Statistics
- LinearAlgebra


## Authors
François X. P. Bourassa, 2026