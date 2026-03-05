# Population kinetics analysis of B cell activation microscopy tracks

Code to analyze B cell calcium imaging microscopy tracks in terms of the cumulative distribution of active cells, to produce Figure S2 of the manuscript 

>> 

See the Methods section for a technical description of the mathematical analysis and maximum likelihood fitting of cumulative distribution functions corresponding to the different models. 

## Code structure
The Jupyter notebook `finalPlotsTracks.ipynb` runs the overall analysis and produces the figure panels using the modules in `utils/`. 
The module `utils/tracks.jl` contains data processing, analysis, and distribution fitting functions, while `utils/tracksPlots.jl` contains plotting functions. 
The Jupyter notebook `linear_quadratic_cartoons.ipynb` makes the simplified cumulative distribution plots for the 1- and 2-step models depicted in panel C. 


## Requirements

This code was run using Julia version 1.12.2 and requires installation of the following packages:
- Plots
- ColorSchemes
- DataFrames
- CSV
- Distributions
- Optim
- Measures
- Printf
- Statistics
- LinearAlgebra