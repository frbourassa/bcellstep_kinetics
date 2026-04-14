#=
Simplified plotting module for ML fits across multiple datasets.

Provides functions to plot empirical CDFs with overlaid model fits.
=#

using DataFrames: DataFrames as DF
using Plots: Plots as pl
using Measures: mm
using Printf: @sprintf
using Statistics: mean

module tracksAnalysis
    include("tracks.jl")
end
using .tracksAnalysis: prepcdf, concfromstr


"""
Plot a single dataset's empirical CDF with both model fits overlaid.

Arguments:
- tactiv: activation times for this dataset
- name: dataset name for title
- duration: length of imaging experiment
- cdfthresh: CDF threshold used for fitting
- kappa1, k2_1, t0_1: parameters for one-hit model
- kappa2, k2_2, t0_2: parameters for two-hit model
- conc: concentration for this dataset
- cdffct1, cdffct2: CDF functions for the two models
"""
function plot_single_dataset_cdf(tactiv, name, duration, cdfthresh,
                                  kappa1, k2_1, t0_1,
                                  kappa2, k2_2, t0_2,
                                  cdffct1, cdffct2)
    
    # Prepare empirical CDF
    tu, cumul, ilim1, ncells, tlim = prepcdf(tactiv, cdfthresh, duration)
    ilim0 = findfirst(tu .> 0.0)
    rms_i = (ilim1 > ilim0) ? ilim0 : 1
    conc_scale = 1.0  # pM
    conc = concfromstr(name, "pM", ' ') / conc_scale
    println("Concentration:", conc)
    
    # Scale CDF for plotting
    goodt = vcat([0], tu)
    ncumul = vcat([0], cumul .* ncells)
    
    # Create plot
    p = pl.plot(size=(600, 500), legend=:topleft, 
                left_margin=3mm, right_margin=3mm, bottom_margin=3mm)
    
    # Plot empirical CDF as step function
    pl.plot!(p, goodt, ncumul, linetype=:steppost, 
             color=:black, linewidth=2.5, label="Data", alpha=0.8)
    
    # Add markers at data points
    pl.scatter!(p, tu, cumul .* ncells, color=:black, 
                markersize=4, label=nothing, alpha=0.6)
    
    # Generate smooth curves for model fits over the fitting range
    trange = collect(LinRange(0, tlim + 0.1*tlim, 150))
    
    # One-hit model
    cdf1 = cdffct1(trange, conc, kappa1, k2_1; t0=t0_1)
    y1 = cdf1 .* ncells
    pl.plot!(p, trange, y1, label="1-hit model", 
             color=:blue, linewidth=2, linestyle=:solid)
    
    # Two-hit model
    cdf2 = cdffct2(trange, conc, kappa2, k2_2; t0=t0_2)
    y2 = cdf2 .* ncells
    pl.plot!(p, trange, y2, label="2-hit model", 
             color=:red, linewidth=2, linestyle=:dash)
    
    pl.xlabel!(p, "Time (s)", fontsize=12)
    pl.ylabel!(p, "Number of activated cells", fontsize=12)
    pl.title!(p, name, fontsize=13)
    
    return p
end


"""
Create a multi-panel plot comparing one-hit and two-hit models across all datasets.

Arguments:
- tactiv_alldsets: list of activation time dataframes
- nicenames: list of dataset names
- durations: list of experiment durations
- cdfthresh: CDF threshold used for fitting
- onehit_params: [kappa, k2, t0_1, t0_2, ...]
- twohit_params: [kappa, k2, t0_1, t0_2, ...]
- onehit_cdf, twohit_cdf: CDF functions
- onehit_rms, twohit_rms: RMS errors for model comparison
"""
function plot_ml_cdf_comparison(tactiv_alldsets, nicenames, durations, cdfthresh,
                                 onehit_params, twohit_params,
                                 onehit_cdf, twohit_cdf,
                                 onehit_rms, twohit_rms)
    
    n_dsets = length(tactiv_alldsets)
    conc_scale = 1.0  # pM

    # Determine layout (try to make roughly square)
    nrows = ceil(Int, sqrt(n_dsets))
    ncols = ceil(Int, n_dsets / nrows)
    
    # Create subplot layout
    p = pl.plot(layout=(nrows, ncols), size=(400*ncols, 350*nrows),
                left_margin=3mm, right_margin=3mm, bottom_margin=3mm)

    for i in 1:n_dsets
        tactiv = tactiv_alldsets[i]

        # Extract concentration
        conc = concfromstr(nicenames[i], "pM", ' ') / conc_scale
        
        # Prepare empirical CDF
        tu, cumul, ilim1, ncells, tlim = prepcdf(tactiv, cdfthresh, durations[i])
        
        # Scale CDF for plotting
        goodt = vcat([0], tu)
        ncumul = vcat([0], cumul .* ncells)
        
        # Get parameters for this dataset
        kappa_1, k2_1, t0_1 = onehit_params[1], onehit_params[2], onehit_params[2+i]
        kappa_2, k2_2, t0_2 = twohit_params[1], twohit_params[2], twohit_params[2+i]
        
        # Generate smooth curves
        trange = collect(LinRange(0, tlim + 0.1*tlim, 100))
        
        cdf1 = onehit_cdf(trange, conc, kappa_1, k2_1; t0=t0_1)
        y1 = cdf1 .* ncells
        
        cdf2 = twohit_cdf(trange, conc, kappa_2, k2_2; t0=t0_2)
        y2 = cdf2 .* ncells
        
        # Plot on subplot i
        pl.plot!(p[i], goodt, ncumul, linetype=:steppost,
                 color=:black, linewidth=2, label="Data", alpha=0.7, 
                 legend=:bottomleft, legendfontsize=8)
        
        pl.scatter!(p[i], tu, cumul .* ncells, color=:black,
                    markersize=3, label=nothing, alpha=0.5)
        
        pl.plot!(p[i], trange, y1, label="1-hit",
                 color=:blue, linewidth=1.8, linestyle=:solid)
        
        pl.plot!(p[i], trange, y2, label="2-hit",
                 color=:red, linewidth=1.8, linestyle=:dash)
        
        pl.xlabel!(p[i], "Time (s)", fontsize=10)
        pl.ylabel!(p[i], "Active cells", fontsize=10)
        pl.title!(p[i], nicenames[i], fontsize=11)
    end
    
    # Add overall title with RMS comparison
    title_text = @sprintf "ML Fits: %d dataset(s) | RMS: 1-hit=%.2e, 2-hit=%.2e" n_dsets onehit_rms twohit_rms
    pl.plot!(title=title_text, plot_title_fontsize=12)
    
    return p
end
