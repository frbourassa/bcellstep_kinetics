#= 
Fit CDF activation models using Maximum Likelihood Estimation
across multiple datasets in the JR064 experiment.

This script:
1. Loads all datasets in JR064_tracks
2. Preprocesses activation times
3. Fits one-hit and two-hit models across all datasets simultaneously
4. Computes RMS errors for each model
5. Generates comparison plots

François X. P. Bourassa, 2025
=#

using DataFrames: DataFrames as DF
using Plots: Plots as pl
using Printf: @sprintf
using Statistics: mean

# Load the analysis module
module tracksAnalysis
    include("analysis/tracks.jl")
end
using .tracksAnalysis: prepalldsets, fitmodel, onehit_loglhood_total, 
    onehit_cdf, twohit_loglhood_total, twohit_cdf, compute_rms_all_dsets, 
    prepcdf, concfromstr, cleannames

# Load the new plotting module
module plotsML
    include("analysis/tracksPlots.jl")
end
using .plotsML: plot_ml_cdf_comparison, plot_single_dataset_cdf


### MAIN ANALYSIS FUNCTION ###

function fit_ml_all_datasets(; 
    datadir="data/JR064_tracks",
    plotdir="figures/ml_fits",
    fracabove=0.6,
    cdfthresh=0.5)
    
    # Create output directory if it doesn't exist
    if !isdir(plotdir)
        mkpath(plotdir)
    end
    
    # Get all CSV files from the data directory
    fnames = vec(collect(f for f in readdir(datadir) if endswith(f, ".csv")))
    
    if isempty(fnames)
        error("No CSV files found in $datadir")
    end
    
    # Filter to keep only the approved list of datasets
    # (excluding low cell count or duplicate datasets)
    approved_files = [
        "pHELD-AF594_10pM_2_z2.csv", 
        "pHELD-AF594_10pM_z2.csv", 
        "pHELD-AF594_1pM_2.csv", 
        #"pHELD-AF594_1pM.csv",   # low cell count, dropped dataset
        "pHELD-AF594_0.1pM_2.csv", 
        "pHELD-AF594_0.1pM.csv",
        "pHELD-AF594_0.01pM_2.csv", 
        "pHELD-AF594_0.01pM.csv"
    ]
    fnames = collect(fp for fp in fnames if fp in approved_files)
    fpaths = collect(joinpath(datadir, f) for f in fnames)
    
    if isempty(fpaths)
        error("No approved CSV files found in $datadir")
    end
    
    println("\n" * "="^70)
    println("MAXIMUM LIKELIHOOD FITTING - JR064 DATASETS")
    println("="^70)
    println("\nLoading and preprocessing datasets...")
    
    # Step 1: Preprocess all datasets
    tactiv_alldsets, durations, nicenames = prepalldsets(fpaths; 
        fracabove=fracabove, disp=false)
    
    n_dsets = length(tactiv_alldsets)
    println("\nLoaded $n_dsets datasets:")
    for (i, name) in enumerate(nicenames)
        println("  [$i] $name")
    end
    
    # Step 2: Fit the one-hit model
    println("\n" * "-"^70)
    println("Fitting ONE-HIT model (kappa, k2, one t0 per dataset)...")
    println("-"^70)
    onehit_result = fitmodel(tactiv_alldsets, nicenames, durations, cdfthresh, 
                             onehit_loglhood_total, onehit_cdf)
    onehit_params = onehit_result[1:end-1]
    onehit_ks = onehit_result[1:2]
    onehit_t0s = onehit_result[3:end-1]
    onehit_rms = onehit_result[end]
    
    println("\nOne-hit model results:")
    println(@sprintf "  κc_0 (kappa * 1 pM) = %.4f s^-1" onehit_ks[1])
    println(@sprintf "  k2        = %.4f s^-1" onehit_ks[2])
    println("  t0 per dataset:")
    for (i, t0) in enumerate(onehit_t0s)
        println(@sprintf "    Dataset %d (%s): t0 = %.4f s" i nicenames[i] t0)
    end
    
    # Step 3: Fit the two-hit model
    println("\n" * "-"^70)
    println("Fitting TWO-HIT model (kappa, k2, one t0 per dataset)...")
    println("-"^70)
    twohit_result = fitmodel(tactiv_alldsets, nicenames, durations, cdfthresh,
                             twohit_loglhood_total, twohit_cdf)
    twohit_params = twohit_result[1:end-1]
    twohit_ks = twohit_result[1:2]
    twohit_t0s = twohit_result[3:end-1]
    twohit_rms = twohit_result[end]
    
    println("\nTwo-hit model results:")
    println(@sprintf "  κc_0 (kappa * 1 pM) = %.4f s^-1" twohit_ks[1])
    println(@sprintf "  k2        = %.4f s^-1" twohit_ks[2])
    println("  t0 per dataset:")
    for (i, t0) in enumerate(twohit_t0s)
        println(@sprintf "    Dataset %d (%s): t0 = %.4f s" i nicenames[i] t0)
    end
    
    # Step 4: Compute and compare RMS errors
    println("\n" * "="^70)
    println("RMS ERROR COMPARISON")
    println("="^70)
    
    println(@sprintf "\nOne-hit model:  RMS error = %.6e" onehit_rms)
    println(@sprintf "Two-hit model:  RMS error = %.6e" twohit_rms)
    
    improvement = (onehit_rms - twohit_rms) / onehit_rms * 100
    if improvement > 0
        println(@sprintf "\nTwo-hit model is better by %.1f%%" improvement)
    else
        println(@sprintf "\nOne-hit model is better by %.1f%%" -improvement)
    end
    
    # Step 5: Create comparison plots
    println("\n" * "-"^70)
    println("Generating plots...")
    println("-"^70)
    
    # Plot: All datasets, both models
    p_all = plot_ml_cdf_comparison(tactiv_alldsets, nicenames, durations, cdfthresh,
                                    onehit_params, twohit_params, 
                                    onehit_cdf, twohit_cdf,
                                    onehit_rms, twohit_rms)
    plot_file = joinpath(plotdir, "ml_cdf_all_datasets_comparison.pdf")
    pl.savefig(p_all, plot_file)
    println("  Saved: $plot_file")
    
    # Plot: Individual dataset panels
    for i in 1:n_dsets
        p_single = plot_single_dataset_cdf(
            tactiv_alldsets[i], nicenames[i], durations[i], cdfthresh,
            onehit_params[1], onehit_params[2], onehit_params[2+i],
            twohit_params[1], twohit_params[2], twohit_params[2+i],
            onehit_cdf, twohit_cdf
        )
        plot_file = joinpath(plotdir, "ml_cdf_dataset_$(i)_$(nicenames[i]).pdf")
        pl.savefig(p_single, plot_file)
        println("  Saved: $plot_file")
    end
    
    println("\n" * "="^70)
    println("Analysis complete!")
    println("="^70 * "\n")
    
    return (onehit_params, twohit_params, onehit_rms, twohit_rms,
            nicenames, tactiv_alldsets, tunique_all, cumuls_all)
end


# Run the analysis
function (@main)(_)
    fit_ml_all_datasets(fracabove=0.6, cdfthresh=0.5)
end
