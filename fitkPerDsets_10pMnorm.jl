#= 
Fit CDF activation models using Maximum Likelihood Estimation
across multiple datasets in the JR064 experiment. The cell-intrinsic
rate is fitted across datasets, but the hit rate is fitted
on each dataset, to account for replicate variability. 

We normalize CDFs based on the fraction of cells that never activate 
in the 10 pM experiments. We assume that such a high concentration, 
all cells that could have been activated will have been by the end of a movie;
tracks that do not must be from a distinct population of dead/anergic cells
and spurios tracks, not expected to follow the activation models. 
There always seems to be a 15-20 % of tracks that are in that situation. 

This script:
1. Loads all valid datasets in JR064_tracks
2. Preprocesses activation times
3. Defines one-hit and two-hit models
3. Fits one-hit and two-hit models across all datasets simultaneously
4. Computes RMS errors for each model
5. Generates comparison plots

François X. P. Bourassa, 2026
=#

using DataFrames: DataFrames as DF
using Plots: Plots as pl
using Printf: @sprintf
using Measures: mm
using Optim: Optim, optimize, IPNewton
using LinearAlgebra: UniformScaling, diag, diagm, svd

pl.default(fontfamily="Arial")


# Load the data processing module
module tracksAnalysis
    include("analysis/tracks.jl")
end
using .tracksAnalysis: prepalldsets, concfromstr, cleannames, 
    loadalldsets, threshalldsets, prepare_all_cdfs, get_csv_file_names

module modelsML
    include("analysis/modelsML.jl")
end
using .modelsML: onehit_cdf, onehit_ccdf, onehit_pdf, onehit_loglhood_total,
    twohit_cdf, twohit_ccdf, twohit_pdf, twohit_loglhood_total,
    threehit_cdf, threehit_ccdf, threehit_pdf, threehit_loglhood_total

module tracksPlots
    include("analysis/tracksPlots.jl")
end
using .tracksPlots: plotrobustness

conc_scale = 1.0  # pM


### RMS ERROR COMPUTATION ###

"""
Compute the root-mean-squared error of the fitted cumulative distribution 
function across all datasets.

Arguments:
- p_all: fitted parameters array [k2, kc_1, kc_2, ..., kc_n, t0_1, t0_2, ..., t0_n]
  where k2 is shared across all datasets, while kc_n and t0_n
  are the hit rate and the activation time offset for dataset n.
- tu_all: list of unique time arrays for each dataset
- cumuls_all: list of empirical cumulative distributions for each dataset
- ilim0s_all: list of lower indices (first point with t > 0) for each dataset
- ilim1s_all: list of upper indices (last point to include) for each dataset
- concs_all: list of concentrations for each dataset
- cdffct: the CDF function to use (e.g., onehit_cdf, twohit_cdf)

Returns:
- rms_errors: the root-mean-squared error computed on each dataset
- overall_rms: the total RMS across all datasets
"""
function compute_rms_all_dsets(p_all, tu_all, cumuls_all, ilim0s_all, 
                                ilim1s_all, concs_all, cdffct)
    n_dsets = length(tu_all)
    rms_errors = zeros(Float64, n_dsets)
    sres_total = 0.0
    npts_total = 0
    
    k2 = p_all[1]
    
    for n in 1:n_dsets
        kc = p_all[1 + n]
        tu = tu_all[n]
        cumul = cumuls_all[n]
        ilim0 = ilim0s_all[n]
        ilim1 = ilim1s_all[n]
        conc = concs_all[n]
        t0 = p_all[1 + n_dsets + n]  # t0 for dataset n
        
        # Determine starting index for RMS calculation
        # Prevent issues if there is no point with t_activ > 0
        rms_i = (ilim1 > ilim0) ? ilim0 : 1
        
        # Compute the fitted CDF for this dataset
        cdfestim = cdffct(tu[rms_i:ilim1], kc, k2; t0=t0)
        
        # Accumulate squared residuals
        dataset_sres = sum((cumul[rms_i:ilim1] .- cdfestim).^2)
        dataset_npts = (ilim1 - rms_i + 1)
        rms_errors[n] = sqrt(dataset_sres / dataset_npts)

        sres_total += dataset_sres
        npts_total += dataset_npts
    end

    # Compute overall RMS error for the full fit across all datasets
    overall_rms = sqrt(sres_total / npts_total)
    return rms_errors, overall_rms
end


### MAIN FITTING FUNCTIONS ###


function fitmodel(tactiv_alldsets, dsetnames, durations, cdfthresh, loglfct, cdffct)
    # kc estimate: diffusion-limited, 0.4 pM^-1 s^-1, 
    # times concentration with default unit scale of 1 pM. 
    # k2 estimate: fast to begin with
    lower = [1e-4]  # k2
    upper = [20.0]
    init_p = [0.1]
    
    # For each dataset in the list of datasets, prepare the cdf, 
    # numbers of cells, concentrations, and limits for the fit.
    res = prepare_all_cdfs(tactiv_alldsets, dsetnames, durations, cdfthresh)
    concs_all, ncell_all, k0_all, k1_all, te_all, tu_all, cumuls_all = res

    # Limits for parameter kc of each dataset
    n_dsets = length(tactiv_alldsets)
    for i in 1:1:n_dsets
        concs_all
        push!(lower, 1e-4 * concs_all[i])  # kc lower bound, times conc
        push!(upper, 20.0 * concs_all[i])  # kc upper bound, times conc
        push!(init_p, 0.4 * concs_all[i])  # kc initial guess, diffusion-limit
    end
    

    # Limits for parameter t0 for each dataset: first recorded time point
    for i in 1:1:n_dsets
        push!(lower, -20.0)
        push!(upper, tu_all[i][1]-1e-3)
        push!(init_p, -0.1)
    end


    # Minimize -log likelihood
    cost = p_all -> -loglfct(p_all, tu_all, concs_all, ncell_all, 
                        k0_all, k1_all; te_all=te_all)
    inner_optimizer = IPNewton()  # or NelderMead()
    #println("Start linear fit with data points: ", tu)
    
    results = optimize(cost,
                lower, upper, 
                init_p, inner_optimizer
            )
    #println("Finished maximum likelihood fit")

    # Get the results
    fit_params = Optim.minimizer(results)
    k2 = fit_params[1]
    kcs = fit_params[2:1+n_dsets]
    t0s = fit_params[2+n_dsets:end]
    flag = Optim.converged(results)
    if !flag
        println("Did not converge for cdfhresh = ", cdfthresh)
        println("k1_all = ", k1_all, " , ncells = ", ncell_all, " , init_p = ", init_p)
    end

    # RMS error on the cdf across all datasets
    rms_per_dset, rmserr = compute_rms_all_dsets(
        fit_params, tu_all, cumuls_all, 
        k0_all, k1_all, concs_all, cdffct)
    
    result = Array{Float64}(vcat(fit_params, [rmserr]))

    return result, rms_per_dset
end


"""
Create a multi-panel plot comparing one-hit and two-hit models across all datasets.

Arguments:
- tactiv_alldsets: list of activation time dataframes
- nicenames: list of dataset names
- durations: list of experiment durations
- cdfthresh: CDF threshold used for fitting
- onehit_params: [k2, kc_1, kc_2, ..., kc_n, t0_1, t0_2, ..., t0_n]
- twohit_params: [k2, kc_1, kc_2, ..., kc_n, t0_1, t0_2, ..., t0_n]
- threehit_params: [k2, kc_1, kc_2, ..., kc_n, t0_1, t0_2, ..., t0_n]
- onehit_cdf, twohit_cdf, threehit_cdf: CDF functions
- onehit_rms, twohit_rms, threehit_rms: RMS errors for model comparison
"""
function plot_ml_cdf_comparison(cdf_prep_results, nicenames, 
                                 onehit_params, twohit_params, threehit_params,
                                 onehit_cdf, twohit_cdf, threehit_cdf,
                                 onehit_rms, twohit_rms, threehit_rms)
    (concs_all, ncell_all, k0_all, k1_all, 
                te_all, tu_all, cumuls_all) = cdf_prep_results
    n_dsets = length(tu_all)

    # Determine layout (try to make roughly square)
    nrows = ceil(Int, sqrt(n_dsets))
    ncols = ceil(Int, n_dsets / nrows)
    
    # Create subplot layout
    p = pl.plot(layout=(nrows, ncols), size=(400*ncols, 350*nrows),
                left_margin=3mm, right_margin=3mm, bottom_margin=3mm)
    tplotlim = [-5.0, maximum([maximum(a) for a in tu_all]) * 1.05]
    fracplotlim = [-0.05, 1.05]
    for i in 1:n_dsets
        # Extract concentration
        conc = concfromstr(nicenames[i], "pM", ' ') / conc_scale
        
        tu = tu_all[i]
        cumul = cumuls_all[i]
        ilim1 = k1_all[i]
        ncells = ncell_all[i]
        tlim = te_all[i]
        ncells = 1  # plot normalized cdfs
        
        # Scale CDF for plotting
        goodt = vcat([0], tu)
        ncumul = vcat([0], cumul .* ncells)
        
        # Get parameters for this dataset
        k2_1, kc_1, t0_1 = onehit_params[1], onehit_params[1+i], onehit_params[1+n_dsets+i]
        k2_2, kc_2, t0_2 = twohit_params[1], twohit_params[1+i], twohit_params[1+n_dsets+i]
        k2_3, kc_3, t0_3 = threehit_params[1], threehit_params[1+i], threehit_params[1+n_dsets+i]
        
        # Generate smooth curves
        trange = collect(LinRange(0, tu[end], 100))
        
        cdf1 = onehit_cdf(trange, kc_1, k2_1; t0=t0_1)
        y1 = cdf1 .* ncells
        
        cdf2 = twohit_cdf(trange, kc_2, k2_2; t0=t0_2)
        y2 = cdf2 .* ncells
        
        cdf3 = threehit_cdf(trange, kc_3, k2_3; t0=t0_3)
        y3 = cdf3 .* ncells
        
        # Plot on subplot i
        pl.plot!(p[i], goodt, ncumul, linetype=:steppost,
                 color=:black, linewidth=2, label="Data", alpha=0.7, 
                 legend=:bottomright, legendfontsize=8, 
                 xlim=tplotlim, ylim=fracplotlim)
        
        pl.scatter!(p[i], tu, cumul .* ncells, color=:black,
                    markersize=3, label=nothing, alpha=0.5)
        
        pl.plot!(p[i], trange, y1, label="1-hit",
                 color=:blue, linewidth=1.8, linestyle=:solid)
        
        pl.plot!(p[i], trange, y2, label="2-hit",
                 color=:red, linewidth=1.8, linestyle=:dash)
        
        pl.plot!(p[i], trange, y3, label="3-hit",
                 color=:purple, linewidth=1.8, linestyle=:dot)
        
        pl.xlabel!(p[i], "Time (s)", fontsize=10)
        pl.ylabel!(p[i], "Fraction active cells", fontsize=10)
        pl.title!(p[i], nicenames[i], fontsize=11)
    end
    for i in n_dsets+1:1:nrows * ncols
        pl.plot!(p[i], [], [], axis=false, legend=false, grid=false)
    end
    
    # Add overall title with RMS comparison
    #title_text = @sprintf "ML Fits: %d dataset(s) | RMS: 1-hit=%.2e, 2-hit=%.2e" n_dsets onehit_rms twohit_rms
    #pl.plot!(title=title_text, plot_title_fontsize=12)
    
    return p
end


### MAIN ANALYSIS FUNCTION ###

function fit_ml_all_datasets(; 
    datadir="data/JR064_tracks",
    plotdir="figures/ml_fits_kperdset_10pMnorm/",
    fracabove=0.8,
    cdfthresh=0.75)
    
    # Create output directory if it doesn't exist
    if !isdir(plotdir)
        mkpath(plotdir)
    end
    
    # Get a cleaned up list of data file names and paths
    fnames, fpaths = get_csv_file_names(datadir)
    
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
    println("Fitting ONE-HIT model (overall k2, one kc and one t0 per dataset)...")
    println("-"^70)
    onehit_result, onehit_all_rms = fitmodel(tactiv_alldsets, 
                    nicenames, durations, cdfthresh, 
                    onehit_loglhood_total, onehit_cdf)
    onehit_params = onehit_result[1:end-1]
    onehit_k2 = onehit_params[1]
    onehit_kcs = onehit_result[2:1+n_dsets]
    onehit_t0s = onehit_result[2+n_dsets:end-1]
    onehit_rms = onehit_result[end]
    
    println("\nOne-hit model results:")
    println(@sprintf "  k2        = %.4f s^-1" onehit_k2)
    println("  t0 per dataset:")
    for (i, t0) in enumerate(onehit_t0s)
        println(@sprintf "    Dataset %d (%s):  kc = %.4f s^-1" i nicenames[i] onehit_kcs[i])
        println(@sprintf "    Dataset %d (%s): t0 = %.4f s" i nicenames[i] t0)
    end
    
    # Step 3: Fit the two-hit model
    println("\n" * "-"^70)
    println("Fitting TWO-HIT model (overall k2, one kc and one t0 per dataset)...")
    println("-"^70)
    twohit_result, twohit_all_rms = fitmodel(tactiv_alldsets, 
                        nicenames, durations, cdfthresh,
                        twohit_loglhood_total, twohit_cdf)
    twohit_params = twohit_result[1:end-1]
    twohit_k2 = twohit_params[1]
    twohit_kcs = twohit_result[2:1+n_dsets]
    twohit_t0s = twohit_result[2+n_dsets:end-1]
    twohit_rms = twohit_result[end]
    
    println("\nTwo-hit model results:")
    println(@sprintf "  k2        = %.4f s^-1" twohit_k2)
    println("  t0 per dataset:")
    for (i, t0) in enumerate(twohit_t0s)
        println(@sprintf "    Dataset %d (%s):  kc = %.4f s^-1" i nicenames[i] twohit_kcs[i])
        println(@sprintf "    Dataset %d (%s): t0 = %.4f s" i nicenames[i] t0)
    end
    
    # Step 4: Fit the three-hit model
    println("\n" * "-"^70)
    println("Fitting THREE-HIT model (overall k2, one kc and one t0 per dataset)...")
    println("-"^70)
    threehit_result, threehit_all_rms = fitmodel(tactiv_alldsets, 
                            nicenames, durations, cdfthresh,
                            threehit_loglhood_total, threehit_cdf)
    threehit_params = threehit_result[1:end-1]
    threehit_k2 = threehit_params[1]
    threehit_kcs = threehit_result[2:1+n_dsets]
    threehit_t0s = threehit_result[2+n_dsets:end-1]
    threehit_rms = threehit_result[end]
    
    println("\nThree-hit model results:")
    println(@sprintf "  k2        = %.4f s^-1" threehit_k2)
    println("  t0 per dataset:")
    for (i, t0) in enumerate(threehit_t0s)
        println(@sprintf "    Dataset %d (%s):  kc = %.4f s^-1" i nicenames[i] threehit_kcs[i])
        println(@sprintf "    Dataset %d (%s): t0 = %.4f s" i nicenames[i] t0)
    end

    # Step 5: Compute and compare RMS errors
    println("\n" * "="^70)
    println("RMS ERROR COMPARISON")
    println("="^70)

    println("RMS per dataset:")
    for (i, t0) in enumerate(threehit_t0s)
        println(@sprintf "\n    Dataset %d (%s), one-hit = %.6e" i nicenames[i] onehit_all_rms[i])
        println(@sprintf "    Dataset %d (%s), two-hit = %.6e" i nicenames[i] twohit_all_rms[i])
        println(@sprintf "    Dataset %d (%s), three-hit = %.6e" i nicenames[i] threehit_all_rms[i])
    end
    
    println("\n" * "-"^70)
    println("Total RMS:")
    println(@sprintf "    One-hit model:  RMS error = %.6e" onehit_rms)
    println(@sprintf "    Two-hit model:  RMS error = %.6e" twohit_rms)
    println(@sprintf "    Three-hit model:  RMS error = %.6e" threehit_rms)
    
    improvement = (onehit_rms - twohit_rms) / onehit_rms * 100
    if improvement > 0
        println(@sprintf "\nTwo-hit model is better by %.1f%%" improvement)
    else
        println(@sprintf "\nOne-hit model is better by %.1f%%" -improvement)
    end
    
    # Step 6: Create comparison plots
    println("\n" * "-"^70)
    println("Generating plots...")
    println("-"^70)

    # Prepare all empirical CDFs
    prep_cdf_res = prepare_all_cdfs(tactiv_alldsets, nicenames, durations, cdfthresh)
    
    # Plot: All datasets, both models
    p_all = plot_ml_cdf_comparison(prep_cdf_res, nicenames, 
                                    onehit_params, twohit_params, threehit_params,
                                    onehit_cdf, twohit_cdf, threehit_cdf,
                                    onehit_rms, twohit_rms, threehit_rms)
    plot_file = joinpath(plotdir, "ml_cdf_all_datasets_comparison.pdf")
    pl.savefig(p_all, plot_file)
    println("  Saved: $plot_file")

    println("\n" * "="^70)
    println("Analysis complete!")
    println("="^70 * "\n")
    
    return (onehit_params, twohit_params, threehit_params, 
            onehit_rms, twohit_rms, threehit_rms,
            nicenames, tactiv_alldsets)
end



# Robustness analysis
# as we vary the fluorescence threshold
# Main analysis + plotting function for varying the fluorescence threshold
function robustness_vs_thresh_main(; 
    datadir="data/JR064_tracks",
    plotdir="figures/ml_fits_kperdset_10pMnorm/",
    fracabove_default=0.8,
    cdfthresh=0.75, 
    dosave=true)
    
    # Create output directory if it doesn't exist
    if !isdir(plotdir)
        mkpath(plotdir)
    end
    
    # Get a cleaned up list of data file names and paths
    fnames, fpaths = get_csv_file_names(datadir)

    nfluopts = 50
    
    # Save CDF fits for each file and each threshold fraction tried
    # number of parameters per dataset: one k2 plus (kc, t0) for each dset
    n_dsets = length(fnames)
    all_p1 = zeros(Float64, nfluopts, 1 + 2*n_dsets)
    all_rms1 = zeros(Float64, nfluopts, n_dsets)
    all_p2 = zeros(Float64, nfluopts, 1 + 2*n_dsets)
    all_rms2 = zeros(Float64, nfluopts, n_dsets)
    all_p3 = zeros(Float64, nfluopts, 1 + 2*n_dsets)
    all_rms3 = zeros(Float64, nfluopts, n_dsets)
    
    fluothreshrange = collect(LinRange(0.1, 1.5, nfluopts))

    # Load data once
    tracks_alldsets, nicenames = loadalldsets(fpaths; disp=false)

    # Repeat analysis for a range of fluorescence thresholds
    for j in 1:1:nfluopts
        tactiv_alldsets, durations = threshalldsets(
            tracks_alldsets, nicenames, fluothreshrange[j]; disp=false)
    
        # Fit the one-hit, two-hit, and three-hit models
        p1, rms1 = fitmodel(tactiv_alldsets, nicenames, durations, cdfthresh, 
                    onehit_loglhood_total, onehit_cdf)
        all_p1[j, :] = p1[1:end-1]  # last entry is RMS
        all_rms1[j, :] = rms1

        p2, rms2 = fitmodel(tactiv_alldsets, nicenames, durations, cdfthresh, 
                    twohit_loglhood_total, twohit_cdf)
        all_p2[j, :] = p2[1:end-1]
        all_rms2[j, :] = rms2

        p3, rms3 = fitmodel(tactiv_alldsets, nicenames, durations, cdfthresh, 
                    threehit_loglhood_total, threehit_cdf)
        all_p3[j, :] = p3[1:end-1]
        all_rms3[j, :] = rms3
        println(@sprintf "Finished fits for fluorescence threshold factor = %.2f" fluothreshrange[j])
    end

    # Plot slope and goodness of linear vs quadratic fits
    # as a function of the threshold fraction
    # linfits, quadfits, allnames, fracrange, fname

    allplots = plotrobustness((all_p1, all_p2, all_p3), 
                    (all_rms1, all_rms2, all_rms3), fnames, fluothreshrange; 
                    fluothresh_default=fracabove_default)
    
    # Display and save the resulting plots
    ppp, pp2, pp3 = allplots

    param_names = ["kc", "t0"]
    for i in 1:1:2
        nm = param_names[i]
        fname = joinpath(plotdir, "paramfits_$(nm)_vs_threshold.pdf")
        display(ppp[i])
        pl.savefig(ppp[i], fname)
    end

    fname = joinpath(plotdir, "onehit_vs_twohit_fitRMS.pdf")
    display(pp2)
    if dosave
        pl.savefig(pp2, fname)
    end

    fname = joinpath(plotdir, "best12_vs_3-hit_fitRMS.pdf")
    display(pp3)
    if dosave
        pl.savefig(pp3, fname)
    end
    return 0
end



# Run the analysis
function (@main)(_)
    fit_ml_all_datasets(fracabove=0.8, cdfthresh=0.75)
    robustness_vs_thresh_main(fracabove_default=0.8, cdfthresh=0.75)
    return 0
end
