#= 
Fit a 1-step activation model on each dataset using 
Maximum Likelihood Estimation and plot the fitted observed 
effective rate, k_obs, as a function of concentration. 

Variant where we normalize each eCDF to 1, to follow what
was done in figure S2E. 

This script:
1. Loads all datasets in JR064_tracks
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
using Statistics: mean
using Measures: mm
using Optim: Optim, optimize, IPNewton
using LinearAlgebra: UniformScaling, diag, diagm, svd


# Load the data processing module
module tracksAnalysis
    include("analysis/tracks.jl")
end
using .tracksAnalysis: prepalldsets, concfromstr, cleannames, uniquecdf


### NORMALIZED CDF PREPARATION  ###
# We need to re-define prepcdf to normalize each CDF to 1. 
"""
Prepare empirical cumulative distribution function
from a list of activation times.
"""
function prepcdf_norm1(tactiv, cdfthresh, duration)
    # 1. Extract a sorted list of activation times
    t = sort(collect(skipmissing(tactiv[1, :])))

    # 2. remove duplicate experimental times, 
    # compute the cumulative number after each unique time
    tu, cumul = uniquecdf(t; addzero=false, makeunique=false)
    # For normalization, only cells that activate
    ncells = length(cumul)
    # Normalize by total number of cells
    cumul = cumul / ncells

    # 3. Get the upper limit of the early time, up to F_T(t) = cdfthresh
    # For cases with very few points (because activ threshold is large) 
    # or all points being activated immediately (so 1-2 points before cdf=0.5)
    # fit to all available points minus one
    ilim = findfirst(x -> (x !== missing && x >= cdfthresh), cumul)
    cdfthresh2 = cdfthresh
    if !isnothing(ilim)
        tlim = tu[ilim]
    else
        tlim = 0.0
    end
    # If too few points are included
    while (isnothing(ilim) || (ilim < 3) || tlim < 1e-6) && (cdfthresh2 < 1.0)
        cdfthresh2 += 0.1
        ilim = findfirst(x -> (x !== missing && x >= cdfthresh2), cumul)
        if !isnothing(ilim)
            tlim = tu[ilim]
        else
            tlim = 0.0
        end
    end
    # If the limit is above the largest fraction of active cells in the data
    if (isnothing(ilim) || (ilim < 3) || tlim < 1e-6)
        ilim = length(cumul)  # include all points
        tlim = duration
    end
    
    return tu, cumul, ilim, ncells, tlim
end


### MODELS PDFs AND CDFs ###

"""
For observed k fit, a simple 1-step (exponential) cdf. 
"""
function expocdf(t::Array, rate::Float64; t0::Float64=0.0)
    rtt0 = clamp!(rate .* (t .- t0), 0.0, Inf)
    cdf = 1.0 .- exp.(-rtt0)
    return cdf
end


"""
Probability density for a 1-step model (exponential). 
"""
function expopdf(t::Array, rate::Float64; t0::Float64=0.0)
    rtt0 = clamp!(rate .* (t .- t0), 0.0, Inf)
    pdf = rate .* exp.(-rtt0)
    return pdf
end

function logexpopdf(t::Array, rate::Float64; t0::Float64=0.0)
    rtt0 = clamp!(rate .* (t .- t0), 0.0, Inf)
    return log(rate) .- rtt0
end

function expoccdf(t::Array, rate::Float64; t0::Float64=0.0)
    rtt0 = clamp!(rate .* (t .- t0), 0.0, Inf)
    return exp.(-rtt0)
end


function logexpoccdf(t::Array, rate::Float64; t0::Float64=0.0)
    rtt0 = clamp!(rate .* (t .- t0), 0.0, Inf)
    return -rtt0
end


### LOG-LIKELIHOOD FUNCTIONS AND DERIVATIVES ###

"""
log-likelihood for 1 dataset for the 1-step model
"""
function expo_loglhood_1dset(p, times, ncell, klim0, klim1; te=nothing)
    k, t0 = p
    if te === nothing
        te = times[min(klim1, length(times))]
    end
    if (k < 0.0) || (times[1] < t0) 
        ll = -1e8
    else
        # Note that ncell includes those never activated, 
        # whereas times only includes cells eventually activated
        nc0 = klim0 - 1  # number of cells with t_activ = 0
        nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
        nc2 = ncell - nc0 - nc1
        if nc0 > 0
            ll = nc0 * log(expocdf([0.0], k; t0=t0)[1])
        else
            ll = 0.0
        end
        ll += sum(logexpopdf(times[klim0:klim1], k; t0=t0))
        if nc2 > 0
            ll += nc2 * logexpoccdf([te], k; t0=t0)[1]
        end
    end
    return ll
end


### RMS ERROR COMPUTATION ###

"""
Compute the root-mean-squared error of the fitted cumulative distribution 
function for each dataset TODO

Arguments:
- p_all: fitted parameters array, shape [2, n_datasets]
  and t0_n is the activation time offset for dataset n.
- tu_all: list of unique time arrays for each dataset
- cumuls_all: list of empirical cumulative distributions for each dataset
- ilim0s_all: list of lower indices (first point with t > 0) for each dataset
- ilim1s_all: list of upper indices (last point to include) for each dataset
- cdffct: the CDF function to use (e.g., expocdf)

Returns:
- rmserr: the root-mean-squared error computed over all datasets combined
"""
function compute_rms_all_dsets(p_all, tu_all, cumuls_all, ilim0s_all, 
                                ilim1s_all, cdffct)
    ndsets = length(tu_all)
    if ndsets !== size(p_all, 2)
        error("Number of datasets and parameters not matching")
    end
    rms_errors = zeros(Float64, ndsets)
    sres_total = 0.0
    npts_total = 0

    for n in 1:ndsets
        tu = tu_all[n]
        cumul = cumuls_all[n]
        ilim0 = ilim0s_all[n]
        ilim1 = ilim1s_all[n]
        k = p_all[1, n]
        t0 = p_all[2, n]

        # Determine starting index for RMS calculation
        rms_i = (ilim1 > ilim0) ? ilim0 : 1

        # Compute the fitted CDF for this dataset
        cdfestim = cdffct(tu[rms_i:ilim1], k; t0=t0)

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

"""
Prepare all cdfs with normalization, numbers of cells, limit on t0, cutoff time,
and so on from the list of activation time DataFrames.  
"""
function prepare_all_cdfs(tactiv_alldsets, dsetnames, durations, cdfthresh)
    # For each dataset in the list of datasets, prepare the cdf, 
    # numbers of cells, concentrations, and limits for the fit.
    conc_scale = 1.0  # pM
    n_dsets = length(tactiv_alldsets)
    concs_all = []
    ncell_all = []
    k0_all = []
    k1_all = []
    te_all = []
    tu_all = []
    cumuls_all = []
    for i in 1:1:n_dsets
        conc = concfromstr(dsetnames[i], "pM", ' ') / conc_scale
        # Prepare a unique list of cell number increase times to fit.
        tactiv = tactiv_alldsets[i]
        tu, cumul, ilim1, ncells, tlim = prepcdf_norm1(tactiv, cdfthresh, durations[i])
        # Find first cell with t > 0: use CDF term instead of pdf in log-likelihood
        # for cells before that, whose activation time could be before t = 0. 
        ilim0 = findfirst(tu .> 0.0)
        push!(tu_all, tu)
        push!(cumuls_all, cumul)
        push!(concs_all, conc)  # for now, all datasets at 1 pM
        push!(ncell_all, ncells)
        push!(k0_all, ilim0)
        push!(k1_all, ilim1)
        push!(te_all, tlim)
    end

    return concs_all, ncell_all, k0_all, k1_all, te_all, tu_all, cumuls_all
end


function fitmodel(tactiv_alldsets, dsetnames, durations, cdfthresh)
    n_dsets = length(tactiv_alldsets)
    res = prepare_all_cdfs(tactiv_alldsets, dsetnames, durations, cdfthresh)
    concs_all, ncell_all, k0_all, k1_all, te_all, tu_all, cumuls_all = res

    fit_params = zeros(Float64, 2, n_dsets)
    for i in 1:n_dsets
        lower = [1e-6, -20.0]
        upper = [10.0, te_all[i]-1e-3]
        init_p = [0.1, -0.1]

        klim0 = isnothing(k0_all[i]) ? 1 : k0_all[i]
        klim1 = k1_all[i]
        te = te_all[i]

        cost = p -> -expo_loglhood_1dset(p, tu_all[i], ncell_all[i], klim0, klim1; te=te)
        inner_optimizer = IPNewton()
        results = optimize(cost, lower, upper, init_p, inner_optimizer)

        if !Optim.converged(results)
            println("Dataset $(dsetnames[i]) did not converge for cdfthresh = $cdfthresh")
        end
        fit_params[:, i] = Optim.minimizer(results)
    end

    rms_errors, overall_rms = compute_rms_all_dsets(fit_params, 
                                tu_all, cumuls_all, k0_all, k1_all, expocdf)
    return fit_params, rms_errors, overall_rms
end




"""
Create a multi-panel plot comparing the empirical normalized CDFs to
fitted exponential model curves for each dataset.

Arguments:
- cdf_prep_results: output from prepare_all_cdfs
- nicenames: list of dataset names
- fit_params: 2 x n_dsets array of [k; t0] per dataset
- cdffct: exponential CDF function to plot
- overall_rms: overall RMS error for the fit
- durations: list of experiment durations
"""
function plot_ml_cdf_comparison_old(cdf_prep_results, nicenames, 
            fit_params, cdffct, overall_rms, durations)
    (concs_all, ncell_all, k0_all, k1_all, 
                te_all, tu_all, cumuls_all) = cdf_prep_results
    n_dsets = length(tu_all)

    nrows = ceil(Int, sqrt(n_dsets))
    ncols = ceil(Int, n_dsets / nrows)

    p = pl.plot(layout=(nrows, ncols), size=(400*ncols, 350*nrows),
                left_margin=3mm, right_margin=3mm, bottom_margin=3mm)
    tplotlim = [-5.0, maximum([maximum(a) for a in tu_all]) * 1.05]
    fracplotlim = [-0.05, 1.05]

    for i in 1:n_dsets
        tu = tu_all[i]
        cumul = cumuls_all[i]
        ilim1 = k1_all[i]

        goodt = vcat([0.0], tu)
        ncumul = vcat([0.0], cumul)

        k = fit_params[1, i]
        t0 = fit_params[2, i]
        trange = collect(LinRange(0, durations[i], 100))
        yfit = cdffct(trange, k; t0=t0)

        pl.plot!(p[i], goodt, ncumul, linetype=:steppost,
                 color=:black, linewidth=2, label="Data", alpha=0.7,
                 legend=:bottomright, legendfontsize=8,
                 xlim=tplotlim, ylim=fracplotlim)
        pl.scatter!(p[i], tu, cumul, color=:black,
                    markersize=3, label=nothing, alpha=0.5)
        pl.plot!(p[i], trange, yfit, label="Exponential fit",
                 color=:blue, linewidth=1.8, linestyle=:solid)

        pl.xlabel!(p[i], "Time (s)", fontsize=10)
        pl.ylabel!(p[i], "Fraction active cells", fontsize=10)
        pl.title!(p[i], nicenames[i], fontsize=11)
    end
    for i in n_dsets+1:nrows * ncols
        pl.plot!(p[i], [], [], axis=false, legend=false, grid=false)
    end

    pl.plot!(p[1], title=@sprintf("Exponential fit for %d datasets | RMS = %.3e", n_dsets, overall_rms), plot_title_fontsize=12)
    return p
end

"""
Create a single-panel plot comparing the empirical normalized CDFs to
fitted exponential model curves for each dataset.

Arguments:
- cdf_prep_results: output from prepare_all_cdfs
- nicenames: list of dataset names
- fit_params: 2 x n_dsets array of [k; t0] per dataset
- cdffct: exponential CDF function to plot
- overall_rms: overall RMS error for the fit
- durations: list of experiment durations
"""
function plot_ml_cdf_comparison(cdf_prep_results, nicenames, 
            fit_params, cdffct, overall_rms, durations)
    (concs_all, ncell_all, k0_all, k1_all, 
                te_all, tu_all, cumuls_all) = cdf_prep_results
    n_dsets = length(tu_all)

    lblconcs = [concfromstr(name, "pM", ' ') for name in nicenames]
    if any(x -> x === nothing, lblconcs)
        error("Some dataset names do not contain a parsable concentration value")
    end
    conc_labels = [@sprintf("%.3g", c) for c in lblconcs]
    colors = pl.cgrad(:magma)

    tplotlim = [-5.0, maximum([maximum(a) for a in tu_all]) * 1.05]
    fracplotlim = [-0.05, 1.05]

    p = pl.plot(size=(900, 600), left_margin=3mm, right_margin=3mm,
                bottom_margin=3mm, xlabel="Time (s)", ylabel="Fraction active cells",
                xlim=tplotlim, ylim=fracplotlim, legend=:bottomright,
                tickfont=pl.font(13), legendfont=pl.font(11), legendfontsize=11,
                titlefontsize=12, legendmargin=8mm)

    for i in 1:n_dsets
        tu = tu_all[i]
        cumul = cumuls_all[i]
        ilim1 = k1_all[i]

        goodt = vcat([0.0], tu)
        ncumul = vcat([0.0], cumul)

        k = fit_params[1, i]
        t0 = fit_params[2, i]
        trange = collect(LinRange(0, durations[i], 200))
        yfit = cdffct(trange, k; t0=t0)
        line_color = colors[(i-1)/(n_dsets)]
        conc_label = conc_labels[i]

        pl.plot!(p, goodt, ncumul, linetype=:steppost,
                 color=:grey, linewidth=3.0, label="Data $conc_label pM", alpha=0.8)
        pl.scatter!(p, tu, cumul, color=:grey, markersize=3,
                    label=nothing, alpha=0.6)
        pl.plot!(p, trange, yfit, label="\$k_{\\mathrm{obs}}\$ fit " * conc_label * " pM",
                 color=line_color, linewidth=4.4, linestyle=:solid)
    end

    return p
end


"""
Plot fitted k_obs for each dataset as a function of concentration. 
"""
function plot_kobs_vs_concentration(nicenames, fit_params, ncell_all)
    lblconcs = [concfromstr(name, "pM", ' ') for name in nicenames]
    valid = findall(x -> x !== nothing, lblconcs)
    if length(valid) != length(lblconcs)
        error("Some dataset names do not contain a parsable concentration value")
    end
    unique_concs = unique(sort(lblconcs))

    selected_indices = Int[]
    for conc in unique_concs
        indices_for_conc = findall(c -> c ≈ conc, lblconcs)
        cells_for_conc = [ncell_all[i] for i in indices_for_conc]
        max_idx_in_group = indices_for_conc[argmax(cells_for_conc)]
        push!(selected_indices, max_idx_in_group)
    end

    selected_concs = [lblconcs[i] for i in selected_indices]
    fitted_ks = [fit_params[1, i] for i in selected_indices]

    p = pl.plot(selected_concs, fitted_ks, marker=:circle, markersize=7,
                linewidth=2, xlabel="Concentration (pM)", ylabel="Fitted rate k (s^-1)",
                legend=false, left_margin=3mm, right_margin=3mm, bottom_margin=3mm)
    pl.plot!(selected_concs, fitted_ks, linestyle=:solid, color=:blue)
    pl.title!(p, "Fitted exponential rate vs particle concentration", fontsize=12)
    return p
end


### MAIN ANALYSIS FUNCTION ###

function fit_ml_all_datasets(; 
    datadir="data/JR064_tracks",
    plotdir="figures/ml_fits_kobs/",
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
        "pHELD-AF594_10pM_z2.csv", 
        "pHELD-AF594_10pM_2_z2.csv", 
        "pHELD-AF594_1pM_2.csv", 
        #"pHELD-AF594_1pM.csv",   # low cell count, dropped dataset
        "pHELD-AF594_0.1pM.csv",
        "pHELD-AF594_0.1pM_2.csv", 
        "pHELD-AF594_0.01pM.csv", 
        "pHELD-AF594_0.01pM_2.csv"
    ]
    fnames = collect(fp for fp in fnames if fp in approved_files)
    fpaths = collect(joinpath(datadir, f) for f in fnames)

    # Sort names in decreasing concentration order
    # Extract concentration of each name
    shortnames, lblconcs = cleannames(fnames)
    sortidx = reverse(sortperm(lblconcs), dims=1)
    shortnames = shortnames[sortidx]
    fnames = fnames[sortidx]
    fpaths = fpaths[sortidx]
    
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

    # Step 2: Fit the exponential model to each dataset separately
    println("\n" * "-"^70)
    println("Fitting exponential model for each dataset separately...")
    println("-"^70)
    fitres = fitmodel(tactiv_alldsets, nicenames, durations, cdfthresh)
    fit_params, rms_errors, overall_rms = fitres

    println("\nExponential fit results:")
    for i in 1:n_dsets
        println(@sprintf("  Dataset %d (%s): k = %.4f s^-1, t0 = %.4f s, RMS = %.3e", 
                         i, nicenames[i], fit_params[1,i], fit_params[2,i], rms_errors[i]))
    end
    println(@sprintf("\nOverall RMS = %.3e", overall_rms))

    # Select one dataset per concentration for plotting
    lblconcs = [concfromstr(name, "pM", ' ') for name in nicenames]
    unique_concs = unique(sort(filter(x -> x !== nothing, lblconcs)))
    selected_indices = Int[]
    for conc in unique_concs
        indices_for_conc = findall(c -> c ≈ conc, lblconcs)
        cells_for_conc = [length(tactiv_alldsets[i][1, :]) for i in indices_for_conc]
        max_idx = indices_for_conc[argmax(cells_for_conc)]
        push!(selected_indices, max_idx)
    end

    selected_tactiv = tactiv_alldsets[selected_indices]
    selected_durations = durations[selected_indices]
    selected_names = nicenames[selected_indices]
    selected_fit_params = fit_params[:, selected_indices]

    println("\nSelected $(length(selected_indices)) datasets for plotting:")
    for (j, name) in enumerate(selected_names)
        println("  [$j] $name")
    end

    # Step 3: Create comparison plots
    println("\n" * "-"^70)
    println("Generating plots...")
    println("-"^70)

    prep_cdf_res = prepare_all_cdfs(selected_tactiv, selected_names, selected_durations, cdfthresh)

    p_all = plot_ml_cdf_comparison(prep_cdf_res, selected_names, selected_fit_params, 
                        expocdf, overall_rms, selected_durations)
    plot_file = joinpath(plotdir, "exp_cdf_all_datasets.pdf")
    pl.savefig(p_all, plot_file)
    println("  Saved: $plot_file")

    p_conc = plot_kobs_vs_concentration(selected_names, selected_fit_params, prep_cdf_res[2])
    plot_file = joinpath(plotdir, "exp_k_vs_concentration.pdf")
    pl.savefig(p_conc, plot_file)
    println("  Saved: $plot_file")

    println("\n" * "="^70)
    println("Analysis complete!")
    println("="^70 * "\n")

    return (fit_params, overall_rms, nicenames, tactiv_alldsets)
end


# Run the analysis
function (@main)(_)
    fit_ml_all_datasets(fracabove=0.8, cdfthresh=1.0)
    return 0
end
