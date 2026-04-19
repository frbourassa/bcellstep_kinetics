#= 
Fit a 1-step activation model on each dataset using 
Maximum Likelihood Estimation and plot the fitted observed 
effective rate, k_obs, as a function of concentration. 

Variant where we normalize each eCDF based on the fraction
of cells that do not get activated even at 10 pM: these are
likely dead or anergic B cells, or spurious tracks. We assume
that sub-population is as abundant in all datasets, and so
remove at most that fraction of inactive tracks in lower
concentration conditions. 

This script:
1. Loads all datasets in JR064_tracks
2. Preprocesses activation times
3. Fits one-hit, two-hit, three-hit models across all datasets
4. Computes RMS errors for each model
5. Generates comparison plots

François X. P. Bourassa, 2026
=#

using Plots: Plots as pl
using Printf: @sprintf
using Measures: mm
using Optim: Optim, optimize, IPNewton


# Load the data processing module
module tracksAnalysis
    include("analysis/tracks.jl")
end
using .tracksAnalysis: prepalldsets, concfromstr,
    prepare_all_cdfs, get_csv_file_names


module modelsML
    include("analysis/modelsML.jl")
end
using .modelsML: expocdf, expopdf, expoccdf, logexpoccdf, logexpopdf, 
    expo_loglhood_1dset, hess_minuslogl_expo!, deriv_minuslogl_expo!


pl.default(fontfamily="Arial")


### RMS ERROR COMPUTATION ###

"""
Compute the root-mean-squared error of the fitted cumulative distribution 
function for each dataset

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


### FITTING FUNCTIONS ###


function fitmodel(tactiv_alldsets, dsetnames, durations, cdfthresh)
    n_dsets = length(tactiv_alldsets)
    prepres = prepare_all_cdfs(tactiv_alldsets, dsetnames, durations, cdfthresh)
    concs_all, ncell_all, k0_all, k1_all, te_all, tu_all, cumuls_all = prepres

    fit_params = zeros(Float64, 2, n_dsets)
    fit_params_errors = zeros(Float64, 2, n_dsets)
    for i in 1:n_dsets
        lower = [1e-6, -20.0]
        upper = [10.0, tu_all[i][1]-1e-3]
        init_p = [0.1, -0.1]

        klim0 = isnothing(k0_all[i]) ? 1 : k0_all[i]
        klim1 = k1_all[i]
        te = te_all[i]

        cost = p -> -expo_loglhood_1dset(p, 
                            tu_all[i], ncell_all[i], klim0, klim1; te=te)
        minusgrad! = (g, p) -> deriv_minuslogl_expo!(g, p, 
                                tu_all[i], ncell_all[i], klim0, klim1; te=te)
        minushess! = (h, p) -> hess_minuslogl_expo!(h, p, 
                                tu_all[i], ncell_all[i], klim0, klim1; te=te)
        inner_optimizer = IPNewton()
        results = optimize(cost, minusgrad!, minushess!, 
                            lower, upper, init_p, inner_optimizer)

        if !Optim.converged(results)
            println("Dataset $(dsetnames[i]) did not converge for cdfthresh = $cdfthresh")
        end
        params = Optim.minimizer(results)
        fit_params[:, i] = params

        # Get variance of these parameter fits. Hessian is not positive definite
        # because the likelihood is monotonically increasing with t0, 
        # so best fit is t0 = boundary, k* derived from dlogL/dk = 0
        # So the uncertainty on the slope is estimated by inverting the FIM = -hessian^-1 element, 
        # while that on t0 is the uncertainty on an experimental time
        # so let's say (0.5 * time bin)^2. 
        minushessmat = zeros(2, 2)
        minushess!(minushessmat, params)
        varislope = 1.0 / minushessmat[1, 1]
        varit0 = (0.5 * (tu_all[i][2] - tu_all[i][1]))^2
        fit_params_errors[:, i] = [sqrt(varislope), sqrt(varit0)]

    end

    rms_errors, overall_rms = compute_rms_all_dsets(fit_params, 
                                tu_all, cumuls_all, k0_all, k1_all, expocdf)
    return fit_params, fit_params_errors, rms_errors, overall_rms, prepres
end


### PLOTTING FUNCTIONS ###

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
        line_color = colors[(2*i-2)/(2*n_dsets-1)]
        conc_label = conc_labels[i]
        conc_label = nicenames[i]

        pl.plot!(p, goodt, ncumul, linetype=:steppost,
                 color=:grey, linewidth=3.0, label="Data $conc_label", alpha=0.8)
        # Legend spacing
        pl.scatter!(p, [minimum(tu)],[0], label=" ", ms=0, mc=:white, msc=:white)
        pl.scatter!(p, tu, cumul, color=:grey, markersize=3,
                    label=nothing, alpha=0.6)
        pl.plot!(p, trange, yfit, label="\$k_{\\mathrm{obs}}\$ fit " * conc_label,
                 color=line_color, linewidth=4.4, linestyle=:solid)
        pl.scatter!(p, [minimum(tu)],[0], label=" ", ms=0, mc=:white, msc=:white)
    end

    return p
end

function make_paired_colors(nicenames)
    lblconcs = [concfromstr(name, "pM", ' ') for name in nicenames]
    if any(x -> x === nothing, lblconcs)
        error("Some dataset names do not contain a parsable concentration value")
    end
    n_dsets = length(nicenames)
    tab20 = pl.palette(:tab20, 20)
    conc_colors = Dict(
        0.01 => (tab20[1], tab20[2]),
        0.1  => (tab20[3], tab20[4]),
        10.0 => (tab20[5], tab20[6]),
        1.0  => (tab20[7], tab20[8])
    )

    grouped_indices = Dict{Float64, Vector{Int}}()
    for i in 1:n_dsets
        c = lblconcs[i]
        push!(get!(grouped_indices, c, Int[]), i)
    end

    line_colors = Vector{Any}(undef, n_dsets)
    for (conc, inds) in grouped_indices
        if !haskey(conc_colors, conc)
            conc_colors[conc] = (tab20[9], tab20[10])
        end
        pair = conc_colors[conc]
        if length(inds) == 1
            line_colors[inds[1]] = pair[1]
        else
            line_colors[inds[1]] = pair[1]
            line_colors[inds[2]] = pair[2]
            for j in 3:length(inds)
                line_colors[inds[j]] = pair[1]
            end
        end
    end
    return grouped_indices, line_colors
end


"""
Plot fitted k_obs for each dataset as a function of concentration. 
"""
function plot_kobs_vs_concentration(nicenames, fit_params, fit_errors, ncell_all)
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
    fit_ks_errs = [fit_errors[1, i] for i in selected_indices]

    p = pl.plot(selected_concs, fitted_ks, yerr=fit_ks_errs, marker=:circle, markersize=7,
                linewidth=2, xlabel="Concentration (pM)", ylabel="Fitted rate k (s^-1)",
                legend=false, left_margin=3mm, right_margin=3mm, bottom_margin=3mm)
    pl.plot!(selected_concs, fitted_ks, linestyle=:solid, color=:blue)
    pl.title!(p, "Fitted exponential rate vs particle concentration", fontsize=12)
    return p
end


"""
Plot fitted k_obs for all replicate datasets on a log-log scale.
Also overlay the diffusion-limited line k_obs = k0 * c with k0 = 0.4 pM^-1 s^-1.
"""
function plot_kobs_replicates_loglog(nicenames, fit_params)
    lblconcs = [concfromstr(name, "pM", ' ') for name in nicenames]
    valid = findall(x -> x !== nothing, lblconcs)
    if length(valid) != length(lblconcs)
        error("Some dataset names do not contain a parsable concentration value")
    end

    n_dsets = length(lblconcs)
    fitted_ks = [fit_params[1, i] for i in 1:n_dsets]
    grouped_indices, line_colors = make_paired_colors(nicenames)

    p = pl.plot(size=(900, 600), left_margin=3mm, right_margin=3mm,
                bottom_margin=3mm, xlabel="Concentration (pM)",
                ylabel=raw"Fitted rate $k_\mathrm{obs} (s^{-1})$",
                xscale=:log10, yscale=:log10, labelfont=pl.font(14),
                tickfont=pl.font(13), legendfontsize=12)

    for i in 1:n_dsets
        color_i = line_colors[i]
        pl.scatter!(p, [lblconcs[i]], [fitted_ks[i]], color=line_colors[i],
                    marker=:circle, markersize=6, label=nicenames[i], alpha=0.9)
    end

    k0 = 0.4
    c_line = collect(LinRange(minimum(lblconcs) * 0.8, maximum(lblconcs) * 1.2, 100))
    k_line = k0 .* c_line
    pl.plot!(p, c_line, k_line, linestyle=:dash, color=:black,
             linewidth=2.5, label=raw"Diffusion-limited rate")

    return p
end



### MAIN ANALYSIS FUNCTION ###

function fit_ml_all_datasets(; 
    datadir="data/JR064_tracks",
    plotdir="figures/ml_fits_kobs_10pMnorm/",
    fracabove=0.8,
    cdfthresh=0.75)
    
    # Create output directory if it doesn't exist
    if !isdir(plotdir)
        mkpath(plotdir)
    end

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

    # Step 2: Fit the exponential model to each dataset separately
    println("\n" * "-"^70)
    println("Fitting exponential model for each dataset separately...")
    println("-"^70)
    fitres = fitmodel(tactiv_alldsets, nicenames, durations, cdfthresh)
    fit_params, fit_errors, rms_errors, overall_rms, prep_cdf_res = fitres

    println("\nExponential fit results:")
    for i in 1:n_dsets
        println(@sprintf("  Dataset %d (%s): k = %.4f s^-1, t0 = %.4f s, RMS = %.3e", 
                         i, nicenames[i], fit_params[1,i], fit_params[2,i], rms_errors[i]))
    end
    println(@sprintf("\nOverall RMS = %.3e", overall_rms))

    # Select one dataset per concentration for plotting
    #=
    lblconcs = [concfromstr(name, "pM", ' ') for name in nicenames]
    unique_concs = unique(sort(filter(x -> x !== nothing, lblconcs)))
    selected_indices = Int[]
    for conc in unique_concs
        indices_for_conc = findall(c -> c ≈ conc, lblconcs)
        cells_for_conc = [length(tactiv_alldsets[i][1, :]) for i in indices_for_conc]
        max_idx = indices_for_conc[argmax(cells_for_conc)]
        push!(selected_indices, max_idx)
    end
    =#
    # Manual selection, to show we can get different appearances
    selected_indices = Int[7, 4, 3, 1]
    sind = join((string(a) for a in selected_indices), "")
    selected_tactiv = tactiv_alldsets[selected_indices]
    selected_durations = durations[selected_indices]
    selected_names = nicenames[selected_indices]
    selected_fit_params = fit_params[:, selected_indices]
    selected_fit_errors = fit_errors[:, selected_indices]

    println("\nSelected $(length(selected_indices)) datasets for plotting:")
    for (j, name) in enumerate(selected_names)
        println("  [$j] $name")
    end

    # Step 3: Create comparison plots
    println("\n" * "-"^70)
    println("Generating plots...")
    println("-"^70)

    selected_cdf_prep = collect(a[selected_indices] for a in prep_cdf_res)

    p_all = plot_ml_cdf_comparison(selected_cdf_prep, selected_names, selected_fit_params, 
                        expocdf, overall_rms, selected_durations)
    plot_file = joinpath(plotdir, "exp_cdf_all_datasets_$(sind).pdf")
    pl.savefig(p_all, plot_file)
    println("  Saved: $plot_file")

    p_conc = plot_kobs_vs_concentration(selected_names, 
                selected_fit_params, selected_fit_errors, prep_cdf_res[2])
    plot_file = joinpath(plotdir, "exp_k_vs_concentration_$(sind).pdf")
    pl.savefig(p_conc, plot_file)
    println("  Saved: $plot_file")

    p_repl = plot_kobs_replicates_loglog(nicenames, fit_params)
    plot_file = joinpath(plotdir, "exp_kobs_replicates_loglog.pdf")
    pl.savefig(p_repl, plot_file)
    println("  Saved: $plot_file")

    println("\n" * "="^70)
    println("Analysis complete!")
    println("="^70 * "\n")

    return (fit_params, overall_rms, nicenames, tactiv_alldsets)
end


# Run the analysis
function (@main)(_)
    fit_ml_all_datasets(fracabove=0.8, cdfthresh=0.75)
    return 0
end
