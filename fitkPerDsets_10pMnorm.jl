#= 
Fit CDF activation models using Maximum Likelihood Estimation
across multiple datasets in the JR064 experiment. The cell-intrinsic
rate is fitted across datasets, but the hit rate is fitted
on each dataset, to account for replicate variability and non-trivial
scaling with concentration. 

Variant where we normalize CDFs based on the fraction of cells
that never activate in the 10 pM experiments. We assume that such a high
concentration, all cells that could have been activated will have been
by the end of the movie; tracks that do not must be from a distinct population
not expected to follow the model fitted. There always seems
to be a 15-20 % of tracks that are in that situation. 

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
using .tracksAnalysis: prepalldsets, prepcdf, concfromstr, cleannames

conc_scale = 1.0  # pM

### MODELS PDFs AND CDFs ###

"""
For observed k fit, showing there may be a cell activation-limited step
"""
function expocdf(t::Array, rate::Float64; t0::Float64=0.0)
    rtt0 = clamp!(rate .* (t .- t0), 0.0, Inf)
    cdf = 1.0 .- exp.(-rtt0)
    return cdf
end


"""
One-hit model with cell activation
Concentration is in pM, kappa is relative to that. 
"""
function onehit_pdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = clamp.(t .- t0, 0.0, Inf)
    factor = k2 * kc / (k2 - kc)
    fctpart = exp.(-kc .* tt0) .- exp.(-k2 .* tt0)
    pdf = factor .* fctpart
    return pdf
end


"""
CDF of the 1-hit model
"""
function onehit_cdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = clamp.(t .- t0, 0.0, Inf)
    kdiff = (k2 - kc)
    fctpart = kc .* exp.(-k2 .* tt0) .- k2 .* exp.(-kc .* tt0)
    cdf = 1.0 .+ fctpart ./ kdiff
    return cdf
end


"""
Complementary CDF of the 1-hit model
"""
function onehit_ccdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = clamp.(t .- t0, 0.0, Inf)
    kdiff = (k2 - kc)
    fctpart = k2 .* exp.(-kc .* tt0) .- kc .* exp.(-k2 .* tt0)
    #fctpart = exp.(-kc .* tt0) .* (k2 .- kc .* exp.(-kdiff .* tt0))
    ccdf = fctpart ./ kdiff
    return ccdf
end


"""
Two-hit model with cell activation step
"""
function twohit_pdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = clamp.(t .- t0, 0.0, Inf)
    kdiff = k2 - kc
    factor = k2 * kc*kc / kdiff
    fctpart1 = tt0 .* exp.(-kc .* tt0)
    fctpart2 = (exp.(-kc .* tt0) .- exp.(-k2 .* tt0)) ./ kdiff
    pdf = factor .* (fctpart1 .- fctpart2)
    return pdf
end


"""
CDF of the 2-hit model, coded in a way that prevents large
cancellation errors when tt0 is approximately 0. TODO
"""
function twohit_cdf_float(t::Float64, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    kdiff = k2 - kc
    tt0 = clamp.(t .- t0, 0.0, Inf)
    if max(kc, k2) * tt0 < 1e-3
        # Taylor approximation to prevent large cancellation errors
        # Leading term is in t^3, so error is (1e-2)^4 < 1e-8, good
        cdf = k2 * kc * kc / 6.0 * tt0^3
    else
        prefact = k2 * kc / kdiff
        fact1 = (k2 - 2.0 * kc) / kdiff / kc
        fact2 = kc / kdiff / k2
        tt0ekct0 = tt0 * exp(-kc * tt0)
        term1 = fact1 * (1.0 - exp(-kc * tt0))
        term2 = fact2 * (1.0 - exp(-k2 * tt0))
        cdf = prefact * (term1 - tt0ekct0 + term2)
    end
    cdf = clamp(cdf, 1e-12, 1.0 - 1e-12)
    return cdf
end

function twohit_cdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    return collect(map(
        x -> twohit_cdf_float(x, conc, kappa, k2; t0=t0), t))
end


"""
Complementary CDF for the 2-hit model, 
coded in a way that prevents 1-1 large cancellation errors. 
"""
function twohit_ccdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    kdiff = k2 - kc
    tt0 = clamp.(t .- t0, 0.0, Inf)
    prefact = k2 * kc / kdiff
    fact23 = kc / kdiff
    ekct0 = exp.(-kc .* tt0)
    term1 = ekct0 .* (1.0 .+ kc .* tt0) ./ kc
    term2 = exp.(-k2 .* tt0) ./ k2  .- ekct0 ./ kc  
    ccdf = prefact .* (term1 .+ fact23 .* term2)
    return ccdf
end


### LOG-LIKELIHOOD FUNCTIONS AND DERIVATIVES ###

"""
log-likelihood for 1 dataset for the 1-hit model
"""
function onehit_loglhood_1dset(p, times, conc, ncell, klim0, klim1; te=nothing)
    kappa, k2, t0 = p
    if te === nothing
        te = times[min(klim1, length(times))]
    end
    if (kappa < 0.0) || (k2 < 0.0) || (times[1] < t0) 
        ll = -1e8
    else
        # Note that ncell includes those never activated, 
        # whereas times only includes cells eventually activated
        nc0 = klim0 - 1  # number of cells with t_activ = 0
        nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
        nc2 = ncell - nc0 - nc1
        lowerlim = max(1, klim0-1)
        if nc0 > 0
            ll = nc0 * log(onehit_cdf(times[lowerlim:lowerlim], conc, kappa, k2; t0=t0)[1])
        else
            ll = 0.0
        end
        ll += sum(log.(onehit_pdf(times[klim0:klim1], conc, kappa, k2; t0=t0)))
        if nc2 > 0
            ll += nc2 * log(onehit_ccdf([te], conc, kappa, k2; t0=t0)[1])
        end
    end
    return ll
end


"""
Log-likelihood across datasets for the 1-hit model
"""
function onehit_loglhood_total(p_all, times_all, concs, ncells, 
            klim0s, klim1s; te_all=nothing)
    # From the lists of times, klim0s, klim1s, ncells
    # compute likelihood contribution from each dataset, 
    # with one t0 per dataset, and combine at the end. 
    n_dsets = length(concs)
    ll_total = 0.0
    for n in 1:1:n_dsets
        kappa = p_all[1 + n]
        t0 = p_all[1 + n_dsets + n]
        p = [kappa, p_all[1], t0]  # kappa, k2, t0 for dataset n
        ll = onehit_loglhood_1dset(p, times_all[n], concs[n], ncells[n], 
                        klim0s[n], klim1s[n]; te=te_all[n])
        ll_total += ll
    end
    return ll_total
end


"""
log-likelihood for 1 dataset for the 1-hit model
"""
function twohit_loglhood_1dset(p, times, conc, ncell, klim0, klim1; te=nothing)
    kappa, k2, t0 = p
    if te === nothing
        te = times[min(klim1, length(times))]
    end
    if (kappa < 0.0) || (k2 < 0.0) || (times[1] < t0)
        ll = -1e8
    else
        # Note that ncell includes those never activated, 
        # whereas times only includes cells eventually activated
        nc0 = klim0 - 1  # number of cells with t_activ = 0
        nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
        nc2 = (ncell - nc0 - nc1)
        lowerlim = max(1, klim0-1)
        if nc0 > 0
            ll = nc0 * log(twohit_cdf(times[lowerlim:lowerlim], conc, kappa, k2; t0=t0)[1])
        else
            ll = 0.0
        end
        # Two-hit pdf also has large cancellation issues
        # This model is a numerical nightmare. 
        ll += sum(log.(twohit_pdf(times[klim0:klim1], conc, kappa, k2; t0=t0)))
        if nc2 > 0
            ll += nc2 * log(twohit_ccdf([te], conc, kappa, k2; t0=t0)[1])
        end
    end
    return ll
end


"""
Log-likelihood across datasets for the 2-hit model
"""
function twohit_loglhood_total(p_all, times_all, concs, ncells, 
            klim0s, klim1s; te_all=nothing)
    # From the lists of times, klim0s, klim1s, ncells
    # compute likelihood contribution from each dataset, 
    # with one t0 per dataset, and combine at the end. 
    n_dsets = length(concs)
    ll_total = 0.0
    for n in 1:1:n_dsets
        kappa = p_all[1 + n]
        t0 = p_all[1 + n_dsets + n]
        p = [kappa, p_all[1], t0]  # kappa, k2, t0 for dataset n
        ll = twohit_loglhood_1dset(p, times_all[n], concs[n], ncells[n], 
                        klim0s[n], klim1s[n]; te=te_all[n])
        ll_total += ll
    end
    return ll_total
end


# Three-hit model, for complete comparison

""" CDF """
function threehit_cdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = clamp.(t .- t0, 0.0, Inf)
    exptt0 = exp.(-kc .* tt0)
    cdf = ones(size(tt0))
    cdf .+= (k2 * (3.0*kc*kc - 3.0*kc*k2 + k2*k2) / (kc - k2)^3) .* exptt0
    cdf .+= (kc / (k2 - kc))^3 .* exp.(-k2 .* tt0)
    cdf .+= ((kc * k2 * (2.0*kc - k2))/(kc - k2)^2) .* tt0 .* exptt0
    cdf .+= ((kc*kc * k2)/(2.0*kc - 2.0*k2)) .* (tt0 .* tt0) .* exptt0
    return cdf
end


""" CCDF """
function threehit_ccdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    cdf = threehit_cdf(t, conc, kappa, k2; t0=t0)
    return 1.0 .- cdf
end


""" PDF """ 
function threehit_pdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = clamp.(t .- t0, 0.0, Inf)
    exptt0 = exp.(-kc .* tt0)
    pdf = (0.5 / (k2 - kc)) .* tt0.*tt0 .* exptt0
    pdf .-= (tt0 .* exptt0) ./ (kc - k2)^2 
    pdf .-= exptt0 ./ (kc - k2)^3 
    pdf .+= exp.(-k2 .* tt0) ./ (kc - k2)^3
    pdf .*= (kc^3 * k2)
    return pdf
end


"""
log-likelihood for 1 dataset for the 3-hit model
"""
function threehit_loglhood_1dset(p, times, conc, ncell, klim0, klim1; te=nothing)
    kappa, k2, t0 = p
    if te === nothing
        te = times[min(klim1, length(times))]
    end
    if (kappa < 0.0) || (k2 < 0.0) || (times[1] < t0)
        ll = -1e8
    else
        # Note that ncell includes those never activated, 
        # whereas times only includes cells eventually activated
        nc0 = klim0 - 1  # number of cells with t_activ = 0
        nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
        nc2 = (ncell - nc0 - nc1)
        lowerlim = max(1, klim0-1)
        if nc0 > 0
            cdf0 = max(threehit_cdf(times[lowerlim:lowerlim], conc, kappa, k2; t0=t0)[1], 1e-12)
            ll = nc0 * log(cdf0)
        else
            ll = 0.0
        end
        # Three-hit model can be numerically unstable; clamp small probabilities.
        pdfvals = threehit_pdf(times[klim0:klim1], conc, kappa, k2; t0=t0)
        pdfvals = clamp.(pdfvals, 1e-14, Inf)
        ll += sum(log.(pdfvals))
        if nc2 > 0
            ccdf_val = max(threehit_ccdf([te], conc, kappa, k2; t0=t0)[1], 1e-12)
            ll += nc2 * log(ccdf_val)
        end
    end
    return ll
end


"""
Log-likelihood across datasets for the 3-hit model
"""
function threehit_loglhood_total(p_all, times_all, concs, ncells, 
            klim0s, klim1s; te_all=nothing)
    # From the lists of times, klim0s, klim1s, ncells
    # compute likelihood contribution from each dataset, 
    # with one t0 per dataset, and combine at the end. 
    n_dsets = length(concs)
    ll_total = 0.0
    for n in 1:1:n_dsets
        kappa = p_all[1 + n]
        t0 = p_all[1 + n_dsets + n]
        p = [kappa, p_all[1], t0]  # kappa, k2, t0 for dataset n
        ll = threehit_loglhood_1dset(p, times_all[n], concs[n], ncells[n], 
                        klim0s[n], klim1s[n]; te=te_all[n])
        ll_total += ll
    end
    return ll_total
end


### RMS ERROR COMPUTATION ###

"""
Compute the root-mean-squared error of the fitted cumulative distribution 
function across all datasets.

Arguments:
- p_all: fitted parameters array [kappa, k2, t0_1, t0_2, ..., t0_n]
  where kappa and k2 are shared across all datasets,
  and t0_n is the activation time offset for dataset n.
- tu_all: list of unique time arrays for each dataset
- cumuls_all: list of empirical cumulative distributions for each dataset
- ilim0s_all: list of lower indices (first point with t > 0) for each dataset
- ilim1s_all: list of upper indices (last point to include) for each dataset
- concs_all: list of concentrations for each dataset
- cdffct: the CDF function to use (e.g., onehit_cdf, twohit_cdf)

Returns:
- rmserr: the root-mean-squared error computed over all datasets combined
"""
function compute_rms_all_dsets(p_all, tu_all, cumuls_all, ilim0s_all, 
                                ilim1s_all, concs_all, cdffct)
    n_dsets = length(tu_all)
    sres_total = 0.0
    npts_total = 0
    
    k2 = p_all[1]
    
    for n in 1:n_dsets
        kappa = p_all[1 + n]
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
        cdfestim = cdffct(tu[rms_i:ilim1], conc, kappa, k2; t0=t0)
        
        # Accumulate squared residuals
        sres_total += sum((cumul[rms_i:ilim1] .- cdfestim).^2)
        npts_total += (ilim1 - rms_i + 1)
    end
    
    # Compute overall RMS error
    rmserr = sqrt(sres_total / npts_total)
    return rmserr
end


### MAIN FITTING FUNCTIONS ###

"""
Based on the highest (10 pM) concentration,
where all cells are assumed to activate, find the fraction of tracks
which never activate and effectively come from a distinct sub-population. 
"""
function normalize_topconc(concs_all, ncell_all, cumuls_all)
    # Find dataset(s) with the highest concentration
    maxconc = maximum(concs_all)
    where_max = findall(x -> (x == maxconc), concs_all)

    # Find the fraction of cells that never activate in them 
    # (highest value in their cumulative fractions)
    frac_when_maxconc = mean(collect(x[end] for x in cumuls_all[where_max]))

    # For each dataset, rescale cumuls and ncell by removing
    # the fraction of cells expected to never activate, or
    # the actual fraction that did not activate, whichever is smallest. 
    ncell_fix = copy(ncell_all)
    cumuls_fix = copy(cumuls_all)
    for i in 1:1:length(concs_all)
        maxfrac_i = max(cumuls_all[i][end], frac_when_maxconc)
        cumuls_fix[i] = cumuls_all[i] ./ maxfrac_i
        ncell_fix[i] = ceil(Int, ncell_all[i] * maxfrac_i)
    end
    return ncell_fix, cumuls_fix
end


"""
Reject inactive tracks (those with missing activation times) based on the number of cells to keep.
Returns a filtered copy of the DataFrame, dropping columns with missing values up to the number
of tracks that need to be dropped (total_tracks - ncells_to_keep).
"""
function reject_inactive_tracks(tactiv, ncells_to_keep)
    # Find all columns that contain any missing values (inactive tracks)
    colnames = DF.names(tactiv)
    cols_with_missing = [col for col in colnames if any(ismissing, tactiv[:, col])]
    
    # Calculate how many tracks need to be dropped
    ncell_drop = size(tactiv, 2) - ncells_to_keep
    
    # Drop the first ncell_drop columns with missing values
    cols_to_drop = cols_with_missing[1:min(ncell_drop, length(cols_with_missing))]
    
    # Create a filtered copy without in-place modification
    tactiv_filtered = DF.select(tactiv, DF.Not(cols_to_drop))
    return tactiv_filtered
end


"""
Prepare all cdfs with normalization, numbers of cells, limit on t0, cutoff time,
and so on from the list of activation time DataFrames.  
"""
function prepare_all_cdfs(tactiv_alldsets, dsetnames, durations, cdfthresh)
    n_dsets = length(tactiv_alldsets)
    concs_all = []
    ncell_all = []
    cumuls_all = []
    for i in 1:1:n_dsets
        conc = concfromstr(dsetnames[i], "pM", ' ') / conc_scale
        # Prepare a unique list of cell number increase times to fit.
        tactiv = tactiv_alldsets[i]
        tu, cumul, ilim1, ncells, tlim = prepcdf(tactiv, cdfthresh, durations[i])
        push!(cumuls_all, cumul)
        push!(concs_all, conc)
        push!(ncell_all, ncells)
    end

    # Adjust cell numbers based on the max conc. conditions
    ncell_all, cumuls_all = normalize_topconc(concs_all, ncell_all, cumuls_all)

    # Analyze again now that we have determined the number of tracks to reject in each
    # dataset; this is necessary to stop at the correct cdfthresh. 
    k0_all = []
    k1_all = []
    te_all = []
    tu_all = []
    cumuls_all = []  # Recompute. 
    for i in 1:1:n_dsets
        ncells = ncell_all[i]  # Corrected cell number
        # Prepare a unique list of cell number increase times to fit.
        tactiv = tactiv_alldsets[i]
        tactiv_filtered = reject_inactive_tracks(tactiv, ncells)
        tu, cumul, ilim1, ncells_2, tlim = prepcdf(tactiv_filtered, cdfthresh, durations[i])
        if ncells_2 !== ncells
            println("Warning: after normalization, number of cells to fit changed for dataset ", dsetnames[i])
            println("Before: ncells = ", ncells, " , after: ncells = ", ncells_2)
            error("See above")
        end
        # Find first cell with t > 0: use CDF term instead of pdf in log-likelihood
        # for cells before that, whose activation time could be before t = 0. 
        ilim0 = findfirst(tu .> 0.0)
        push!(tu_all, tu)
        push!(cumuls_all, cumul)
        push!(k0_all, ilim0)
        push!(k1_all, ilim1)
        push!(te_all, tlim)
    end

    return concs_all, ncell_all, k0_all, k1_all, te_all, tu_all, cumuls_all
end


function fitmodel(tactiv_alldsets, dsetnames, durations, cdfthresh, loglfct, cdffct)
    # kappa estimate: diffusion-limited, 0.4 pM^-1 s^-1, 
    # times default concentration scale of 1 pM. 
    # k2 estimate: fast to begin with
    lower = [1e-4]  # k2
    upper = [20.0]
    init_p = [0.1]
    
    # For each dataset in the list of datasets, prepare the cdf, 
    # numbers of cells, concentrations, and limits for the fit.
    res = prepare_all_cdfs(tactiv_alldsets, dsetnames, durations, cdfthresh)
    concs_all, ncell_all, k0_all, k1_all, te_all, tu_all, cumuls_all = res

    # Limits for parameter kappa of each dataset
    n_dsets = length(tactiv_alldsets)
    for i in 1:1:n_dsets
        push!(lower, 1e-4 / conc_scale)  # kappa lower bound, scaled by conc scale
        push!(upper, 20.0 / conc_scale)  # kappa upper bound, scaled by conc scale
        push!(init_p, 0.4 / conc_scale)  # kappa initial guess, scaled by conc scale
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
    kappas = fit_params[2:1+n_dsets]
    t0s = fit_params[2+n_dsets:end]
    flag = Optim.converged(results)
    if !flag
        println("Did not converge for cdfhresh = ", cdfthresh)
        println("k1_all = ", k1_all, " , ncells = ", ncell_all, " , init_p = ", init_p)
    end

    # RMS error on the cdf across all datasets
    rmserr = compute_rms_all_dsets(fit_params, tu_all, cumuls_all, 
                                    k0_all, k1_all, concs_all, cdffct)
    
    # Use minus log-likelihood instead, should be minimized
    #rmserr = cost([slope, t0])

    #=
    # Compute model evidence p(D) = \int p(D|\theta)p(\theta) d\theta
    # by numerical integration of the likelihood over the flat prior
    # in the parameter boundaries we chose for \theta = (k, t0).
    lhood = theta -> exp(loglhood_linear(theta, tu, ilim1, ncells, ilim0))
    # The return value of hcubature is a tuple (I, E) 
    # of the estimated integral I and an estimated error E
    evid, evid_err = hcubature(lhood, lower, upper)
    rmserr = 1.0/evid  # inverse evidence should be minimized
    =#
    result = Array{Float64}(vcat(fit_params, [rmserr]))

    return result
end




"""
Plot a single dataset's empirical CDF with both model fits overlaid.

Arguments:
- tactiv: activation times for this dataset
- name: dataset name for title
- duration: length of imaging experiment
- cdfthresh: CDF threshold used for fitting
- kappa1, k2_1, t0_1: parameters for one-hit model
- kappa2, k2_2, t0_2: parameters for two-hit model
- kappa3, k2_3, t0_3: parameters for three-hit model
- conc: concentration for this dataset
- cdffct1, cdffct2, cdffct3: CDF functions for the three models
"""
function plot_single_dataset_cdf(cdf_prep_results_one, name, duration, cdfthresh,
                                  kappa1, k2_1, t0_1,
                                  kappa2, k2_2, t0_2,
                                  kappa3, k2_3, t0_3,
                                  cdffct1, cdffct2, cdffct3)
    # Print info to trace back issues
    println("For dataset:", name)
    println("Duration:", duration)
    println("cdfthresh:", cdfthresh)
    println("kappa1:", kappa1)
    println("k2_1:", k2_1)
    println("t0_1:", t0_1)
    println("kappa2:", kappa2)
    println("k2_2:", k2_2)
    println("t0_2:", t0_2)
    println("kappa3:", kappa3)
    println("k2_3:", k2_3)
    println("t0_3:", t0_3)

    # Prepare empirical CDF
    conc, ncells, ilim0, ilim1, tlim, tu, cumul = cdf_prep_results_one
    println("Number of activ times:", length(tu))
    println("ncells:", ncells)
    println("tlim:", tlim)
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
    trange = collect(LinRange(0, tu[end], 150))
    
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

    # Three-hit model
    cdf3 = cdffct3(trange, conc, kappa3, k2_3; t0=t0_3)
    y3 = cdf3 .* ncells
    pl.plot!(p, trange, y3, label="3-hit model", 
             color=:purple, linewidth=2, linestyle=:dot)
    
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
- threehit_params: [kappa, k2, t0_1, t0_2, ...]
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
        k2_1, kappa_1, t0_1 = onehit_params[1], onehit_params[1+i], onehit_params[1+n_dsets+i]
        k2_2, kappa_2, t0_2 = twohit_params[1], twohit_params[1+i], twohit_params[1+n_dsets+i]
        k2_3, kappa_3, t0_3 = threehit_params[1], threehit_params[1+i], threehit_params[1+n_dsets+i]
        
        # Generate smooth curves
        trange = collect(LinRange(0, tu[end], 100))
        
        cdf1 = onehit_cdf(trange, conc, kappa_1, k2_1; t0=t0_1)
        y1 = cdf1 .* ncells
        
        cdf2 = twohit_cdf(trange, conc, kappa_2, k2_2; t0=t0_2)
        y2 = cdf2 .* ncells
        
        cdf3 = threehit_cdf(trange, conc, kappa_3, k2_3; t0=t0_3)
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


"""
Plot model curves across concentrations for 1-hit, 2-hit, and 3-hit models.
Selects the dataset with the most cells for each concentration and plots
its model curves. Uses magma colormap for concentration-dependent coloring.
"""
function plot_models_vs_concentration(nicenames, ncell_all, onehit_params, twohit_params, threehit_params,
                                       onehit_cdf, twohit_cdf, threehit_cdf, durations)
    
    # Extract concentrations and group by concentration
    nicenames, lblconcs = cleannames(nicenames)
    unique_concs = unique(sort(lblconcs))
    
    # For each unique concentration, find dataset with most cells
    selected_indices = []
    for conc in unique_concs
        indices_for_conc = findall(c -> c ≈ conc, lblconcs)
        cells_for_conc = [ncell_all[i] for i in indices_for_conc]
        max_idx_in_group = argmax(cells_for_conc)
        push!(selected_indices, indices_for_conc[max_idx_in_group])
    end
    
    # Create plot
    p = pl.plot(size=(800, 600), xlabel="Time (s)", ylabel="Fraction of activated cells",
                legend=:bottomright, left_margin=3mm, right_margin=3mm, bottom_margin=3mm, 
                ylim=[-0.02, 1.05])
    
    # Get time range for plotting
    max_duration = maximum(durations)
    trange = collect(LinRange(0, max_duration, 200))
    
    # Use magma colormap
    cmap = pl.cgrad(:magma)
    
    # Plot curves for each concentration
    for (idx_in_unique, dset_idx) in enumerate(selected_indices)
        # Normalize color to [0,1] range
        color_val = (idx_in_unique - 1) / max(1, length(unique_concs))
        color = cmap[color_val]
        
        conc = lblconcs[dset_idx]
        kappa_1 = onehit_params[1 + dset_idx]
        k2_1 = onehit_params[1]
        t0_1 = 0.0
        
        kappa_2 = twohit_params[1 + dset_idx]
        k2_2 = twohit_params[1]
        t0_2 = 0.0
        
        kappa_3 = threehit_params[1 + dset_idx]
        k2_3 = threehit_params[1]
        t0_3 = 0.0
        
        # Compute CDFs
        cdf1 = onehit_cdf(trange, conc, kappa_1, k2_1; t0=t0_1)
        cdf2 = twohit_cdf(trange, conc, kappa_2, k2_2; t0=t0_2)
        cdf3 = threehit_cdf(trange, conc, kappa_3, k2_3; t0=t0_3)
        
        # Plot 1-hit model
        label_1hit = @sprintf "1-hit: %.2g pM" conc
        pl.plot!(p, trange, cdf1, label=label_1hit, linestyle=:solid,
                linewidth=2.5, color=color, alpha=1.0)
        
        # Plot 2-hit model
        label_2hit = @sprintf "2-hit: %.2g pM" conc
        pl.plot!(p, trange, cdf2, label=label_2hit, linestyle=:dash,
                linewidth=2.5, color=color, alpha=1.0)

        # Plot 3-hit model
        label_3hit = @sprintf "3-hit: %.2g pM" conc
        pl.plot!(p, trange, cdf3, label=label_3hit, linestyle=:dot,
                linewidth=2.5, color=:purple, alpha=1.0)
    end
    
    pl.title!(p, "Model predictions: 1-hit vs 2-hit (color = concentration)", fontsize=12)
    
    return p
end



### MAIN ANALYSIS FUNCTION ###

function fit_ml_all_datasets(; 
    datadir="data/JR064_tracks",
    plotdir="figures/ml_fits_kperdset_10pMnorm/",
    fracabove=0.6,
    cdfthresh=0.75)
    
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
    
    if isempty(fpaths)
        error("No approved CSV files found in $datadir")
    end

    # Sort names in decreasing concentration order
    # Extract concentration of each name
    shortnames, lblconcs = cleannames(fnames)
    sortidx = reverse(sortperm(lblconcs), dims=1)
    shortnames = shortnames[sortidx]
    fnames = fnames[sortidx]
    fpaths = fpaths[sortidx]
    
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
    onehit_k2 = onehit_params[1]
    onehit_kappas = onehit_result[2:1+n_dsets]
    onehit_t0s = onehit_result[2+n_dsets:end-1]
    onehit_rms = onehit_result[end]
    
    println("\nOne-hit model results:")
    println(@sprintf "  k2        = %.4f s^-1" onehit_k2)
    println("  t0 per dataset:")
    for (i, t0) in enumerate(onehit_t0s)
        println(@sprintf "    Dataset %d (%s):  κc_0 (kappa * 1 pM) = %.4f s^-1" i nicenames[i] onehit_kappas[i])
        println(@sprintf "    Dataset %d (%s): t0 = %.4f s" i nicenames[i] t0)
    end
    
    # Step 3: Fit the two-hit model
    println("\n" * "-"^70)
    println("Fitting TWO-HIT model (kappa, k2, one t0 per dataset)...")
    println("-"^70)
    twohit_result = fitmodel(tactiv_alldsets, nicenames, durations, cdfthresh,
                             twohit_loglhood_total, twohit_cdf)
    twohit_params = twohit_result[1:end-1]
    twohit_k2 = twohit_params[1]
    twohit_kappas = twohit_result[2:1+n_dsets]
    twohit_t0s = twohit_result[2+n_dsets:end-1]
    twohit_rms = twohit_result[end]
    
    println("\nTwo-hit model results:")
    println(@sprintf "  k2        = %.4f s^-1" twohit_k2)
    println("  t0 per dataset:")
    for (i, t0) in enumerate(twohit_t0s)
        println(@sprintf "    Dataset %d (%s):  κc_0 (kappa * 1 pM) = %.4f s^-1" i nicenames[i] twohit_kappas[i])
        println(@sprintf "    Dataset %d (%s): t0 = %.4f s" i nicenames[i] t0)
    end
    
    # Step 4: Fit the three-hit model
    println("\n" * "-"^70)
    println("Fitting THREE-HIT model (kappa, k2, one t0 per dataset)...")
    println("-"^70)
    threehit_result = fitmodel(tactiv_alldsets, nicenames, durations, cdfthresh,
                             threehit_loglhood_total, threehit_cdf)
    threehit_params = threehit_result[1:end-1]
    threehit_k2 = threehit_params[1]
    threehit_kappas = threehit_result[2:1+n_dsets]
    threehit_t0s = threehit_result[2+n_dsets:end-1]
    threehit_rms = threehit_result[end]
    
    println("\nThree-hit model results:")
    println(@sprintf "  k2        = %.4f s^-1" threehit_k2)
    println("  t0 per dataset:")
    for (i, t0) in enumerate(threehit_t0s)
        println(@sprintf "    Dataset %d (%s):  κc_0 (kappa * 1 pM) = %.4f s^-1" i nicenames[i] threehit_kappas[i])
        println(@sprintf "    Dataset %d (%s): t0 = %.4f s" i nicenames[i] t0)
    end

    # Step 5: Compute and compare RMS errors
    println("\n" * "="^70)
    println("RMS ERROR COMPARISON")
    println("="^70)
    
    println(@sprintf "\nOne-hit model:  RMS error = %.6e" onehit_rms)
    println(@sprintf "Two-hit model:  RMS error = %.6e" twohit_rms)
    println(@sprintf "Three-hit model:  RMS error = %.6e" threehit_rms)
    
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

    # Plot: Individual dataset panels
    for i in 1:n_dsets
        prep_cdf_i = [prep_cdf_res[j][i] for j in eachindex(prep_cdf_res)]
        k2_1, kappa_1, t0_1 = onehit_params[1], onehit_params[1+i], onehit_params[1+n_dsets+i]
        k2_2, kappa_2, t0_2 = twohit_params[1], twohit_params[1+i], twohit_params[1+n_dsets+i]
        k2_3, kappa_3, t0_3 = threehit_params[1], threehit_params[1+i], threehit_params[1+n_dsets+i]
        p_single = plot_single_dataset_cdf(
            prep_cdf_i, nicenames[i], durations[i], cdfthresh,
            kappa_1, k2_1, t0_1,
            kappa_2, k2_2, t0_2,
            kappa_3, k2_3, t0_3,
            onehit_cdf, twohit_cdf, threehit_cdf
        )
        plot_file = joinpath(plotdir, "ml_cdf_dataset_$(i)_$(nicenames[i]).pdf")
        pl.savefig(p_single, plot_file)
        println("  Saved: $plot_file")
    end
    
    # Plot: Model curves versus concentration
    ncell_all = prep_cdf_res[2]  # Extract ncell_all from the prepared results
    p_conc = plot_models_vs_concentration(fnames, ncell_all, onehit_params, twohit_params, threehit_params,
                                           onehit_cdf, twohit_cdf, threehit_cdf, durations)
    plot_file = joinpath(plotdir, "ml_models_vs_concentration.pdf")
    pl.savefig(p_conc, plot_file)
    println("  Saved: $plot_file")

    println("\n" * "="^70)
    println("Analysis complete!")
    println("="^70 * "\n")
    
    return (onehit_params, twohit_params, threehit_params, 
            onehit_rms, twohit_rms, threehit_rms,
            nicenames, tactiv_alldsets)
end


# Run the analysis
function (@main)(_)
    fit_ml_all_datasets(fracabove=0.8, cdfthresh=0.7)
    return 0
end
