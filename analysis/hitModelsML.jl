#= 
Distribution functions for the one-hit, two-hit, and three-hit model, 
as well as log-likelihood functions for each of them, 
and parameter optimization procedures. 

François X. P. Bourassa, 2026
=#

using Optim: Optim, optimize, IPNewton


# Data processing
module tracksAnalysis
    include("tracks.jl")
end
using .tracksAnalysis: prepare_all_cdfs

### ONE-HIT MODEL ###

"""
One-hit model with cell activation
Concentration is in pM, kc is relative to that. 
"""
function onehit_pdf(t::Array, kc::Float64, k2::Float64; t0::Float64=0.0)
    tt0 = clamp.(t .- t0, 0.0, Inf)
    factor = k2 * kc / (k2 - kc)
    fctpart = exp.(-kc .* tt0) .- exp.(-k2 .* tt0)
    pdf = factor .* fctpart
    return pdf
end


"""
CDF of the 1-hit model
"""
function onehit_cdf(t::Array, kc::Float64, k2::Float64; t0::Float64=0.0)
    tt0 = clamp.(t .- t0, 0.0, Inf)
    kdiff = (k2 - kc)
    fctpart = kc .* exp.(-k2 .* tt0) .- k2 .* exp.(-kc .* tt0)
    cdf = 1.0 .+ fctpart ./ kdiff
    return cdf
end


"""
Complementary CDF of the 1-hit model
"""
function onehit_ccdf(t::Array, kc::Float64, k2::Float64; t0::Float64=0.0)
    tt0 = clamp.(t .- t0, 0.0, Inf)
    kdiff = (k2 - kc)
    fctpart = k2 .* exp.(-kc .* tt0) .- kc .* exp.(-k2 .* tt0)
    #fctpart = exp.(-kc .* tt0) .* (k2 .- kc .* exp.(-kdiff .* tt0))
    ccdf = fctpart ./ kdiff
    return ccdf
end



# Log-likelihood functions

"""
log-likelihood for 1 dataset for the 1-hit model
"""
function onehit_loglhood_1dset(p, times, ncell, klim0, klim1; te=nothing)
    kc, k2, t0 = p
    if te === nothing
        te = times[min(klim1, length(times))]
    end
    if (kc < 0.0) || (k2 < 0.0) || (times[1] < t0) 
        ll = -1e8
    else
        # Note that ncell includes those never activated, 
        # whereas times only includes cells eventually activated
        nc0 = klim0 - 1  # number of cells with t_activ = 0
        nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
        nc2 = ncell - nc0 - nc1
        lowerlim = max(1, klim0-1)
        if nc0 > 0
            ll = nc0 * log(onehit_cdf(times[lowerlim:lowerlim], kc, k2; t0=t0)[1])
        else
            ll = 0.0
        end
        ll += sum(log.(onehit_pdf(times[klim0:klim1], kc, k2; t0=t0)))
        if nc2 > 0
            ll += nc2 * log(onehit_ccdf([te], kc, k2; t0=t0)[1])
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
        kc = p_all[1 + n]
        t0 = p_all[1 + n_dsets + n]
        p = [kc, p_all[1], t0]  # kc, k2, t0 for dataset n
        ll = onehit_loglhood_1dset(p, times_all[n], ncells[n], 
                        klim0s[n], klim1s[n]; te=te_all[n])
        ll_total += ll
    end
    return ll_total
end



"""
Two-hit model with cell activation step
"""
function twohit_pdf(t::Array, kc::Float64, k2::Float64; t0::Float64=0.0)
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
function twohit_cdf_float(t::Float64, kc::Float64, k2::Float64; t0::Float64=0.0)
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

function twohit_cdf(t::Array, kc::Float64, k2::Float64; t0::Float64=0.0)
    return collect(map(
        x -> twohit_cdf_float(x, kc, k2; t0=t0), t))
end


"""
Complementary CDF for the 2-hit model, 
coded in a way that prevents 1-1 large cancellation errors. 
"""
function twohit_ccdf(t::Array, kc::Float64, k2::Float64; t0::Float64=0.0)
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


# Log-likelihood

"""
log-likelihood for 1 dataset for the 1-hit model
"""
function twohit_loglhood_1dset(p, times, ncell, klim0, klim1; te=nothing)
    kc, k2, t0 = p
    if te === nothing
        te = times[min(klim1, length(times))]
    end
    if (kc < 0.0) || (k2 < 0.0) || (times[1] < t0)
        ll = -1e8
    else
        # Note that ncell includes those never activated, 
        # whereas times only includes cells eventually activated
        nc0 = klim0 - 1  # number of cells with t_activ = 0
        nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
        nc2 = (ncell - nc0 - nc1)
        lowerlim = max(1, klim0-1)
        if nc0 > 0
            ll = nc0 * log(twohit_cdf(times[lowerlim:lowerlim], kc, k2; t0=t0)[1])
        else
            ll = 0.0
        end
        # Two-hit pdf also has large cancellation issues
        # This model is a numerical nightmare. 
        ll += sum(log.(twohit_pdf(times[klim0:klim1], kc, k2; t0=t0)))
        if nc2 > 0
            ll += nc2 * log(twohit_ccdf([te], kc, k2; t0=t0)[1])
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
        kc = p_all[1 + n]
        t0 = p_all[1 + n_dsets + n]
        p = [kc, p_all[1], t0]  # kc, k2, t0 for dataset n
        ll = twohit_loglhood_1dset(p, times_all[n], ncells[n], 
                        klim0s[n], klim1s[n]; te=te_all[n])
        ll_total += ll
    end
    return ll_total
end


### THREE-HIT MODEL ###

""" CDF """
function threehit_cdf(t::Array, 
        kc::Float64, k2::Float64; t0::Float64=0.0)
    tt0 = clamp.(t .- t0, 0.0, Inf)
    exptt0 = exp.(-kc .* tt0)
    kdif = k2 - kc
    cdf = -(k2 * (3.0*kc*kc - 3.0*kc*k2 + k2*k2) / kdif^3) .* exptt0
    cdf .+= ((kc * k2 * (2.0*kc - k2))/kdif^2) .* tt0 .* exptt0
    cdf .-= ((0.5 * kc*kc * k2) / kdif) .* (tt0 .* tt0) .* exptt0
    cdf .+= (kc / kdif)^3 .* exp.(-k2 .* tt0)
    cdf .+= 1.0
    cdf = clamp.(cdf, 1e-10, 1.0 - 1e-10)
    return cdf
end


""" CCDF """
function threehit_ccdf(t::Array, kc::Float64, k2::Float64; t0::Float64=0.0)
    tt0 = clamp.(t .- t0, 0.0, Inf)
    exptt0 = exp.(-kc .* tt0)
    kdif = k2 - kc
    ccdf = (k2 * (3.0*kc*kc - 3.0*kc*k2 + k2*k2) / kdif^3) .* exptt0
    ccdf .-= ((kc * k2 * (2.0*kc - k2))/kdif^2) .* tt0 .* exptt0
    ccdf .+= ((0.5 * kc*kc * k2) / kdif) .* (tt0 .* tt0) .* exptt0
    ccdf .-= (kc / kdif)^3 .* exp.(-k2 .* tt0)
    ccdf = clamp.(ccdf, 1e-10, 1.0 - 1e-10)
    return ccdf
end


""" PDF """ 
function threehit_pdf(t::Array, kc::Float64, k2::Float64; t0::Float64=0.0)
    tt0 = clamp.(t .- t0, 0.0, Inf)
    exptt0 = exp.(-kc .* tt0)
    kdif = k2 - kc
    pdf = (0.5 / kdif) .* tt0.*tt0 .* exptt0
    pdf .-= (tt0 .* exptt0) ./ kdif^2 
    pdf .+= (exptt0 .- exp.(-k2 .* tt0)) ./ kdif^3 
    pdf .*= (kc^3 * k2)
    return pdf
end


# Log-likelihood

"""
log-likelihood for 1 dataset for the 3-hit model
"""
function threehit_loglhood_1dset(p, times, ncell, klim0, klim1; te=nothing)
    kc, k2, t0 = p
    if te === nothing
        te = times[min(klim1, length(times))]
    end
    if (kc < 0.0) || (k2 < 0.0) || (times[1] < t0)
        ll = -1e8
    else
        # Note that ncell includes those never activated, 
        # whereas times only includes cells eventually activated
        nc0 = klim0 - 1  # number of cells with t_activ = 0
        nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
        nc2 = (ncell - nc0 - nc1)
        lowerlim = max(1, klim0-1)
        if nc0 > 0
            ll = nc0 * log(threehit_cdf(times[lowerlim:lowerlim], kc, k2; t0=t0)[1])
        else
            ll = 0.0
        end
        pdfvals = threehit_pdf(times[klim0:klim1], kc, k2; t0=t0)
        ll += sum(log.(pdfvals))
        if nc2 > 0
            ll += nc2 * log(threehit_ccdf([te], kc, k2; t0=t0)[1])
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
        kc = p_all[1 + n]
        t0 = p_all[1 + n_dsets + n]
        p = [kc, p_all[1], t0]  # kc, k2, t0 for dataset n
        ll = threehit_loglhood_1dset(p, times_all[n], ncells[n], 
                        klim0s[n], klim1s[n]; te=te_all[n])
        ll_total += ll
    end
    return ll_total
end




### MAIN FITTING FUNCTION ###

""" 
Fit a k-hit model across datasets: fit one global cell activation rate k2,
and one hit rate kc and start time t0 for each dataset separately. 
"""
function fithitmodel(tactiv_alldsets, dsetnames, durations, cdfthresh, loglfct, cdffct)
    # k2 lower, upper limits and initial guess
    lower = [1e-4]
    upper = [20.0] 
    init_p = [0.1]
    
    # For each dataset in the list of datasets, prepare the cdf, 
    # numbers of cells, concentrations, and limits for the fit.
    res = prepare_all_cdfs(tactiv_alldsets, dsetnames, durations, cdfthresh)
    concs_all, ncell_all, k0_all, k1_all, te_all, tu_all, cumuls_all = res

    # Limits for parameter kc of each dataset
    # kc estimate: diffusion-limited rate = 0.16 pM^-1 s^-1, times concentration
    n_dsets = length(tactiv_alldsets)
    for i in 1:1:n_dsets
        concs_all
        push!(lower, 1e-4 * concs_all[i])  # kc lower bound, times conc
        push!(upper, 20.0 * concs_all[i])  # kc upper bound, times conc
        push!(init_p, 0.4 * concs_all[i])  # kc initial guess
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