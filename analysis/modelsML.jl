#= 
Distribution functions for the one-hit, two-hit, and three-hit model.
Also, log-likelihood functions. 

François X. P. Bourassa, 2026
=#


### ONE-STEP MODEL (for k_obs) ###

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


### Log-likelihood

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

