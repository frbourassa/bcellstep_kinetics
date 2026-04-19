#= 
Distribution functions and maximum likelihood fitting for a 
one-step model to extract k_obs on each dataset. 

François X. P. Bourassa, 2026
=#


using Optim: Optim, optimize, IPNewton


# Data processing
module tracksAnalysis
    include("tracks.jl")
end
using .tracksAnalysis: prepare_all_cdfs


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


### Gradient and hessian of the log-likelihood, for covariance calculation
function deriv_minuslogl_expo!(dp, p, times, ncell, klim0, klim1; te=nothing)
    k, t0 = p
    nc0 = klim0 - 1  # number of cells with t_activ = 0
    nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
    nc2 = ncell - nc0 - nc1
    if te === nothing
        te = times[min(klim1, length(times))]
    end
    expkt0 = exp(k*t0)
    if nc0 > 0
        common = nc0 * expkt0 / (1.0 - expkt0)
    else
        common = 0.0 
    end
    dp[1] = -nc1 / k + nc2 * (te - t0)
    dp[1] += common * t0
    dp[2] = -(ncell - nc0) * k + common * k
    for nn in klim0:1:klim1
        dp[1] += (times[nn] - t0)
    end
    return dp
end


function hess_minuslogl_expo!(mat, p, times, ncell, klim0, klim1; te=nothing)
    k, t0 = p
    nc0 = klim0 - 1  # number of cells with t_activ = 0
    nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
    nc2 = ncell - nc0 - nc1
    if te === nothing
        te = times[min(klim1, length(times))]
    end
    expkt0 = exp(k*t0)
    if nc0 > 0
        common = nc0 * expkt0 / (1.0 - expkt0)^2
    else
        common = 0.0
    end
    mat[1, 1] = nc1 / k^2 + common * t0*t0
    mat[1, 2] = -(ncell - nc0) - common * (expkt0 - k*t0 - 1.0)
    mat[2, 1] = mat[1, 2]
    mat[2, 2] = common * k*k
    return mat
end



### FITTING FUNCTIONS ###

function fitkobs(tactiv_alldsets, dsetnames, durations, cdfthresh)
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

    return fit_params, fit_params_errors, prepres
end

