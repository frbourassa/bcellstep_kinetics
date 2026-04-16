#=
Fitting a k_obs on the model curves from the fitted one-hit model. 
Load parameters that were saved in a simple csv file. 
We ignore all t0s of the fitted one-hit model: 
this is a purely conceptual point we are trying to make. 
However, we include t0 in the one-step model for k_obs, 
because  

François X. P. Bourassa, 2026
=#

using Plots: Plots as pl
using Measures: mm
using Optim: Optim, optimize, IPNewton
using DelimitedFiles: readdlm


# One-step CDF curve
"""
For observed k fit, a simple 1-step (exponential) cdf. 
"""
function expocdf(t::Array, rate::Float64; t0::Float64=0.0)
    rtt0 = clamp!(rate .* (t .- t0), 0.0, Inf)
    cdf = 1.0 .- exp.(-rtt0)
    return cdf
end


# One-hit model CDF curve
"""
One-hit model CDF, to generate curves to fit
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
Cost function: squared residuals
"""
function squared_resids(p, times, cdfpoints)
    # Compute model values
    modelpoints = expocdf(times, p[1]; t0=p[2])
    ressum = sum((modelpoints .- cdfpoints).^2)
    return ressum
end


"""
Curve fitting function
"""
function fitcurve(times, cdfpoints, conc)
    # Boundaries on k_obs and t0
    lower = [1e-6, -20.0]
    upper = [10.0, 5.0]  # not more than 5 s for t0
    init_p = [0.1, 0.0]
    # Minimize -log likelihood
    cost = p -> squared_resids(p, times, cdfpoints)
    inner_optimizer = IPNewton()
    
    results = optimize(cost, lower, upper, init_p, inner_optimizer)
    # Get the results
    fit_params = Optim.minimizer(results)
    flag = Optim.converged(results)
    if !flag
        println("Did not converge for conc = ", conc)
        println("init_p = ", init_p)
        println("lower: ", lower, ", upper: ", upper, ", init: ", init_p)
        println("Extrema of CDF curve:", extrema(cdfpoints))
        println("Extrema of time points:", extrema(times))
    end
    return fit_params
end


""" 
Plotting k_obs vs concentration
"""
function plot_kobs_vs_concentration(k_obs, concs, k2)
    # Create plot
    p = pl.plot(concs, k_obs, marker=:circle, label="k_obs",
        xlabel="Concentration (pM)", xscale=:log10, yscale=:log10,
        ylabel=raw"$k_\mathrm{obs}$ fitted on 1-hit model curve",
        legend=:bottomright, linewidth=3.5,
        left_margin=3mm, right_margin=3mm, bottom_margin=3mm
    )
    # Make a line for the diffusion limit, another for the k2 plateau
    difflim = 0.2 .* concs  # 0.2 pM^-1 s^-1 for a B cell of radius 5 um
    pl.plot!(concs, difflim, label="Diffusion-limited", 
        color=:black, linestyle=:dash, linewidth=2.5)
    pl.plot!(concs, fill(k2, length(concs)), label=raw"$k_2$ plateau", 
        linestyle=:dash, color=:green, linewidth=2.5)
    return p
end



""" 
Main function to run the analysis
"""
function (@main)(_)
    # Load one-hit model parameters
    fname = "allDsets_10pMnorm_params1.csv"
    onehit_pms = readdlm(joinpath("results", fname), ',', Float64, '\n')

    # Compute curves for various concentrations, relative to 1 pM
    log10range = collect(-2.0:0.1:1.0)
    conc_choices = 10.0 .^ log10range
    n_curves = length(conc_choices)
    timepoints = collect(LinRange(0.0, 300.0, 600))
    curves = zeros((n_curves, length(timepoints)))
    for (i, c) in enumerate(conc_choices)
        curve = onehit_cdf(timepoints, c, onehit_pms[1], onehit_pms[2]; t0=0.0)
        curves[i, :] = curve[:]
    end
    println("Fitting of k_obs completed for all concentrations")

    # For each, fit k_obs using fitcurve
    all_kobs = zeros(n_curves)
    for i in 1:1:n_curves
        pms = fitcurve(timepoints, curves[i, :], conc_choices[i])
        all_kobs[i] = pms[1]
    end

    # Plot k_obs as a function of concentration, log-log scale
    # Show there is a clear plateau -- hopefully... 
    k2 = onehit_pms[2]
    p_conc = plot_kobs_vs_concentration(all_kobs, conc_choices, k2)
    plotdir = joinpath("figures", "model_misc")
    plot_file = joinpath(plotdir, "k_obs_vs_concentration_on_1-hit_model.pdf")
    pl.savefig(p_conc, plot_file)
    println("Saved: $plot_file")

    return 0
end