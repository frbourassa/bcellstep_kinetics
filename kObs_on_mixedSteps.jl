#=
Fitting a k_obs on the curves of a model with 
Load parameters that were saved in a simple csv file. 
We ignore all t0s of the reference model: we are making
a conceptual point. However, we include t0 in the one-step 
model for k_obs, because it helps capture the most correct slope. 

If we find that the mixed steps model has a k_obs that effectively
scales as c^{1/2}, as the data does, then maybe it will be worth
fitting such a model on the data, 

François X. P. Bourassa, 2026
=#

using Plots: Plots as pl
using Measures: mm
using Optim: Optim, optimize, IPNewton
using DelimitedFiles: readdlm
using Printf: @sprintf


# One-step CDF curve
"""
For observed k fit, a simple 1-step (exponential) cdf. 
"""
function expocdf(t::Array, rate::Float64; t0::Float64=0.0)
    rtt0 = clamp!(rate .* (t .- t0), 0.0, Inf)
    cdf = 1.0 .- exp.(-rtt0)
    return cdf
end


# Mixed step model CDF curve
"""
Mixed 3 steps model CDF, to generate curves to fit.
Version with a trap in state 3 VLP bounds, which can't activate the B cell.
"""
function mixedsteps_cdf(t::Array, conc::Float64, 
        kappa::Float64, k1::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = clamp.(t .- t0, 0.0, Inf)
    cdf = 1.0 .- exp.(-kc .* tt0)
    cdf .+= (kc / (kc + k1)) * exp.(-(kc + k1) .* tt0)
    cdf .+= (kc*kc*k2/(k1*(kc + k1)*(k2-kc-k1))) .* exp.(-(kc + k1) .* tt0)
    cdf .-= (kc * k2 / (k1 * (k2 - kc))) .* exp.(-kc .* tt0)
    cdf .-= (kc*kc / ((kc - k2) * (kc + k1 - k2))) .* exp.(-k2 .* tt0)
    return cdf
end


"""
Mixed 3 steps model CDF, to generate curves to fit.
Version with a trap in state 3 VLP bounds, which can't activate the B cell.
"""
function mixedsteps_pdf(t::Array, conc::Float64, 
        kappa::Float64, k1::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = clamp.(t .- t0, 0.0, Inf)
    error("Not implemented and not needed")
    pdf = 0.0
    return pdf
end


# Version made by mistake, where there is a third bound state
# that cannot activate the B cell. 
"""
Mixed 3 steps model CDF, to generate curves to fit.
Version with a trap in state 3 VLP bounds, which can't activate the B cell.
"""
function mixedsteps_cdf_with_trap(t::Array, conc::Float64, 
        kappa::Float64, k1::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = clamp.(t .- t0, 0.0, Inf)
    cdf = k1 / (kc +k1) .- exp.(-kc .* tt0) 
    cdf .+= kc .* exp.(-(kc + k1) .* tt0) ./ (kc + k1)
    cdf .+= (kc * k2 / (kc + k1) / (kc + k2))
    cdf .-= (kc / k1) .* exp.(-kc .* tt0)
    cdf .+= (kc*kc * k2 / (k1 * (k1 + kc) * (k2 - k1))) .* exp.(-(kc+k1) .* tt0)
    cdf .-= (kc*kc / (kc + k2) * (k2 - k1)) .* exp.(-(kc+k2) .* tt0)
    return cdf
end


"""
Mixed 3 steps model CDF, to generate curves to fit.
Version with a trap in state 3 VLP bounds, which can't activate the B cell.
"""
function mixedsteps_pdf_with_trap(t::Array, conc::Float64, 
        kappa::Float64, k1::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = clamp.(t .- t0, 0.0, Inf)
    pdf = 1.0 .- exp.(-k1 .* tt0) .+ kc / k1
    pdf .-= (kc * k2 / (k1 * (k2 - k1))) .* exp.(-k1 .* tt0)
    pdf .+= (kc / (k2 - k1)) .* exp.(-k2 .* tt0)
    pdf = pdf .* (kc .* exp.(-kc .* tt0))
    return pdf
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
function plot_kobs_vs_concentration(k_obs, concs, k1, k2)
    # Create plot
    p = pl.plot(concs, k_obs, marker=:circle, label="k_obs",
        xlabel="Concentration (pM)", xscale=:log10, yscale=:log10,
        ylabel=raw"$k_\mathrm{obs}$ fitted on mixed steps model curve",
        legend=:bottomright, linewidth=3.5,
        left_margin=3mm, right_margin=3mm, bottom_margin=3mm
    )
    # Make a line for the diffusion limit, another for the k2 plateau
    difflim = 0.2 .* concs  # 0.2 pM^-1 s^-1 for a B cell of radius 5 um
    pl.plot!(concs, difflim, label="Diffusion-limited", 
        color=:black, linestyle=:dash, linewidth=2.5)
    pl.plot!(concs, fill(max(k1, k2), length(concs)), label=raw"$k_2$ plateau", 
        linestyle=:dash, color=:green, linewidth=2.5)
    return p
end


"""
Plot the CDFs of the mixed model for different concentrations
compared to the k_obs fits. 
"""
function plot_cdf_vs_kobs_fits(times, cdfpoints, concs, all_params)
    # Compute k_obs curves
    n_concs = length(concs)
    obs_curves = zeros((n_concs, length(times)))
    for (i, c) in enumerate(concs)
        pms = all_params[i, :]
        curve = expocdf(times, pms[1]; t0=pms[2])
        obs_curves[i, :] = curve[:]
    end

    # Plot curves
    # Use magma colormap
    cmap = pl.cgrad(:magma)
    p = pl.plot(
        xlabel="Time (s)", ylabel=raw"Fraction active cells (CDF)",
        legend=:bottomright, linewidth=3.5,
        left_margin=3mm, right_margin=3mm, bottom_margin=3mm
    )
    
    for (i, c) in enumerate(concs)
        color = cmap[i / n_concs]
        if i % 4 == 0
            conc_str = @sprintf "%.2f pM" c
            pl.plot!(p, times, cdfpoints[i, :], ls=:solid, color=color, label=conc_str)
        else
            pl.plot!(p, times, cdfpoints[i, :], ls=:solid, color=color, label="")
        end
        pl.plot!(p, times, obs_curves[i, :], ls=:dash, color=color, label="")
    end
    return p
end

""" 
Main function to run the analysis
"""
function (@main)(_)
    # Choose mixed-steps model parameters
    # Start from one-hit model, adjust k2
    fname = "allDsets_10pMnorm_params1.csv"
    onehit_pms = readdlm(joinpath("results", fname), ',', Float64, '\n')
    k1 = onehit_pms[2]
    k2 = 2.0 * k1
    # kappa, k1, k2, t0 (although we ignore t0)
    mixed_pms = [onehit_pms[1], k1, k2, onehit_pms[3]]

    # Compute curves for various concentrations, relative to 1 pM
    log10range = collect(-2.0:0.1:1.0)
    conc_choices = 10.0 .^ log10range
    n_curves = length(conc_choices)
    timepoints = collect(LinRange(0.0, 300.0, 600))
    curves = zeros((n_curves, length(timepoints)))
    for (i, c) in enumerate(conc_choices)
        curve = mixedsteps_cdf(timepoints, c, mixed_pms[1], 
                    mixed_pms[2], mixed_pms[3]; t0=0.0)
        curves[i, :] = curve[:]
    end
    println("Fitting of k_obs completed for all concentrations")

    # For each, fit k_obs using fitcurve
    all_kobs = zeros(n_curves)
    all_t0s = zeros(n_curves)
    for i in 1:1:n_curves
        pms = fitcurve(timepoints, curves[i, :], conc_choices[i])
        all_kobs[i] = pms[1]
        all_t0s[i] = pms[2]
    end

    # Plot k_obs as a function of concentration, log-log scale
    # Show there is a clear plateau -- hopefully... 
    p_conc = plot_kobs_vs_concentration(all_kobs, conc_choices, k1, k2)
    plotdir = joinpath("figures", "model_misc")
    plot_file = joinpath(plotdir, "k_obs_vs_concentration_on_mixedsteps_model.pdf")
    pl.savefig(p_conc, plot_file)
    println("Saved: $plot_file")

    # Plot fits vs mixed step model CDF curves.
    all_fit_params = hcat(all_kobs, all_t0s)
    p_conc = plot_cdf_vs_kobs_fits(timepoints, curves, conc_choices, all_fit_params)
    plotdir = joinpath("figures", "model_misc")
    plot_file = joinpath(plotdir, "cdf_vs_kobs_fits_mixedsteps_model.pdf")
    pl.savefig(p_conc, plot_file)
    println("Saved: $plot_file")

    return 0
end