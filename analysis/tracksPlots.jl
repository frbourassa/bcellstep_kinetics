#=
Plotting functions for k_obs and hit models analyses. 

François X. P. Bourassa, 2026
=#

using DataFrames: DataFrames as DF  # all methods available as DF.method
using Plots: Plots as pl
using Measures: mm
using Printf: @sprintf
using Statistics: mean

green_color = :green


module tracksAnalysis
    include("tracks.jl")
end
using .tracksAnalysis: concfromstr, cleannames


module hitModels
    include("hitModelsML.jl")
end
using .hitModels: onehit_cdf, twohit_cdf, threehit_cdf


pl.default(fontfamily="Arial")


### CLEANUP FUNCTIONS ###

"""
Clip excessive, artifactual RMS errors for plotting on log scale
"""
function cliprms!(rms; log10bound=4)
    log_rms = log10.(rms)
    mean_log_rms = mean(log_rms)
    wh = findall(log_rms .<= (mean_log_rms - log10bound))
    rms[wh] .= 10.0^mean_log_rms
    return rms
end


### DATA PROCESSING PLOTS ###


"""
Calcium tracks and thresholding method illustrated on a chosen dataset.
"""
function plot_trajs_activ_cdf(dfi, activtimes, activfluo, thresh, backgnd,
    high_i; conclbl=nothing)
    # high_i: track to highlight
    idnames = filter(x -> tryparse(Int64, x) !== nothing, DF.names(dfi))
    nperplot = 8
    nplots = div(length(idnames), nperplot) + min(1, rem(length(idnames), nperplot))
    nplots = convert(Int64, nplots)
    # Create subplots layout, put legends to the right
    ppp = pl.plot(layout=(1, 1), size=(550, 450), left_margin=3mm, 
            right_margin=3mm, legend=nothing)
    p = ppp[1]
    for col in idnames
        if tryparse(Int64, col) !== nothing
            if col == high_i
                continue  # will plot separately and last
            end
            vals = DF.dropmissing(dfi, col)
            times = vals.TIME
            vals = vals[:, col]
            linecolor = :lightgrey
            lw = 1.0
            li = pl.plot!(p, times, vals, color=linecolor, alpha=0.5, lw=lw)
            if activtimes[1, col] !== missing
                pl.scatter!(p, [activtimes[1, col]], [activfluo[1, col]]; 
                color=linecolor, label=nothing, alpha=0.5, markersize=3.0)
            end
        end
    end
    # Plot the highlight and the threshold line
    vals = DF.dropmissing(dfi, high_i)
    col = high_i
    times = vals.TIME
    vals = vals[:, col]
    linecolor = green_color
    li = pl.plot!(p, times, vals, color=linecolor)
    if activtimes[1, col] !== missing
        pl.scatter!(p, [activtimes[1, col]], [activfluo[1, col]]; 
        color=linecolor, label=nothing, alpha=1, markersize=6.0)
    end
    pl.hline!(p, [thresh], linestyle=:dash, color=:black, label=nothing, lw=1.5)
    annot_clr = :black
    tbk = times[div(length(times), 6)]
    pl.annotate!(p, tbk*1.05,1.05*thresh, ("Threshold", :left, :bottom, 10, annot_clr))


    # Annotate background and threshold factor f
    pl.hline!(p, [backgnd], linestyle=:dash, color=annot_clr, label=nothing, lw=1.5)
    pl.annotate!(p, tbk*1.05, 0.955 * backgnd, ("Background", :left, :top, 10, annot_clr))
    pl.plot!(p, [tbk, tbk], [backgnd+60, thresh-thresh*0.05],arrow=true, color=annot_clr, linewidth=1.5)
    pl.annotate!(p, tbk*1.15, 0.5 * (backgnd + thresh), (raw"$\times f$", :left, 14, annot_clr))

    # Annotate active or not
    tn = times[2*div(length(times), 3)]
    pl.plot!(p, [tn, tn], [thresh+60.0, thresh+thresh*0.3],arrow=true,color=:black,linewidth=2)
    pl.annotate!(p, tn*1.05, thresh+0.3*thresh, ("Active", :left, :bottom, 10))
    pl.plot!(p, [tn, tn], [thresh-60.0, thresh-0.3*thresh],arrow=true,color=:black,linewidth=2,label="Inactive")
    pl.annotate!(p, tn*1.05, thresh-0.32*thresh, ("Inactive", :left, :bottom, 10))

    pl.xlabel!(p, "Time (s)")
    pl.ylabel!(p, "Calcium fluorescence (a.u.)")

    # Title the panel
    if conclbl !== nothing
        xmid = times[div(length(times), 2)]
        ytop = maximum(maximum.(skipmissing.(eachrow(dfi))))
        conclbl2 = replace(conclbl, "1pM 2" => "1pM")  # replicate 1 low cell count
        pl.annotate!(p, xmid, ytop, ("Example: " * conclbl2, 12))
    end

    return ppp
end


### K_OBS FITTING PLOTS ###

"""
Create a single-panel plot comparing the empirical normalized CDFs to
fitted exponential model curves for each dataset.

Arguments:
- cdf_prep_results: output from prepare_all_cdfs
- nicenames: list of dataset names
- fit_params: 2 x n_dsets array of [k; t0] per dataset
- cdffct: exponential CDF function to plot
- durations: list of experiment durations
"""
function plot_kobs_fits(cdf_prep_results, nicenames, 
            fit_params, cdffct, durations)
    # Unwrap data
    (_, _, _, _, _, tu_all, cumuls_all) = cdf_prep_results
    n_dsets = length(tu_all)

    lblconcs = [concfromstr(name, "pM", ' ') for name in nicenames]
    if any(x -> x === nothing, lblconcs)
        error("Some dataset names do not contain a parsable concentration value")
    end
    colors = pl.cgrad(:magma)

    tplotlim = [-5.0, maximum([maximum(a) for a in tu_all]) * 1.05]
    fracplotlim = [-0.05, 1.05]

    p = pl.plot(left_margin=3mm, right_margin=3mm,
                bottom_margin=3mm, xlabel="Time (s)", ylabel="Fraction of activated cells",
                xlim=tplotlim, ylim=fracplotlim, legend=:bottomright,
                tickfont=pl.font(13), legendfontsize=11,
                titlefontsize=12, legendmargin=4mm, foreground_color_legend=nothing,
                title=raw"1-step model for $k_\mathrm{obs}$ fits")

    # Plot all data points first
    markertypes = [:circle, :diamond, :square, :utriangle]
    for i in 1:n_dsets
        tu = tu_all[i]
        cumul = cumuls_all[i]

        goodt = vcat([0.0], tu)
        ncumul = vcat([0.0], cumul)

        k = fit_params[1, i]
        t0 = fit_params[2, i]
        trange = collect(LinRange(0, durations[i], 200))
        yfit = cdffct(trange, k; t0=t0)
        line_color = colors[(2*i-2)/(2*n_dsets-1)]
        conc_label = nicenames[i]

        pl.plot!(p, goodt, ncumul, linetype=:steppost, 
                 color=:grey, linewidth=3.0, alpha=0.8, label=nothing)
        pl.scatter!(p, tu, cumul, color=:grey, markersize=4, m=markertypes[i],
                    alpha=0.6, label="Data $conc_label")
        pl.plot!(p, trange, yfit, label=raw"Fit " * conc_label,
                 color=line_color, linewidth=4.4, linestyle=:solid)
    end

    return p
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
    fit_ks_errs = fit_errors[1, :]
    fitted_ks = fit_params[1, :]
    selected_concs = lblconcs
    n_dsets = length(selected_concs)
    mark_colors = pl.cgrad(:magma)[[(2*i-2)/(2*n_dsets-1) for i in 1:1:n_dsets]]
    p = pl.plot(selected_concs, fitted_ks, linewidth=2, color=:grey, label=nothing)
    pl.scatter!(p, selected_concs, fitted_ks, marker=:circle, markersize=7,
                color=mark_colors, xlabel="Particle concentration (pM)", 
                ylabel=raw"Effective rate $k_{\mathrm{obs}} (\mathrm{s^{-1}})$", 
                legend=false, left_margin=3mm, right_margin=3mm, bottom_margin=3mm)
    
    # Compare to diffusion-limited rate
    k0 = 0.16  # pM^-1 s^-1, for B cell radius 2.5 um
    c_line = collect(LinRange(minimum(lblconcs)-0.005, minimum(lblconcs) + 0.2, 100))
    k_line = k0 .* c_line
    pl.plot!(p, c_line, k_line, linestyle=:dot, color=:black,
             linewidth=2.0)
    
    # Annotate "Diffusion-limited"
    ytxt, xtxt = minimum(fitted_ks)+0.001, minimum(lblconcs)+1.5
    pl.plot!(p, [xtxt, xtxt-1.0], [ytxt, ytxt], 
        arrow=true, color=:black, linewidth=2, label=nothing)
    pl.annotate!(p, xtxt+0.2, ytxt, ("Diffusion-limited", :left, 12))

    return p
end



### HIT MODEL FITTING PLOTS ###

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
"""
function plot_ml_cdf_comparison(cdf_prep_results, nicenames, 
                                 onehit_params, twohit_params, threehit_params,
                                 onehit_cdf, twohit_cdf, threehit_cdf)
    # Unwrap data
    (_, ncell_all, _, _, _, tu_all, cumuls_all) = cdf_prep_results
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
        
        tu = tu_all[i]
        cumul = cumuls_all[i]
        ncells = ncell_all[i]
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
    
    return p
end


"""
Plot a single dataset's empirical CDF with both model fits overlaid.

Arguments:
- tactiv: activation times for this dataset
- name: dataset name for title
- duration: length of imaging experiment
- cdfthresh: CDF threshold used for fitting
- kc1, k2_1, t0_1: parameters for one-hit model
- kc2, k2_2, t0_2: parameters for two-hit model
- kc3, k2_3, t0_3: parameters for three-hit model
- conc: concentration for this dataset
- cdffct1, cdffct2, cdffct3: CDF functions for the three models
"""
function plot_single_dataset_cdf(cdf_prep_results_one, name, duration, 
            cdfthresh, kc1, k2_1, t0_1, kc2, k2_2, t0_2, kc3, k2_3, t0_3,
            rms1, rms2, rms3, cdffct1, cdffct2, cdffct3)
    # Print info to trace back issues
    println("For dataset:", name)
    println("Duration:", duration)
    println("cdfthresh:", cdfthresh)
    println("kc1:", kc1)
    println("k2_1:", k2_1)
    println("t0_1:", t0_1)
    println("kc2:", kc2)
    println("k2_2:", k2_2)
    println("t0_2:", t0_2)
    println("kc3:", kc3)
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
    ncumul = vcat([0], cumul)
    
    # Create plot
    p = pl.plot(size=(550, 450), legend=:topleft, 
                left_margin=3mm, right_margin=3mm, bottom_margin=3mm)
    
    # Plot empirical CDF as step function
    pl.plot!(p, goodt, ncumul, linetype=:steppost, 
             color=:grey, linewidth=2.5, label="Data $name", alpha=0.8)
    
    # Add markers at data points
    pl.scatter!(p, tu, cumul, color=:grey, 
                markersize=4, label=nothing, alpha=0.6)
    
    # Generate smooth curves for model fits over the fitting range
    trange = collect(LinRange(0, tu[end], 150))

    color1, color2, color3 = :blue, :red, :purple
    
    # One-hit model
    cdf1 = cdffct1(trange, kc1, k2_1; t0=t0_1)
    y1 = cdf1
    pl.plot!(p, trange, y1, label="1-hit model", 
             color=color1, linewidth=2.5, linestyle=:solid)
    
    # Two-hit model
    cdf2 = cdffct2(trange, kc2, k2_2; t0=t0_2)
    y2 = cdf2
    pl.plot!(p, trange, y2, label="2-hit model", 
             color=color2, linewidth=2.5, linestyle=:dash)

    # Three-hit model
    cdf3 = cdffct3(trange, kc3, k2_3; t0=t0_3)
    y3 = cdf3
    pl.plot!(p, trange, y3, label="3-hit model", 
             color=color3, linewidth=2.5, linestyle=:dot)
    
    pl.xlabel!(p, "Time (s)", fontsize=12)
    pl.ylabel!(p, "Fraction of activated cells", fontsize=12)

    # Annotate with RMS error of each model fit
    lbl_rms = @sprintf "\n%.3f" rms1
    lbl_rms = raw"RMSE$_1$ :" * lbl_rms
    d = div(length(y1), 4)
    pl.annotate!(p, trange[3*d+20], y1[3*d], 
        (lbl_rms, 12, color1, :top, :left))
    lbl_rms2 = @sprintf "\n%.3f" rms2
    lbl_rms2 = raw"RMSE$_2$ :" * lbl_rms2
    pl.annotate!(p, trange[2*d+20], y2[2*d], 
        (lbl_rms2, 12, color2, :top, :left))
    lbl_rms3 = @sprintf "\n%.3f" rms3
    lbl_rms3 = raw"RMSE$_3$ :" * lbl_rms3
    pl.annotate!(p, trange[d+30], y3[d], 
        (lbl_rms3, 12, color3, :bottom, :left))
    
    return p
end




"""
Function making plots for the plot showing vs against fluorescence threshold. 
Update: make k and t0 separate panels, we only show k because the choice of range
of fluorescence thresholds is based on it, while t0 is noisy because of individual
tracks that pop in and out of the active set as the threshold changes. 
"""
function plotrobustness(fits123, rms123, allnames, 
            fracrange; fluothresh_default=0.8)
    # Unwrap arguments
    fits1, fits2, fits3 = fits123
    rms1, rms2, rms3 = rms123

    shortnames, lblconcs = cleannames(allnames)
    n_dsets = length(allnames)

    # Filter to only the datasets at 0.01pM and 0.1pM concentrations
    selected = findall(x -> x == 0.01 || x == 0.1, lblconcs)
    if isempty(selected)
        error("plotrobustness: no 0.01pM or 0.1pM datasets found in allnames")
    end
    sel_names = shortnames[selected]

    # Extract parameter indices for the selected datasets
    kc_indices = 1 .+ selected

    # Use magma colors for the low-concentration datasets, matching their rank among all datasets.
    all_magma = pl.cgrad(:magma, n_dsets-2, categorical=true)
    sort_idx = sortperm(lblconcs)
    inverse_rank = zeros(Int, n_dsets)
    for (rank, idx) in enumerate(sort_idx)
        inverse_rank[idx] = rank
    end
    linecolors = [all_magma[inverse_rank[i]] for i in selected]

    # First plot: kc across threshold choices, per dataset and model
    fracrangep = fracrange .+ 1.0
    p_kc = pl.plot(layout=(1, 1), left_margin=6mm)
    for (sidx, dataset_idx) in enumerate(selected)
        lbl = sel_names[sidx]
        pl.plot!(p_kc, fracrangep, fits1[:, kc_indices[sidx]]; 
                label=lbl, color=linecolors[sidx], lw=2)
        pl.plot!(p_kc, fracrangep, fits2[:, kc_indices[sidx]]; 
                label=nothing, color=linecolors[sidx], linestyle=:dash, lw=2)
        pl.plot!(p_kc, fracrangep, fits3[:, kc_indices[sidx]]; 
                label=nothing, color=linecolors[sidx], linestyle=:dot, lw=2)
    end

    # Add legend entries describing the model styles
    pl.plot!(p_kc, [NaN, NaN], [NaN, NaN]; label="One-hit", color=:black, linestyle=:solid)
    pl.plot!(p_kc, [NaN, NaN], [NaN, NaN]; label="Two-hit", color=:black, linestyle=:dash)
    pl.plot!(p_kc, [NaN, NaN], [NaN, NaN]; label="Three-hit", color=:black, linestyle=:dot)

    pl.vline!(p_kc, [fluothresh_default+1.0]; linestyle=:dash, color=:black, label=nothing)

    pl.xlabel!(p_kc, raw"Fluorescence threshold factor $f$")
    pl.ylabel!(p_kc, raw"Hit rate $k_1$ (s$^\mathrm{-1}$)")
    pl.plot!(p_kc; yscale=:log10, minorgrid=true, fgcolor_minorgrid=:white,
            legend=:topright)

    # Shade area where fits are reasonable
    good_frac_range = [0.25, 1.2]
    pl.vspan!(p_kc, good_frac_range.+1.0, color=:grey, alpha=0.3, label=nothing)
    pl.annotate!(p_kc, 1.3, maximum(fits3[:, kc_indices])*0.85, 
        ("Optimal region", :grey, :bottom, :left))

    # Final plots: RMS error of linear vs quadratic fits
    # and RMS error of 3-step vs best of 1 or 2 steps,
    # across various threshold fractions, for each file. 
    rms1_sel = copy(rms1[:, selected])
    rms2_sel = copy(rms2[:, selected])
    rms3_sel = copy(rms3[:, selected])

    # Remove points where threshold is too small and many trajectories start
    # already active, which is typically factor <= 0.4
    mask = (good_frac_range[1] .<= fracrange .<= good_frac_range[2])
    npoints = length(fracrange)
    skp = 2
    rms1_sel = rms1_sel[mask, :][1:skp:end, :]
    rms2_sel = rms2_sel[mask, :][1:skp:end, :]
    rms3_sel = rms3_sel[mask, :][1:skp:end, :]
    # For excessively small RMS, set them to NaNs
    rms1_sel = cliprms!(rms1_sel)
    rms2_sel = cliprms!(rms2_sel)
    rms3_sel = cliprms!(rms3_sel)

    msmin, msmax = 4.0, 8.0
    npts = size(rms1_sel)[1]
    msizes = collect(LinRange(msmin, msmax, npts))

    ## Plot of one-hit vs two-hit RMS error
    pp2 = pl.plot(layout=(1, 1), size=(550, 450), bottom_margin=2mm,
            legend=false, legendfontsize=9)
    plotlimits = [1e-2, max(maximum(rms1_sel), maximum(rms2_sel), maximum(rms3_sel))*1.5]
    pl.plot!(pp2, plotlimits, plotlimits; color=:black, 
                linestyle=:dot, label=nothing)
    for i in 1:length(selected)
        pl.scatter!(pp2, rms1_sel[:, i], rms2_sel[:, i]; 
                color=linecolors[i], ms=msizes, label=sel_names[i])
    end

    # Hand-build a legend by plotting and labeling next to it
    # Stupidest thing I had to do in Julia because Plots does not give control
    # over legend entries and does not reproduce marker sizes in them
    # Switch to makie in future projects, it's as easy to do there as in Matplotlib.
    # Grey rectangle behind, x, y: position of bottom left corner
    logrectangle(w, h, x, y) = pl.Shape(
        exp.(log(10) .* (x .+ [0,w,w,0])), 
        exp.(log(10) .* (y .+ [0,0,h,h])))

    function customlegend(pp, logrmslims, ms_sel, frac_sel)
        logrmsx = 0.64 * (logrmslims[2]-logrmslims[1]) + logrmslims[1]
        logrmsy = 0.25 * (logrmslims[2]-logrmslims[1]) + logrmslims[1]
        dx, dy = 0.15, 0.07

        rect1 = logrectangle(0.71, 0.45, logrmsx-0.35*dx, logrmsy-length(ms_sel)*dy)
        pl.plot!(pp, rect1, lw=0.0, fillalpha=0.3, 
            color=:grey, linecolor=nothing, label=nothing)

        # Legend title
        pl.annotate!(pp, 10^(logrmsx), 10^(logrmsy+0.7*dy), 
            ("Fluorescence\nthreshold factor " * raw"$f$", :black, :left, :bottom))
        
        for i in 1:1:length(ms_sel)
            # Marker for lowest frac range
            pl.scatter!(pp, [10^logrmsx], [10^(logrmsy-(i-1)*dy)], 
                color=:black, ms=ms_sel[i], label=nothing)
            # Label of that marker, next to it
            pl.annotate!(pp, 10^(logrmsx+dx), 10^(logrmsy-(i-1)*dy), 
                ((@sprintf "%.1f" frac_sel[i]+1.0), :black, :left, :center))
        end
    end

    # Add custom legend
    ms_inter = collect(LinRange(msmin, msmax, npts))[div(npts, 2)]
    frac_inter = fracrange[div(npts, 2)]
    ms_selection = [msmin, ms_inter, msmax]
    frac_selection = [good_frac_range[1], frac_inter, good_frac_range[2]]
    logrmslimit = log10.(plotlimits)
    customlegend(pp2, logrmslimit, ms_selection, frac_selection)

    pl.xlabel!(pp2, "RMS error for one-hit model")
    pl.ylabel!(pp2, "RMS error for two-hit model")
    pl.plot!(pp2; xscale=:log10, yscale=:log10, xlims=plotlimits, ylims=plotlimits,
        minorgrid=true, fgcolor_minorgrid=:white)

    pct_one_better = sum(rms1_sel .<= rms2_sel) / length(rms1_sel) * 100.0
    lbl_one = (@sprintf "%.1f" pct_one_better) * " %"
    lbl_two = (@sprintf "%.1f" (100.0 - pct_one_better)) * " %"
    pl.annotate!(pp2, plotlimits[1]*1.1, plotlimits[2]*0.9, 
        ("One-hit is better: " * lbl_one, :black, :left, :top))
    pl.annotate!(pp2, plotlimits[2]*0.9, plotlimits[1]*1.1, 
        ("Two-hit is better: " * lbl_two, :black, :right, :bottom))

    
    ## Plot of best(one-hit, two-hit) vs three-hit RMS error
    best12_rms = min.(rms1_sel, rms2_sel)
    pp3 = pl.plot(layout=(1, 1), size=(550, 450), bottom_margin=2mm,
            legend=false, minorgrid=true, fgcolor_minorgrid=:white)
    pl.plot!(pp3, plotlimits, plotlimits; color=:black, 
            linestyle=:dot, label=nothing)
    for i in 1:length(selected)
        pl.scatter!(pp3, best12_rms[:, i], rms3_sel[:, i]; 
                color=linecolors[i], ms=msizes, label=sel_names[i])
    end

    customlegend(pp3, logrmslimit, ms_selection, frac_selection)
    pl.xlabel!(pp3, "RMS error for one or two-hit model")
    pl.ylabel!(pp3, "RMS error for three-hit model")
    pl.plot!(pp3; xscale=:log10, yscale=:log10, 
        xlims=plotlimits, ylims=plotlimits,
        minorgrid=true, fgcolor_minorgrid=:white)

    pct_three_better = sum(rms3_sel .<= best12_rms) / length(best12_rms) * 100.0
    lbl_best12 = (@sprintf "%.1f" (100.0 - pct_three_better)) * " %"
    lbl_three = (@sprintf "%.1f" pct_three_better) * " %"
    pl.annotate!(pp3, plotlimits[1]*1.1, plotlimits[2]*0.9, 
        ("One- or two-hit is better: " * lbl_best12, :black, :left, :top))
    pl.annotate!(pp3, plotlimits[2]*0.9, plotlimits[1]*1.1, 
        ("Three-hit is better: " * lbl_three, :black, :right, :bottom))

    return [p_kc, pp2, pp3]
end
