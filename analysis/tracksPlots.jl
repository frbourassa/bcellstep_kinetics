#=
Plotting functions to tracks analysis. 

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


module modelsML
    include("modelsML.jl")
end
using .modelsML: onehit_cdf, twohit_cdf, threehit_cdf


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


### PLOTTING FUNCTIONS ###
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
    ppp = pl.plot(layout=(1, 1), size=(450, 450), left_margin=3mm, 
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
    pl.ylabel!(p, raw"Ca2+ fluorescence (a.u.)")

    # Title the panel
    if conclbl !== nothing
        xmid = times[div(length(times), 2)]
        ytop = maximum(maximum.(skipmissing.(eachrow(dfi))))
        conclbl2 = replace(conclbl, "1pM 2" => "1pM")  # replicate 1 low cell count
        pl.annotate!(p, xmid, ytop, ("Example: " * conclbl2, 12))
    end

    return ppp
end

"""
Given a list of activation times, plot the cumulative distribution 
of activation times. Check whether the number of activated cells as 
a function of time increases linearly, quadratically, etc. 
"""
function plottimescdffit!(ppinfo, tactiv, cdfparams, quadparams, pms3; 
        title=nothing)
    tu, cumul, _, ncells = prepcdf(tactiv, 1.0)
    # Prepend 0 for plotting
    goodt = vcat([0], tu)
    cumul = vcat([0], cumul)
    ncumul = cumul .* ncells

    ppp, p1, p1idx, insetidx = ppinfo

    # Also plot the fitted straight line. Annotate RMS error. 
    slope, _, tlim, rmslin, t0, _ = cdfparams
    trange = collect(LinRange(minimum(goodt), tlim, 100))
    yrange = expocdf(trange, slope; t0=t0)
    #yrange = slope .* trange .+ inter
    # Scale to number of cells
    yrange = yrange .* ncells

    # Also quadratic regression
    slope2, _, tlim2, rmsquad, t02, _ = quadparams
    #yrange2 = (slope2 .* trange.^2 .+ inter2) .* ncells
    trange2 = collect(LinRange(minimum(goodt), tlim2, 100))
    yrange2 = twostepcdf(trange2, slope2; t0=t02)
    yrange2 = yrange2 .* ncells

    # Also 3-step cdf
    slope3, _, tlim3, rms3, t03, _ = pms3
    trange3 = collect(LinRange(minimum(goodt), tlim3, 100))
    yrange3 = threestepcdf(trange3, slope3; t0=t03)
    yrange3 = yrange3 .* ncells
    
    # CDF plot
    lwdat = 1.5
    lwfit = 2.0
    pl.plot!(p1, goodt, ncumul, linetype=:steppost, 
        color=:black, label=nothing, lw=lwdat)
    # Add marker at actual data points
    pl.plot!(p1, goodt[2:end], ncumul[2:end], linetype=:scatter, 
        color=:black, label=nothing, m=(3, :circle))

    lincolor = :blue
    pl.plot!(p1, trange, yrange, label="1-step", color=lincolor, lw=lwfit)

    # Also plot quadratic fit
    quadcolor = :red
    quadls = :dash
    pl.plot!(p1, trange2, yrange2, label="2-step", color=quadcolor, 
        lw=lwfit, linestyle=quadls)
    # And 3-step fit
    threecolor = :purple
    threels = :dot
    pl.plot!(p1, trange3, yrange3, label="3-step", color=threecolor,
        legend=:topleft, lw=lwfit, linestyle=threels)
    
    pl.xlabel!(p1, "Time (s)")
    pl.ylabel!(p1, raw"Number of activated cells")


    # Add inset by plotting data directly with subplot and inset parameters
    mask = goodt .<= min(tlim, tlim2)
    p3 = pl.plot!(ppp, goodt[mask], ncumul[mask];
        linetype=:steppost,
        inset = (p1idx, pl.bbox(0.05, 0.05, 0.4, 0.75, :bottom, :right)),
        subplot = insetidx, framestyle = :box, bg_inside = nothing, 
        ticks = nothing, label = nothing, lw=lwdat * 2.0, color = :black,
    )
    # Add marker at actual data points
    pl.plot!(ppp, goodt[mask][2:end], ncumul[mask][2:end];
        subplot = insetidx, linetype=:scatter, 
        color=:black, label=nothing, m=(5, :circle)
    )
    
    # Add linear fit to the inset (subplot 2)
    pl.plot!(ppp, trange, yrange;
        subplot = insetidx, label = nothing, 
        lw = lwfit * 2.5, color = lincolor,
    )
    # Also quadratic and cubic fits
    p2 = pl.plot!(ppp, trange2, yrange2;
        subplot = insetidx, label = nothing,
        lw=lwfit * 2.5, linestyle=quadls, color=quadcolor
    )
    p3 = pl.plot!(ppp, trange3, yrange3;
        subplot = insetidx, label = nothing,
        lw=lwfit * 2.5, linestyle=threels, color=threecolor
    )
    # Annotate RMS error in the inset, cleaner
    lbl_rms = @sprintf "\n%.1e" rmslin
    lbl_rms = raw"RMS$_1$ :" * lbl_rms
    pl.annotate!(ppp, trange[1], yrange[end], 
        (lbl_rms, 10, lincolor, :top, :left); subplot=insetidx)
    mid = div(length(yrange2), 3)
    lbl_rms2 = @sprintf "\n%.1e" rmsquad
    lbl_rms2 = raw"RMS$_2$ :" * lbl_rms2
    pl.annotate!(ppp, trange[end-5], yrange2[mid], 
        (lbl_rms2, 10, quadcolor, :bottom, :right); subplot=insetidx)
    lbl_rms3 = @sprintf "\n%.1e" rms3
    lbl_rms3 = raw"RMS$_3$ :" * lbl_rms3
    pl.annotate!(ppp, trange[end-15], yrange2[1]-1, 
        (lbl_rms3, 10, threecolor, :bottom, :right); subplot=insetidx)

    # To mark where the inset was taken from, 
    # define a function that returns a Plots.Shape
    # https://discourse.julialang.org/t/how-to-draw-a-rectangular-region-with-plots-jl/1719/2
    rectangle(w, h, x, y) = pl.Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    rect1 = rectangle(tlim, max(yrange[end], yrange2[end])+0.05,0,0)

    pl.plot!(p1, rect1, opacity=1.0, lw=1.0, fillalpha=0.0, 
        color=:transparent, linecolor=:black, label=nothing)

    # Add small lines connecting the boxes
    xx = maximum(rect1.x)
    y1, y2 = extrema(rect1.y)
    xinsbox = goodt[end] * (1.0 - 0.475)
    yinsbox = 0.8 * maximum(ncumul)
    pl.plot!(p1, [xx, xinsbox], [y1, y1], color=:black, lw=1.0, label=nothing)
    pl.plot!(p1, [xx, xinsbox], [y2, yinsbox], color=:black, lw=1.0, label=nothing)

    # Title the plot if possible
    if title !== nothing
        xmid = 0.5 * sum(pl.xlims(p1))
        ytop = maximum(ncumul)
        title2 = replace(title, "1pM 2" => "1 pM")  # replicate 1 low cell count
        pl.annotate!(p1, xmid, ytop, ("Example: " * title2, 12, :center, :bottom))
    end
    return p1
end

"""
Plot all acquired CDFs, two per VLP concentration. Assume datasets are already sorted.
"""
function plotallcdfs(p, alltactivs, shortnames, all_cdfparams)

    # Define a color scheme increasing with concentration
    #linecolors = pl.palette([:purple, :green], length(shortnames))
    linecolors = pl.cgrad(:thermal, length(shortnames), rev = true, categorical = true)

    # Single out the duplicates
    pl.plot!(p, legend_title="pHELD-AF594")
    for i in 1:1:length(alltactivs)
        lbl = shortnames[i]
        tactiv = alltactivs[i]
        tu, cumul, _, ncells = prepcdf(tactiv, 1.0)
        #println("For $lbl, found $ncells cells")
        # Prepend 0 for plotting
        goodt = vcat([0], tu)
        cumul = vcat([0], cumul)
        p = pl.plot!(p, goodt, cumul, linetype=:steppost, 
            label=lbl, color=linecolors[i], marker=(:circle, 3),
            markerstrokewidth=0.0)

        # Also plot the fitted straight line
        cdfparams = all_cdfparams[i]
        slope, _, tlim, _, t0, _ = cdfparams
        trange = collect(LinRange(minimum(goodt), tlim, 20))
        yrange = expocdf(trange, slope; t0=t0)

        lbl = @sprintf "%.1f" slope
        lbl = lbl *  raw"s$^{-1}$"
        lbl = raw"$\kappa v =$ " * lbl
        p = pl.plot!(p, trange, yrange, label=nothing, color=linecolors[i], lw=2.0)
        #pl.annotate!(p, trange[end], yrange[end], (lbl, 8, linecolors[i], :bottom))
    end
    pl.xlabel!(p, "Time (s)")
    pl.ylabel!(p, raw"Fraction activated cells")
    return p
end


"""
Function making plots for the plot showing vs against fluorescence threshold. 
Update: make k and t0 separate panels, we only show k because the choice of range
of fluorescence thresholds is based on it, while t0 is noisy because of individual
tracks that pop in and out of the active set as the threshold changes. 
"""
function plotrobustness(fits123, rms123, allnames, 
            fracrange; fluothresh_default=0.5)
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
    kappa_indices = 1 .+ selected
    t0_indices = 1 .+ n_dsets .+ selected

    # Use magma colors for the low-concentration datasets, matching their rank among all datasets.
    all_magma = pl.cgrad(:magma, n_dsets-1, categorical=true)
    sort_idx = sortperm(lblconcs)
    inverse_rank = zeros(Int, n_dsets)
    for (rank, idx) in enumerate(sort_idx)
        inverse_rank[idx] = rank
    end
    linecolors = [all_magma[inverse_rank[i]] for i in selected]

    # First plot: kappa and t0 across threshold choices, per dataset and model
    fracrangep = fracrange .+ 1.0
    p_kappa = pl.plot(layout=(1, 1), left_margin=6mm)
    p_t0 = pl.plot(layout=(1, 1), left_margin=6mm)
    for (sidx, dataset_idx) in enumerate(selected)
        lbl = sel_names[sidx]
        pl.plot!(p_kappa, fracrangep, fits1[:, kappa_indices[sidx]]; 
                label=lbl, color=linecolors[sidx], lw=2)
        pl.plot!(p_kappa, fracrangep, fits2[:, kappa_indices[sidx]]; 
                label=nothing, color=linecolors[sidx], linestyle=:dash, lw=2)
        pl.plot!(p_kappa, fracrangep, fits3[:, kappa_indices[sidx]]; 
                label=nothing, color=linecolors[sidx], linestyle=:dot, lw=2)

        pl.plot!(p_t0, fracrangep, fits1[:, t0_indices[sidx]]; 
                label=lbl, color=linecolors[sidx], lw=2)
        pl.plot!(p_t0, fracrangep, fits2[:, t0_indices[sidx]]; 
                label=nothing, color=linecolors[sidx], linestyle=:dash, lw=2)
        pl.plot!(p_t0, fracrangep, fits3[:, t0_indices[sidx]]; 
                label=nothing, color=linecolors[sidx], linestyle=:dot, lw=2)
    end

    # Add legend entries describing the model styles
    pl.plot!(p_kappa, [NaN, NaN], [NaN, NaN]; label="One-hit", color=:black, linestyle=:solid)
    pl.plot!(p_kappa, [NaN, NaN], [NaN, NaN]; label="Two-hit", color=:black, linestyle=:dash)
    pl.plot!(p_kappa, [NaN, NaN], [NaN, NaN]; label="Three-hit", color=:black, linestyle=:dot)

    for p in (p_kappa, p_t0)
        pl.vline!(p, [fluothresh_default+1.0]; linestyle=:dash, color=:black, label=nothing)
    end

    pl.xlabel!(p_kappa, raw"Fluorescence threshold factor $f$")
    pl.ylabel!(p_kappa, raw"Activation rate $\kappa$ (s$^\mathrm{-1}$)")
    pl.plot!(p_kappa; yscale=:log10, minorgrid=true, fgcolor_minorgrid=:white,
            legend=:topright, legendfontsize=9)

    pl.xlabel!(p_t0, raw"Fluorescence threshold factor $f$")
    pl.ylabel!(p_t0, raw"Start time $t_0$ (s)")
    pl.plot!(p_t0; legend=:topleft, legendfontsize=9, legend_title="Dataset")

    # Shade area where fits are reasonable
    good_frac_range = [0.25, 1.2]
    pl.vspan!(p_kappa, good_frac_range.+1.0, color=:grey, alpha=0.3, label=nothing)
    pl.vspan!(p_t0, good_frac_range.+1.0, color=:grey, alpha=0.3, label=nothing)
    pl.annotate!(p_kappa, 1.63, maximum(fits3[:, kappa_indices]), 
        ("Optimal region", 10, :grey, :bottom, :left))

    # Final plots: RMS error of linear vs quadratic fits
    # and RMS error of 3-step vs best of 1 or 2 steps,
    # across various threshold fractions, for each file. 
    rms1_sel = copy(rms1[:, selected])
    rms2_sel = copy(rms2[:, selected])
    rms3_sel = copy(rms3[:, selected])

    # Remove points where threshold is too small and many trajectories start
    # already active, which is typically factor <= 0.4
    mask = (good_frac_range[1] .<= fracrange .<= good_frac_range[2])
    skp = 1
    rms1_sel = rms1_sel[mask, :][1:skp:end, :]
    rms2_sel = rms2_sel[mask, :][1:skp:end, :]
    rms3_sel = rms3_sel[mask, :][1:skp:end, :]
    # For excessively small RMS, set them to NaNs
    rms1_sel = cliprms!(rms1_sel)
    rms2_sel = cliprms!(rms2_sel)
    rms3_sel = cliprms!(rms3_sel)

    msmin, msmax = 3.0, 7.0
    msizes = collect(LinRange(msmin, msmax, length(fracrange)))

    ## Plot of one-hit vs two-hit RMS error
    pp2 = pl.plot(layout=(1, 1), size=(450, 450), bottom_margin=2mm,
            legend=false, legendfontsize=9)
    plotlimits = [1e-2, max(maximum(rms1_sel), maximum(rms2_sel))*1.5]
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
        logrmsx = 0.65 * (logrmslims[2]-logrmslims[1]) + logrmslims[1]
        logrmsy = 0.35 * (logrmslims[2]-logrmslims[1]) + logrmslims[1]
        dx, dy = 0.15, 0.07

        rect1 = logrectangle(0.7, 0.42, logrmsx-0.35*dx, logrmsy-length(ms_sel)*dy)
        pl.plot!(pp, rect1, lw=0.0, fillalpha=0.3, 
            color=:grey, linecolor=nothing, label=nothing)

        # Legend title
        pl.annotate!(pp, 10^(logrmsx), 10^(logrmsy+0.7*dy), 
            ("Fluorescence\nthreshold factor " * raw"$f$", 10, :black, :left, :bottom))
        
        for i in 1:1:length(ms_sel)
            # Marker for lowest frac range
            pl.scatter!(pp, [10^logrmsx], [10^(logrmsy-(i-1)*dy)], 
                color=:black, ms=ms_sel[i], label=nothing)
            # Label of that marker, next to it
            pl.annotate!(pp, 10^(logrmsx+dx), 10^(logrmsy-(i-1)*dy), 
                ((@sprintf "%.1f" frac_sel[i]+1.0), 10, :black, :left, :center))
        end
    end

    # Add custom legend
    npts = length(fracrange)
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
        ("One-hit is better: " * lbl_one, 10, :black, :left, :top))
    pl.annotate!(pp2, plotlimits[2]*0.9, plotlimits[1]*1.1, 
        ("Two-hit is better: " * lbl_two, 10, :black, :right, :bottom))

    
    ## Plot of best(one-hit, two-hit) vs three-hit RMS error
    best12_rms = min.(rms1_sel, rms2_sel)
    pp3 = pl.plot(layout=(1, 1), size=(450, 450), bottom_margin=2mm,
            legend=false, minorgrid=true, fgcolor_minorgrid=:white)
    plotlimits = [1e-2, max(maximum(best12_rms), maximum(rms3_sel))*1.4]
    pl.plot!(pp3, plotlimits, plotlimits; color=:black, 
            linestyle=:dot, label=nothing)
    for i in 1:length(selected)
        pl.scatter!(pp3, best12_rms[:, i], rms3_sel[:, i]; 
                color=linecolors[i], ms=msizes, label=sel_names[i])
    end

    customlegend(pp3, logrmslimit, ms_selection, frac_selection)
    pl.xlabel!(pp3, "RMS error for better of one-/two-hit model")
    pl.ylabel!(pp3, "RMS error for three-hit model")
    pl.plot!(pp3; xscale=:log10, yscale=:log10, 
        xlims=plotlimits, ylims=plotlimits,
        minorgrid=true, fgcolor_minorgrid=:white)

    pct_three_better = sum(rms3_sel .<= best12_rms) / length(best12_rms) * 100.0
    lbl_best12 = (@sprintf "%.1f" (100.0 - pct_three_better)) * " %"
    lbl_three = (@sprintf "%.1f" pct_three_better) * " %"
    pl.annotate!(pp3, plotlimits[1]*1.1, plotlimits[2]*0.9, 
        ("One- or two-hit is better: " * lbl_best12, 10, :black, :left, :top))
    pl.annotate!(pp3, plotlimits[2]*0.9, plotlimits[1]*1.1, 
        ("Three-hit is better: " * lbl_three, 10, :black, :right, :bottom))

    ppp = [p_kappa, p_t0]
    return [ppp, pp2, pp3]
end
