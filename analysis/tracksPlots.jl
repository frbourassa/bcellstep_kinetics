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
using .tracksAnalysis: expocdf, twostepcdf, prepcdf, threestepcdf, concfromstr, cleannames


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
function plotrobustness(linfits, quadfits, fits3, allnames, 
            fracrange; fluothresh_default=0.5)
    # Extract concentration of each name
    shortnames, lblconcs = cleannames(allnames)
    println(shortnames)

    # Sort trajectories by decreasing concentration (highest at the top)
    sortidx = reverse(sortperm(lblconcs), dims=1)
    shortnames = shortnames[sortidx]
    println(shortnames)
    lblconcs = lblconcs[sortidx]
    linfits = linfits[sortidx, :, :]  # file index is 1st dim, OK to sort this way
    quadfits = quadfits[sortidx, :, :]  # same
    fits3 = fits3[sortidx, :, :]
   
    # Define a color scheme increasing with concentration
    #linecolors = pl.palette([:purple, :green], length(shortnames))
    linecolors = pl.cgrad(:thermal, length(shortnames), rev = true, categorical = true)

    # First plot: fit parameters as a function of threshold fraction, 
    # for each experiment, to show robustness. Linear: solid line, quadratic: dashed. 
    # Top: slope, bottom: t0.
    ppp = [pl.plot(layout=(1, 1), left_margin=6mm),
            pl.plot(layout=(1, 1), left_margin=6mm)]
    for i in 1:1:length(allnames)
        # slope
        lbl = (i == length(allnames)) ? "1-step" : nothing
        pl.plot!(ppp[1], fracrange.+1.0, linfits[i, 1:end, 1]; 
            label=lbl, color=linecolors[i])
        lbl = (i == length(allnames)) ? "2-step" : nothing
        pl.plot!(ppp[1], fracrange.+1.0, quadfits[i, 1:end, 1];
            label=lbl, color=linecolors[i], linestyle=:dash)
        lbl = (i == length(allnames)) ? "3-step" : nothing
        pl.plot!(ppp[1], fracrange.+1.0, fits3[i, 1:end, 1];
            label=lbl, color=linecolors[i], linestyle=:dot)
        # t0
        pl.plot!(ppp[2], fracrange.+1.0, linfits[i, 1:end, 5];
            label=shortnames[i], color=linecolors[i])
        pl.plot!(ppp[2], fracrange.+1.0, quadfits[i, 1:end, 5];
            label=nothing, color=linecolors[i], linestyle=:dash)
        pl.plot!(ppp[2], fracrange.+1.0, fits3[i, 1:end, 5];
            label=nothing, color=linecolors[i], linestyle=:dot)
    end
    # Assemble and label plots    
    for i in 1:1:2
        pl.vline!(ppp[i], [fluothresh_default+1.0]; 
                linestyle=:dash, color=:black, label=nothing)
    end

    # Labels and final parameters
    pl.xlabel!(ppp[1], raw"Fluorescence threshold factor $f$")
    pl.xlabel!(ppp[2], raw"Fluorescence threshold factor $f$")
    pl.plot!(ppp[1]; yscale=:log10, legend=:topright, legendfontsize=9, 
            ylabel=raw"Activation rate $k$ (s$^\mathrm{-1}$)")
    pl.plot!(ppp[2]; ylabel=raw"Start time $t_0$ (s)", 
            legend=:topleft, legendfontsize=9, 
            legend_title="pHELD-AF594")


    # Shade area where fits are reasonable
    good_frac_range = [0.4, 1.1]
    pl.vspan!(ppp[1], good_frac_range.+1.0, color=:grey, alpha=0.3, label=nothing)
    pl.vspan!(ppp[2], good_frac_range.+1.0, color=:grey, alpha=0.3, label=nothing)
    pl.annotate!(ppp[1], 1.63, maximum(fits3[:, end, 1]*4), 
        ("Optimal region", 10, :grey, :bottom, :left))
    
    ydefault = maximum(linfits[:, :, 1]*0.75)

    pl.annotate!(ppp[1], 1.8, ydefault, ("Default", 10, :black, :center, :left))
    # draw an arrow from (1.77, ydefault) to (1.63, ydefault)
    pl.plot!(ppp[1], [1.77, 1.63], [ydefault, ydefault*0.8],
        color=:black, lw=2, label=nothing)


    # Final plots: RMS error of linear vs quadratic fits
    # and RMS error of 3-step vs best of 1 or 2 steps,
    # across various threshold fractions, for each file. 
    rmslinplots = copy(linfits[:, :, 4])
    rmsquadplots = copy(quadfits[:, :, 4])
    rms3plots = copy(fits3[:, :, 4])

    # Remove points where threshold is too small and many trajectories start
    # already active, which is typically factor <= 0.4
    mask = (good_frac_range[1] .<= fracrange .<= good_frac_range[2])
    skp = 3
    rmslinplots = rmslinplots[:, mask][:, 1:skp:end]
    rmsquadplots = rmsquadplots[:, mask][:, 1:skp:end]
    rms3plots = rms3plots[:, mask][:, 1:skp:end]
    # For excessively small RMS, set them to NaNs
    rmslinplots = cliprms!(rmslinplots)
    rmsquadplots = cliprms!(rmsquadplots)
    rms3plots = cliprms!(rms3plots)

    ## Plot of 1 vs 2 steps
    pp2 = pl.plot(layout=(1, 1), size=(450, 450), bottom_margin=2mm,
                    legend=:left, legendfontsize=9)
    plotlimits = [minimum(rmslinplots)*0.8, maximum(rmslinplots)*1.4]
    pl.plot!(pp2, plotlimits, plotlimits; color=:black, 
            linestyle=:dot, label=nothing)
    msmin, msmax = 3.0, 7.0
    for i in 1:1:size(rmslinplots, 1)
        xvals = rmslinplots[i, :]
        yvals = rmsquadplots[i, :]
        sizes = collect(LinRange(msmin, msmax, length(xvals)))
        pl.scatter!(pp2, xvals, yvals; color=linecolors[i], ms=sizes, 
                label=shortnames[i])
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
        logrmsx = 0.7 * (logrmslims[2]-logrmslims[1]) + logrmslims[1]
        logrmsy = 0.18 * (logrmslims[2]-logrmslims[1]) + logrmslims[1]
        dx, dy = 0.15, 0.07

        rect1 = logrectangle(0.6, 0.42, logrmsx-0.35*dx, logrmsy-length(ms_sel)*dy)
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
    
    # Add the legend
    logrmslims = log10.([minimum(min.(rmslinplots, rmsquadplots))*1.1, 
                        maximum(max.(rmslinplots, rmsquadplots))*0.8])
    npts = length(fracrange)
    ms_inter = collect(LinRange(msmin, msmax, npts))[div(npts, 2)]
    frac_inter = fracrange[div(npts, 2)]
    ms_selection = [msmin, ms_inter, msmax]
    frac_selection = [good_frac_range[1], frac_inter, good_frac_range[2]]
    customlegend(pp2, logrmslims, ms_selection, frac_selection)
    
    # Compute and annotate fraction of points where linear or quadratic is better
    pct_lin_better = sum(rmslinplots .<= rmsquadplots) / length(rmslinplots) * 100.0
    lbl_lin = (@sprintf "%.1f" pct_lin_better) * " %"
    lbl_quad = (@sprintf "%.1f" (100.0 - pct_lin_better)) * " %"
    pl.xlabel!(pp2, "RMS error for 1-step fit")
    pl.ylabel!(pp2, "RMS error for 2-step fit")
    pl.annotate!(pp2, plotlimits[1], plotlimits[2], 
        ("1-step is better: "*lbl_lin, :black, :left, :top))
    pl.annotate!(pp2, plotlimits[2]*1.05, plotlimits[1]*0.95, 
        ("2-step is better: " * lbl_quad, :black, :right, :bottom))
    pl.plot!(pp2; xscale=:log10, yscale=:log10)
    
    ## Plot of (best of 1, 2) vs 3 steps
    best_12_rms = min.(rmslinplots, rmsquadplots)
    plotlimits = [minimum(min.(best_12_rms, rms3plots))*0.8, 
                maximum(max.(rms3plots, best_12_rms))*1.4]
    pp3 = pl.plot(layout=(1, 1), size=(450, 450), legend=false, bottom_margin=2mm)
    pl.plot!(pp3, plotlimits, plotlimits; color=:black, 
            linestyle=:dot, label=nothing)
    msmin, msmax = 3.0, 7.0
    for i in 1:1:size(best_12_rms, 1)
        xvals = best_12_rms[i, :]
        yvals = rms3plots[i, :]
        sizes = collect(LinRange(msmin, msmax, length(xvals)))
        pl.scatter!(pp3, xvals, yvals; color=linecolors[i], ms=sizes, label=nothing)
    end

    # Hand-build a legend by plotting and labeling next to it
    logrmslims = log10.([minimum(min.(best_12_rms, rms3plots)), 
                        maximum(max.(rms3plots, best_12_rms))])
    # Add the legend
    customlegend(pp3, logrmslims, ms_selection, frac_selection)

    # Annotations
    pct_3_better = sum(rms3plots .<= best_12_rms) / length(rms3plots) * 100.0
    lbl_12 = (@sprintf "%.1f" (100.0 - pct_3_better)) * " %"
    lbl_3 = (@sprintf "%.1f" pct_3_better) * " %"
    pl.xlabel!(pp3, "RMS error for better of 1- or 2-step")
    pl.ylabel!(pp3, "RMS error for 3-step fit")
    pl.annotate!(pp3, plotlimits[1], plotlimits[2], 
        ("1- or 2-step is better: " * lbl_12, :black, :left, :top))
    pl.annotate!(pp3, plotlimits[2]*1.05, plotlimits[1]*0.95, 
        ("3-step is better: " * lbl_3, :black, :right, :bottom))
    pl.plot!(pp3; xscale=:log10, yscale=:log10, legend=nothing, 
        legend_title="Threshold factor " * raw"$f$")

    return [ppp, pp2, pp3]
end
