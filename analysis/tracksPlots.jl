#=
Plotting functions for tracks analysis. 

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


module models
    include("modelsML.jl")
end
using .models: expocdf

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


# TODO: put back plotrobustness after fine-tuning it