""" Functions to load and preprocess microscopy tracks data """

using DataFrames: DataFrames as DF  # all methods available as DF.method
using CSV: CSV  # all methods available as CSV.method
using Optim: Optim, optimize, IPNewton
using LinearAlgebra: UniformScaling, diag, diagm, svd
using Printf: @sprintf
using Plots: Plots as pl


### SHORT UTILITY FUNCTIONS ###

function padnumber(n)   
    return string(n; pad=3)
end

# From ChatGPT, to replace Python's str.isnumeric() method. 
isnumeric(str::AbstractString) = !isempty(str) && all(isdigit, str)


# Utility function to convert a string to float
function concfromstr(s, unit, delim)
    x = split(rstrip((split(s, unit)[1]), only(delim)), delim)[end]
    # tryparse(type, str; base), e.g. tryparse(Int64, "001") returns 1
    return tryparse(Float64, x)
end


"""
Get nice clean dataset names from the raw file names. 
"""
function cleannames(namelist)
    # Extract concentration of each name
    shortnames = collect(join(split(x, "_")[2:end], " ") for x in namelist)
    shortnames = collect(join(split(x, ".")[1:end-1], ".") for x in shortnames)
    shortnames = collect(map(x -> replace(x, " z2" => ""), shortnames))
    lblconcs = collect(concfromstr(x, "pM", " ") for x in shortnames)  # in pM
    return shortnames, lblconcs
end


### FUNCTIONS TO LOAD AND PREPROCESS TRACKS DATASETS ###

function gettrajs(df, colname)
    # https://stackoverflow.com/questions/61882298/how-to-apply-a-function-columnwise-to-julia-dataframe
    dftrajs = DF.dropmissing(df, :TRACK_ID)
    # a cols => function => target_cols form
    dftrajs = DF.transform(dftrajs, :TRACK_ID => DF.ByRow(padnumber) => :TRACK_ID_STR)
    # Pad track ids with zeros so they will sort properly as string column names
    dftrajs = DF.unstack(dftrajs, :FRAME, :TRACK_ID_STR, colname)
    dfsorted = sort(dftrajs, :FRAME)
    dfsorted2 = dfsorted[:, sort(DF.names(dfsorted))]
    # Add back time points
    dftime = gettimemap(df)
    dfsorted2 = DF.hcat(dfsorted2, dftime, makeunique=true)
    dfsorted2 = DF.select!(dfsorted2, DF.Not(:FRAME_1))
    return dfsorted2, dftime
end

function gettimemap(df)
    timemap = DF.combine(DF.groupby(df, :FRAME), :POSITION_T => DF.median => :TIME)
    return timemap
end

function filtertrajs(df; lenthresh=10, timethresh=30.0)
    # Treat one trajectory at a time
    todrop = Array{Union{String, Missing}}(missing, size(df)[2])
    for (i, col) in enumerate(DF.names(df))
        # Ignore TIME and FRAME columns
        if tryparse(Int64, col) !== nothing
            # Keep only non-missing values to do the processing
            track = collect(skipmissing(df[:, col]))
            if length(track) < lenthresh
                todrop[i] = col
            end
            # Remove trajs that start too late since we can't infer
            # a reliable activation time from them. 
            idxfirst = findfirst(x -> (x !== missing), df[:, col])
            tfirst = df[idxfirst, "TIME"]
            if tfirst > timethresh
                todrop[i] = col
            end
        end
    end
    # remove the missings: these are not to be dropped
    todrop = convert(Array{String}, collect(skipmissing(todrop)))
    filteredf = DF.select(df, DF.Not(todrop))
    return filteredf
end

function preptrajdata(fpath; ch="MEAN_INTENSITY_CH2", filterpms=[50, 30.0])
    # Full list of columns: see data/bcells_ucsf/microscopy/column_names.txt
    # Keep only the relevant data for checking Ca2+ trajectory statistics. Info:
    # CH1 (B220-AF647): B cell marker, outlines the cells
    # CH2 (Ca2+): intracellular calcium, all around the cell interior
    # CH3 (pHELD-AF594): shouldn't see much, these are VLPs, they move fast
    # and are lost in the cell fluo after binding
    # We only care about channel 2 here
    goodcols = ["ID", "TRACK_ID", "QUALITY", "POSITION_T", "FRAME",
                "MEAN_INTENSITY_CH1", "MEDIAN_INTENSITY_CH1",
                "MIN_INTENSITY_CH1", "MAX_INTENSITY_CH1", 
                "TOTAL_INTENSITY_CH1", "STD_INTENSITY_CH1",
                "MEAN_INTENSITY_CH2", "MEDIAN_INTENSITY_CH2",
                "MIN_INTENSITY_CH2", "MAX_INTENSITY_CH2", 
                "TOTAL_INTENSITY_CH2", "STD_INTENSITY_CH2"]
    df = CSV.read(fpath, DF.DataFrame; header=1, skipto=5, select=goodcols)

    # Focus further on the cell-volume-mean intensity: 
    # Ca2+ concentration, normalized by cell size
    dftrajs, frametimemap = gettrajs(df, ch)
    #println(DF.first(dftrajs[:, 1:10], 10))
    #println(DF.last(frametimemap, 10))

    # Filter out excessively short tracks. Cells come in and out of the imaging plane... 
    nalltracks = size(dftrajs)[2] - 2
    # Also remove trajs starting too late to reliably infer an activ. time
    dfiltered = filtertrajs(dftrajs; lenthresh=filterpms[1], timethresh=filterpms[2])
    ndropped = nalltracks - (size(dfiltered)[2] - 2)
    println("Filtering removed $ndropped tracks out of $nalltracks")
    
    return dfiltered
end



#= FUNCTIONS FOR THRESHOLD-BASED ANALYSIS OF INDIVIDUAL TRACKS ACTIVATION TIMES =#

"""
Updated version: linear interpolation between the last inactive
and first active point
"""
function findjump_thresh(tpoints, cpoints, thresh)
    # Get position of first point above threshold
    idxactiv = findfirst(x -> (x !== missing && x >= thresh), cpoints)
    if idxactiv !== nothing
        if idxactiv > 1
            ti = tpoints[idxactiv - 1]  # last inactive
            ci = cpoints[idxactiv - 1]
            ti1 = tpoints[idxactiv]
            ci1 = cpoints[idxactiv]
            # linear interpolation: t at which threshold is reached
            tactiv = ti + (ti1 - ti) * (thresh - ci) / (ci1 - ci)
            yactiv = thresh
        else  # first point
            tactiv = tpoints[idxactiv]  # First time after jump
            yactiv = cpoints[idxactiv]  # y value after jump
        end
    else
        tactiv = missing
        yactiv = missing
    end
    return tactiv, yactiv
end


"""
Estimate the background very conservatively: min value in each trajectory, median
"""
function findbackmin(df; stat="median")
    cback = 0.0
    ntrajs = 0.0
    # We should use the median, not the mean
    # Collect the minimum of each trajectory
    filterednames = filter(isnumeric, names(df))
    filtereddf = @view df[:, filterednames]
    ntrajs = size(filtereddf)[2]
    minvalues = minimum.(skipmissing.(eachcol(filtereddf)))

    if stat == "median"
        cback = sum(minvalues) / ntrajs
    elseif stat == "mean"
        cback = median(minvalues)
    else
        error("stat should be either 'median' or 'mean'. ")
    end
    return cback
end


"""
Function to apply findjump_thresh to each trajectory in the DataFrame of tracks
Also returns movie duration. 
"""
function findactivtimes_thresh(dfc; fracabove=0.5, disp=true)  
    if disp
        display(DF.last(dfc[1:end, 1:10], 10))
    end
    # First, extract background and calculate threshold based on it
    # Naive method: take the average min of each trajectory. This turns out to be best, 
    # it under-estimates the background a bit, but use a higher threshold to compensate
    cback = findbackmin(dfc)
    thresh = cback * (1.0 + fracabove)

    # Treat one trajectory at a time. Drop the TIME and FRAME columns
    # Use Dicts to clarify the column -> activ time association
    filterednames = filter(isnumeric, names(dfc))
    activtimes = Array{Union{Float64, Missing}}(missing, length(filterednames))
    activtimes = Dict(zip(filterednames, activtimes))
    activfluo = Array{Union{Float64, Missing}}(missing, length(filterednames))
    activfluo = Dict(zip(filterednames, activfluo))
    tpoints = dfc.TIME
    
    # Find the jump in each trajectory
    for col in DF.names(dfc)
        # Ignore TIME and FRAME columns
        if tryparse(Int64, col) !== nothing
            t, y = findjump_thresh(tpoints, dfc[:, col], thresh)
            activtimes[col] = t
            activfluo[col] = y
        end
    end
    # Turn into DataFrames
    activtimes = DF.DataFrame(activtimes)
    activfluo = DF.DataFrame(activfluo)
    return activtimes, activfluo, thresh
end


"""
Apply findactivtimes_thresh to a list of datasets, return the list of activation times and durations
"""
function prepalldsets(fpaths; ch="MEAN_INTENSITY_CH2", filterpms=[50, 30.0], fracabove=0.5, disp=true)
    tactiv_alldsets = []
    durations_alldsets = []
    for fpath in fpaths
        dfc = preptrajdata(fpath; ch=ch, filterpms=filterpms)
        duration = maximum(dfc.TIME)
        tactiv, _, _ = findactivtimes_thresh(dfc; fracabove=fracabove, disp=disp)
        push!(tactiv_alldsets, tactiv)
        push!(durations_alldsets, duration)
    end
    fnames = collect(basename(fp) for fp in fpaths)
    nicenames = cleannames(fnames)[1]
    return tactiv_alldsets, durations_alldsets, nicenames
end


"""
Similar but only read and filter trajectories, 
do not infer activation times. Useful to vary the fluorescence threshold. 
"""
function loadalldsets(fpaths; ch="MEAN_INTENSITY_CH2", filterpms=[50, 30.0], disp=true)
    dfc_alldsets = []
    for fpath in fpaths
        dfc = preptrajdata(fpath; ch=ch, filterpms=filterpms)
        push!(dfc_alldsets, dfc)
    end
    fnames = collect(basename(fp) for fp in fpaths)
    nicenames = cleannames(fnames)[1]
    return dfc_alldsets, nicenames
end


""" 
Now, given pre-loaded DataFrames, process tracks to activation times
"""
function threshalldsets(all_dfcs, nicenames, fracabove; disp=true)
    tactiv_alldsets = []
    durations_alldsets = []
    for i in 1:1:length(nicenames)
        dfc = all_dfcs[i]
        duration = maximum(dfc.TIME)
        tactiv, _, _ = findactivtimes_thresh(dfc; fracabove=fracabove, disp=disp)
        push!(tactiv_alldsets, tactiv)
        push!(durations_alldsets, duration)
    end
    return tactiv_alldsets, durations_alldsets
end


### CDF PREPARATION ###

"""
Turn a list of activation times into an array of distinct CDF values
"""
function uniquecdf(t; addzero=false, makeunique=false)
    ngoodpoints = length(t)
    if addzero  # for plotting really
        t = vcat([0.0], t)  # first point is zero
    end

    # Increment cumul 
    if makeunique
        tu = unique(t)
        # Point i in cumul is the number of elements in t
        # smaller or equal to the time at tu[i]
        cumul = zeros(Float64, length(tu))
        iu = 1  # position in array of unique times
        for i in 1:1:ngoodpoints
            # Add 1 to current position iunique if t[i] is not yet larger
            if t[i] <= tu[iu]
                cumul[iu] += 1.0
                # Stay at same iu, next t could still be in this bin
            # There is surely at least one time point smaller or equal to the
            # next unique time, so assume we are not overshooting the next tu[iu+1]
            elseif iu + 1 <= length(cumul)  # Make sure we can go to iu+1
                cumul[iu + 1] = cumul[iu] + 1.0
                iu += 1
            else  # This shouldn't happen. 
                error("Loop went beyond the limits of the unique activation times list")
            end
        end
    else
        tu = t
        cumul = collect(range(1, length(tu)))
    end
    return tu, cumul
end


"""
Prepare empirical cumulative distribution function
from a list of activation times.
"""
function prepcdf(tactiv, cdfthresh, duration)
    # 1. Extract a sorted list of activation times
    t = sort(collect(skipmissing(tactiv[1, :])))
    
    # For normalization, count all cells (including those never activated)
    ncells = length(tactiv[1, :])

    # 2. remove duplicate experimental times, 
    # compute the cumulative number after each unique time
    tu, cumul = uniquecdf(t; addzero=false, makeunique=false)
    # Normalize by total number of cells
    cumul = cumul / ncells

    # 3. Get the upper limit of the early time, up to F_T(t) = cdfthresh
    # For cases with very few points (because activ threshold is large) 
    # or all points being activated immediately (so 1-2 points before cdf=0.5)
    # fit to all available points minus one
    ilim = findfirst(x -> (x !== missing && x >= cdfthresh), cumul)
    cdfthresh2 = cdfthresh
    if !isnothing(ilim)
        tlim = tu[ilim]
    else
        tlim = 0.0
    end
    # If too few points are included
    while (isnothing(ilim) || (ilim < 3) || tlim < 1e-6) && (cdfthresh2 < 1.0)
        cdfthresh2 += 0.1
        ilim = findfirst(x -> (x !== missing && x >= cdfthresh2), cumul)
        if !isnothing(ilim)
            tlim = tu[ilim]
        else
            tlim = 0.0
        end
    end
    # If the limit is above the largest fraction of active cells in the data
    if (isnothing(ilim) || (ilim < 3) || tlim < 1e-6)
        ilim = length(cumul)  # include all points
        tlim = duration
    end
    
    return tu, cumul, ilim, ncells, tlim
end

