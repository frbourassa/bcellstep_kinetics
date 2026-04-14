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


### MODELS PDFs AND CDFs ###

"""
For observed k fit, showing there may be a cell activation-limited step
"""
function expocdf(t::Array, rate::Float64; t0::Float64=0.0)
    rtt0 = clamp!(rate .* (t .- t0), 0.0, Inf)
    cdf = 1.0 .- exp.(-rtt0)
    return cdf
end


"""
One-hit model with cell activation
Concentration is in pM, kappa is relative to that. 
"""
function onehit_pdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = t .- t0
    factor = k2 * kc / (k2 - kc)
    fctpart = exp.(-kc .* tt0) .- exp.(-k2 .* tt0)
    pdf = factor .* fctpart
    return pdf
end


"""
CDF of the 1-hit model
"""
function onehit_cdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = t .- t0
    kdiff = (k2 - kc)
    fctpart = kc .* exp.(-k2 .* tt0) .- k2 .* exp.(-kc .* tt0)
    cdf = 1.0 .+ fctpart ./ kdiff
    return cdf
end


"""
Complementary CDF of the 1-hit model
"""
function onehit_ccdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = t .- t0
    kdiff = (k2 - kc)
    #fctpart = kc .* exp.(-kc .* tt0) .- k2 .* exp.(-k2 .* tt0)
    fctpart = exp.(-kc .* tt0) .* (k2 .- kc .* exp.(-kdiff .* tt0))
    ccdf = fctpart ./ kdiff
    return ccdf
end


"""
Two-hit model with cell activation step
"""
function twohit_pdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    tt0 = t .- t0
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
function twohit_cdf_float(t::Float64, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    kdiff = k2 - kc
    tt0 = t - t0
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

function twohit_cdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    return collect(map(
        x -> twohit_cdf_float(x, conc, kappa, k2; t0=t0), t))
end


"""
Complementary CDF for the 2-hit model, 
coded in a way that prevents 1-1 large cancellation errors. 
"""
function twohit_ccdf(t::Array, conc::Float64, 
        kappa::Float64, k2::Float64; t0::Float64=0.0)
    kc = kappa * conc
    kdiff = k2 - kc
    tt0 = t .- t0
    prefact = k2 * kc / kdiff
    fact23 = kc / kdiff
    ekct0 = exp.(-kc .* tt0)
    term1 = ekct0 .* (1.0 .+ kc .* tt0) ./ kc
    term2 = exp.(-k2 .* tt0) ./ k2  .- ekct0 ./ kc  
    ccdf = prefact .* (term1 .+ fact23 .* term2)
    return ccdf
end


### LOG-LIKELIHOOD FUNCTIONS AND DERIVATIVES ###

"""
log-likelihood for 1 dataset for the 1-hit model
"""
function onehit_loglhood_1dset(p, times, conc, ncell, klim0, klim1; te=nothing)
    kappa, k2, t0 = p
    if te === nothing
        te = times[min(klim1, length(times))]
    end
    if (kappa < 0.0) || (k2 < 0.0) || (times[1] < t0) 
        ll = -1e8
    else
        # Note that ncell includes those never activated, 
        # whereas times only includes cells eventually activated
        nc0 = klim0 - 1  # number of cells with t_activ = 0
        nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
        lowerlim = max(1, klim0-1)
        ll = nc0 * log(onehit_cdf(times[lowerlim:lowerlim], conc, kappa, k2; t0=t0)[1])
        ll += sum(log.(onehit_pdf(times[klim0:klim1-1], conc, kappa, k2; t0=t0)))
        ll += (ncell - nc0 - nc1) * log(onehit_ccdf([te], conc, kappa, k2; t0=t0)[1])
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
    ndsets = length(concs)
    ll_total = 0.0
    for n in 1:1:ndsets
        p = [p_all[1], p_all[2], p_all[2 + n]]  # kappa, k2, t0 for dataset n
        ll = onehit_loglhood_1dset(p, times_all[n], concs[n], ncells[n], 
                        klim0s[n], klim1s[n]; te=te_all[n])
        ll_total += ll  # no weighting for now 
    end
    return ll_total
end


"""
log-likelihood for 1 dataset for the 1-hit model
"""
function twohit_loglhood_1dset(p, times, conc, ncell, klim0, klim1; te=nothing)
    kappa, k2, t0 = p
    if te === nothing
        te = times[min(klim1, length(times))]
    end
    if (kappa < 0.0) || (k2 < 0.0) || (times[1] < t0)
        ll = -1e8
    else
        # Note that ncell includes those never activated, 
        # whereas times only includes cells eventually activated
        nc0 = klim0 - 1  # number of cells with t_activ = 0
        nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
        lowerlim = max(1, klim0-1)
        ll = nc0 * log(twohit_cdf(times[lowerlim:lowerlim], conc, kappa, k2; t0=t0)[1])
        # Two-hit pdf also has large cancellation issues
        # This model is a numerical nightmare. 
        ll += sum(log.(twohit_pdf(times[klim0:klim1-1], conc, kappa, k2; t0=t0)))
        ll += (ncell - nc0 - nc1) * log(twohit_ccdf([te], conc, kappa, k2; t0=t0)[1])
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
    ndsets = length(concs)
    ll_total = 0.0
    for n in 1:1:ndsets
        p = [p_all[1], p_all[2], p_all[2 + n]]  # kappa, k2, t0 for dataset n
        ll = twohit_loglhood_1dset(p, times_all[n], concs[n], ncells[n], 
                        klim0s[n], klim1s[n]; te=te_all[n])
        ll_total += ll  # no weighting for now 
    end
    return ll_total
end


### RMS ERROR COMPUTATION ###

"""
Compute the root-mean-squared error of the fitted cumulative distribution 
function across all datasets.

Arguments:
- p_all: fitted parameters array [kappa, k2, t0_1, t0_2, ..., t0_n]
  where kappa and k2 are shared across all datasets,
  and t0_n is the activation time offset for dataset n.
- tunique_all: list of unique time arrays for each dataset
- cumuls_all: list of empirical cumulative distributions for each dataset
- ilim0s_all: list of lower indices (first point with t > 0) for each dataset
- ilim1s_all: list of upper indices (last point to include) for each dataset
- concs_all: list of concentrations for each dataset
- cdffct: the CDF function to use (e.g., onehit_cdf, twohit_cdf)

Returns:
- rmserr: the root-mean-squared error computed over all datasets combined
"""
function compute_rms_all_dsets(p_all, tunique_all, cumuls_all, ilim0s_all, 
                                ilim1s_all, concs_all, cdffct)
    ndsets = length(tunique_all)
    sres_total = 0.0
    npts_total = 0
    
    kappa = p_all[1]
    k2 = p_all[2]
    
    for n in 1:ndsets
        tu = tunique_all[n]
        cumul = cumuls_all[n]
        ilim0 = ilim0s_all[n]
        ilim1 = ilim1s_all[n]
        conc = concs_all[n]
        t0 = p_all[2 + n]  # t0 for dataset n
        
        # Determine starting index for RMS calculation
        # Prevent issues if there is no point with t_activ > 0
        rms_i = (ilim1 > ilim0) ? ilim0 : 1
        
        # Compute the fitted CDF for this dataset
        cdfestim = cdffct(tu[rms_i:ilim1], conc, kappa, k2; t0=t0)
        
        # Accumulate squared residuals
        sres_total += sum((cumul[rms_i:ilim1] .- cdfestim).^2)
        npts_total += (ilim1 - rms_i + 1)
    end
    
    # Compute overall RMS error
    rmserr = sqrt(sres_total / npts_total)
    return rmserr
end


### MAIN FITTING FUNCTIONS ###

function fitmodel(tactiv_alldsets, dsetnames, durations, cdfthresh, loglfct, cdffct)
    # kappa estimate: diffusion-limited, 0.4 pM^-1 s^-1, 
    # times default concentration scale of 1 pM. 
    # k2 estimate: fast to begin with
    lower = [1e-4, 1e-3]  # kappa, k2
    upper = [20.0, 20.0]
    init_p = [0.4, 1.0]
    
    # For each dataset in the list of datasets, prepare the cdf, 
    # numbers of cells, concentrations, and limits for the fit.
    conc_scale = 1.0  # pM
    n_dsets = length(tactiv_alldsets)
    concs_all = []
    ncell_all = []
    k0_all = []
    k1_all = []
    te_all = []
    tunique_all = []
    cumuls_all = []
    for i in 1:1:n_dsets
        conc = concfromstr(dsetnames[i], "pM", ' ') / conc_scale
        # Prepare a unique list of cell number increase times to fit.
        tactiv = tactiv_alldsets[i]
        tu, cumul, ilim1, ncells, tlim = prepcdf(tactiv, cdfthresh, durations[i])
        # Find first cell with t > 0: use CDF term instead of pdf in log-likelihood
        # for cells before that, whose activation time could be before t = 0. 
        ilim0 = findfirst(tu .> 0.0)
        push!(tunique_all, tu)
        push!(cumuls_all, cumul)
        push!(concs_all, conc)  # for now, all datasets at 1 pM
        push!(ncell_all, ncells)
        push!(k0_all, ilim0)
        push!(k1_all, ilim1)
        push!(te_all, tlim)
        # Limits for parameter t0 for each dataset: first recorded time point
        push!(lower, -20.0)
        push!(upper, tu[1]-1e-3)
        push!(init_p, -0.1)
    end

    # Minimize -log likelihood
    cost = p_all -> -loglfct(p_all, tunique_all, concs_all, ncell_all, 
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
    kappa, k2 = fit_params[1:2]
    t0s = fit_params[3:end]
    flag = Optim.converged(results)
    if !flag
        println("Did not converge for cdfhresh = ", cdfthresh)
        println("k1_all = ", k1_all, " , ncells = ", ncell_all, " , init_p = ", init_p)
    end

    # RMS error on the cdf across all datasets
    rmserr = compute_rms_all_dsets(fit_params, tunique_all, cumuls_all, 
                                    k0_all, k1_all, concs_all, cdffct)
    
    # Use minus log-likelihood instead, should be minimized
    #rmserr = cost([slope, t0])

    #=
    # Compute model evidence p(D) = \int p(D|\theta)p(\theta) d\theta
    # by numerical integration of the likelihood over the flat prior
    # in the parameter boundaries we chose for \theta = (k, t0).
    lhood = theta -> exp(loglhood_linear(theta, tu, ilim1, ncells, ilim0))
    # The return value of hcubature is a tuple (I, E) 
    # of the estimated integral I and an estimated error E
    evid, evid_err = hcubature(lhood, lower, upper)
    rmserr = 1.0/evid  # inverse evidence should be minimized
    =#
    result = Array{Float64}(vcat(fit_params, [rmserr]))

    return result
end


function (@main)(_)

    t0 = -0.001
    trange = collect(0.0:0.01:20.0)
    kappa = 0.4557859788980265
    conc = 1.0
    k2 = 0.45591241085513795
    pdfrange1 = onehit_pdf(trange, conc, kappa, k2; t0=t0)
    pdfrange2 = twohit_pdf(trange, conc, kappa, k2; t0=t0)
    cdfrange1 = onehit_cdf(trange, conc, kappa, k2; t0=t0)
    cdfrange2 = twohit_cdf(trange, conc, kappa, k2; t0=t0)
    p = pl.plot(trange, pdfrange1, label="pdf 1", legend=:topright)
    pl.plot!(p, trange, pdfrange2, label="pdf 2")
    pl.plot!(p, trange, cdfrange1, label="1", legend=:topright)
    pl.plot!(p, trange, cdfrange2, label="2")
    pl.savefig(p, "temp.png")
    display(p)
    return 0
end
