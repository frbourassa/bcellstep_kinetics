""" Functions to load and preprocess microscopy tracks data """

using DataFrames: DataFrames as DF  # all methods available as DF.method
using CSV: CSV  # all methods available as CSV.method
using Optim: Optim, optimize, IPNewton
using LinearAlgebra: UniformScaling, diag, diagm, svd


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


### FUNCTIONS TO LOAD AND PREPROCESS TRACKS DATASETS ###

function gettrajs(df, colname)
    # Not so easy to map a function to a column, makes me miss pandas 
    # (and I don't even like pandas!)
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
Estimate the background very safely: min value in each trajectory, median
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
    # Don't care about other methods because anyways we will try varying
    # One thing to try: different threshold for each track, adapted to its
    # dynamical range, or at least rescaling the tracks to have a common
    # minimum. But that creates issues for tracks active at time 0. 

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


### EXPONENTIAL CDF FITTING FUNCTION ###
# Turn a list of activation times into an array of distinct CDF values
function uniquecdf(t; addzero=false, makeunique=false)
    ngoodpoints = length(t)
    if addzero  # for plotting really
        t = vcat([0.0], t)  # first point is zero
    end
    if makeunique
        tu = unique(t)
    else
        tu = t
    end

    # Increment cumul 
    if makeunique
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
        cumul = collect(range(1, length(tu)))
    end
    return tu, cumul
end


function expocdf(t::Array, rate::Float64; t0::Float64=0.0)
    rtt0 = clamp!(rate .* (t .- t0), 0.0, Inf)
    cdf = 1.0 .- exp.(-rtt0)
    return cdf
end

function twostepcdf(t::Array, rate::Float64; t0::Float64=0.0)
    rt = clamp!(rate .* (t .- t0), 0.0, Inf)
    cdf = 1.0 .- exp.(-rt) .* (1.0 .+ rt)
    return cdf
end


"""
Prepare empirical cumulative distribution function
from a list of activation times.
"""
function prepcdf(tactiv, cdfthresh)
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
    end
    
    return tu, cumul, ilim, ncells
end


### LOG-LIKELIHOOD FUNCTIONS AND DERIVATIVES

function loglhood_linear(p, times, klim1, ncell, klim0)
    k, t0 = p
    nc0 = klim0 - 1  # number of cells with t_activ = 0
    nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
    if k < -10  #(times[1] < t0) || (k <= 0)
        ll = -1e8
    else
        # Number of cells should include those never activated
        # whereas times only includes cells eventually activated
        nactiv = length(times)
        te = times[min(klim1, nactiv)]  # last considered time point
        ll = nc1 * log(k) - (ncell - nc0 - nc1) * k * (te - t0)
        if nc0 > 0
            ll += nc0 * log(expocdf([0.0], k; t0=t0)[1])
        end
        # Loop over cells with 0 < t_activ < tmax
        for nn in klim0:1:klim1
            ll -= k * (times[nn] - t0)
        end
    end
    return ll
end


function deriv_minuslogl_linear!(dp, p, times, klim1, ncell, klim0)
    k, t0 = p
    nc0 = klim0 - 1  # number of cells with t_activ = 0
    nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
    nactiv = length(times)
    te = times[min(klim1, nactiv)]  # last considered time point
    expkt0 = exp(k*t0)
    if nc0 > 0
        common = nc0 * expkt0 / (1.0 - expkt0)
    else
        common = 0.0 
    end
    dp[1] = -nc1 / k + (ncell - nc0 - nc1) * (te - t0)
    dp[1] += common * t0
    dp[2] = -(ncell - nc0) * k + common * k
    for nn in klim0:1:klim1
        dp[1] += (times[nn] - t0)
    end
    return dp
end


function hess_minuslogl_linear!(mat, p, times, klim1, ncell, klim0)
    k, t0 = p
    nc0 = klim0 - 1  # number of cells with t_activ = 0
    nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
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



### Quadratic model
# Now includes F_T(0) for cells activated at (or possibly prior to) t = 0
function loglhood_quadratic(p, times, klim1, ncell, klim0)
    k, t0 = p
    nc0 = klim0 - 1  # number of cells with t_activ = 0
    nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
    if (times[1] < t0) || (k <= 0)
        ll = -1e8
    else
        nactiv = length(times)
        te = times[min(klim1, nactiv)]  # last considered time point
        ktet0 = k * (te - t0)
        ll = nc1 * log(k) + (ncell - nc0 - nc1) * (log(1.0 + ktet0) - ktet0)
        if nc0 > 0
            ll += nc0 * log(twostepcdf([0.0], k; t0=t0)[1])
        end
        # Loop over cells with 0 < t_activ < tmax
        for nn in klim0:1:klim1
            ktnt0 = k * (times[nn] - t0)
            ll += (log(ktnt0) - ktnt0)
        end
    end
    return ll
end


function deriv_minuslogl_quadratic!(dp, p, times, klim1, ncell, klim0)
    k, t0 = p
    nc0 = klim0 - 1  # number of cells with t_activ = 0
    nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
    nactiv = length(times)
    te = times[min(klim1, nactiv)]  # last considered time point
    ktet0 = k * (te - t0)
    if nc0 > 0
        expkt0 = exp(k*t0)
        common = nc0 * expkt0 * k*t0 / (twostepcdf([0.0], k; t0=t0)[1])
    else
        common = 0.0 
    end
    dp[1] = (2 * nc1 - (ncell - nc1 - nc0) * ktet0^2 / (1.0 + ktet0)) / k
    dp[1] = dp[1] + common * t0
    dp[2] = (ncell - nc0 - (ncell - nc1 - nc0) / (1.0 + ktet0)) * k
    dp[2] = dp[2] + common * k
    for nn in klim0:1:klim1
        tnt0 = times[nn] - t0
        dp[1] -= tnt0
        dp[2] -= 1.0 / tnt0
    end
    dp[1] = -dp[1]
    dp[2] = -dp[2]
    return dp
end


function hess_minuslogl_quadratic!(mat, p, times, klim1, ncell, klim0)
    k, t0 = p
    nc0 = klim0 - 1  # number of cells with t_activ = 0
    nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
    nactiv = length(times)
    te = times[min(klim1, nactiv)]  # last considered time point
    ktet0 = k*(te - t0)
    frac = (ncell - nc0 - nc1) / (1.0 + ktet0)^2
    kt0 = k * t0
    expkt0 = exp(kt0)
    if nc0 > 0
        common = nc0 * expkt0 / (twostepcdf([0.0], k; t0=t0)[1])^2
    else
        common = 0.0
    end
    mat[1, 1] = (2.0*klim0 + frac * ktet0^2) / k^2
    mat[1, 1] -= common * t0*t0 * (1.0 + kt0 - expkt0)
    mat[1, 2] = -(ncell - nc0 - frac)
    mat[1, 2] -= common * k * t0 * (2.0 * (1.0 - expkt0) + kt0*(1.0 + expkt0))
    mat[2, 1] = mat[1, 2]
    mat[2, 2] = frac * k^2
    mat[2, 2] -= common * k*k * (1.0 + kt0 - expkt0)
    for nn in klim0:1:klim1
        mat[2, 2] += 1.0 / (times[nn] - t0)^2
    end
    return mat
end


### MAIN FITTING FUNCTIONS

# First try global maximum likelihood up to a predetermined validity period, 
# just to see if the ML approach makes sense at all. 
function fitcdfslope(tactiv, cdfthresh)
    # Prepare a unique list of increase times to fit.
    tu, cumul, ilim1, ncells = prepcdf(tactiv, cdfthresh)
    tlim1 = tu[ilim1]
    # Find first cell with t > 0: use CDF term instead of pdf in log-likelihood
    # for cells before that, whose activation time could be before t = 0. 
    ilim0 = findfirst(tu .> 0.0)

    # To minimize a cost function with extra args, wrap it as
    # vars -> Log_Likelihood(x, y, vars[1:nvar], vars[nvar+1])
    # equivalent to lambda in Python
    # slope estimate: f / t. 
    lower = [1e-6, -20.0]  # slope, t0 (in s?)
    upper = [20.0 * cdfthresh / (tlim1 + 0.1), tu[1]-1e-6]
    init_p = [cdfthresh / (tlim1 + 0.1), -0.1]
    #println("Initialized parameters, bounds are lower = ", lower, ", upper = ", upper)
    
    # Minimize -log likelihood
    cost = p -> -loglhood_linear(p, tu, ilim1, ncells, ilim0)
    # The grad and hess functions must modify G, H in-place
    # and this must already be the derivatives of -log likelihood
    grads! = (G, p) -> deriv_minuslogl_linear!(G, p, tu, ilim1, ncells, ilim0)
    hess! = (H, p) -> hess_minuslogl_linear!(H, p, tu, ilim1, ncells, ilim0)
    inner_optimizer = IPNewton()  # or NelderMead()
    #println("Start linear fit with data points: ", tu)
    
    results = optimize(cost, grads!, hess!,
                lower, upper, 
                init_p, inner_optimizer
            )
    #println("Finished linear fit optimization")

    # Get the results
    slope, t0 = Optim.minimizer(results)
    flag = Optim.converged(results)
    if !flag
        println("Did not converge for cdfhresh = ", cdfthresh)
        println("ilim1 = ", ilim1, " , ncells = ", ncells, " , init_p = ", init_p)
    end

    # Get variance of these parameter fits. Hessian is not positive definite
    # because the likelihood is monotonically increasing with t0, 
    # so best fit is t0 = boundary, k* derived from dlogL/dk = 0
    # So the uncertainty on the slope is estimated by inverting the FIM = -hessian^-1 element, 
    # while that on t0 is the uncertainty on an experimental time
    # so let's say (0.5 * time bin)^2. 
    minushessmat = zeros(2, 2)
    hess_minuslogl_linear!(minushessmat, [slope, t0], tu, ilim1, ncells, ilim0)
    varislope = 1.0 / minushessmat[1, 1]
    varit0 = (0.5 * (tu[2] - tu[1]))^2

    # RMS error on the cdf. Number of data points used = ilim
    # Prevent issues if there is no point with t_activ > 0
    rms_i = (ilim1 > ilim0) ? ilim0 : 1
    # Compute the fitted CDF
    cdfestim = expocdf(tu[rms_i:ilim1], slope; t0=t0)
    sres = sum((cumul[rms_i:ilim1] .- cdfestim).^2)
    rmserr = sqrt(sres / (ilim1 - rms_i))
    
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
    result = Array{Float64}([slope, varislope, tlim1, rmserr, t0, varit0])

    return result
end


# First try global maximum likelihood up to a predetermined validity period, 
# just to see if the ML approach makes sense at all. 
function fitcdfquad(tactiv, cdfthresh)
    # Prepare a unique list of increase times to fit.
    tu, cumul, ilim1, ncells = prepcdf(tactiv, cdfthresh)
    tlim1 = tu[ilim1]
    # Find first cell with t > 0: use CDF term instead of pdf in log-likelihood
    # for cells before that, whose activation time could be before t = 0. 
    # Number of cells with activation at t=0 is ilim0 - 1
    ilim0 = findfirst(tu .> 0.0)

    # To minimize a cost function with extra args, wrap it as
    # vars -> Log_Likelihood(x, y, vars[1:nvar], vars[nvar+1])
    # equivalent to lambda in Python
    # slope estimate: f / t. 
    lower = [1e-6, -20.0]  # slope, t0 (in s?)
    upper = [50.0 * cdfthresh / (tlim1+0.1), tu[1]-1e-3]
    init_p = [cdfthresh / (tlim1 + 0.1), -0.1]
    # Maximize ML, so minimize -ML
    cost = p -> -loglhood_quadratic(p, tu, ilim1, ncells, ilim0)
    grads! = (G, p) -> deriv_minuslogl_quadratic!(G, p, tu, ilim1, ncells, ilim0)
    hess! = (H, p) -> hess_minuslogl_quadratic!(H, p, tu, ilim1, ncells, ilim0)
    inner_optimizer = IPNewton()
    results = optimize(cost, grads!, hess!,
                lower, upper, 
                init_p, inner_optimizer
            )

    # Get the results
    slope, t0 = Optim.minimizer(results)
    flag = Optim.converged(results)
    if !flag
        println("Did not converge for cdfhresh = ", cdfthresh)
        println("ilim = ", ilim1, " , ncells = ", ncells, " , init_p = ", init_p)
    end

    # Get variance of these parameter fits from the hessian
    hessmat = zeros(2, 2)
    hess_minuslogl_quadratic!(hessmat, [slope, t0], tu, ilim1, ncells, ilim0)
    hessmat = -1.0 .* hessmat
    varimat = covfromhess(hessmat)
    varislope, varit0 = abs.(diag(varimat))

    # RMS error on the cdf. Number of data points used = ilim
    # Prevent issues if there is no point with t_activ > 0
    rms_i = (ilim1 > ilim0) ? ilim0 : 1
    # Compute the fitted CDF
    cdfestim = twostepcdf(tu[rms_i:ilim1], slope; t0=t0)
    sres = sum((cumul[rms_i:ilim1] .- cdfestim).^2)
    rmserr = sqrt(sres / (ilim1 - rms_i))

    result = Array{Float64}([slope, varislope, tlim1, rmserr, t0, varit0])

    return result
end


### Three-step activation model

function threestepcdf(t::Array, rate::Float64; t0::Float64=0.0)
    rt = clamp!(rate .* (t .- t0), 0.0, Inf)
    cdf = 1.0 .- exp.(-rt) .* (1.0 .+ rt .* (1.0 .+ 0.5 .* rt))
    return cdf
end

function loglhood_3step(p, times, klim1, ncell, klim0)
    k, t0 = p
    nc0 = klim0 - 1  # number of cells with t_activ = 0
    nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
    if (times[1] < t0) || (k <= 0)
        ll = -1e8
    else
        nactiv = length(times)
        te = times[min(klim1, nactiv)]  # last considered time point
        ktet0 = k * (te - t0)
        ll = nc1 * log(0.5*k) 
        ll += (ncell - nc0 - nc1) * log(1.0 - threestepcdf([te], k; t0=t0)[1])
        if nc0 > 0
            ll += nc0 * log(threestepcdf([0.0], k; t0=t0)[1])
        end
        # Loop over cells with 0 < t_activ < tmax
        for nn in klim0:1:klim1
            ktnt0 = k * (times[nn] - t0)
            ll += (log(0.5 * ktnt0*ktnt0) - ktnt0)
        end
    end
    return ll
end

function deriv_minuslogl_3step!(dp, p, times, klim1, ncell, klim0)
    k, t0 = p
    nc0 = klim0 - 1  # number of cells with t_activ = 0
    nc1 = klim1 - nc0  # number of cells with 0 < t_activ < tlim
    nactiv = length(times)
    te = times[min(klim1, nactiv)]  # last considered time point
    ktet0 = k * (te - t0)
    kt0 = k * t0
    if nc0 > 0
        expkt0 = exp(k*t0)
        common = nc0 * expkt0 * kt0*kt0 / (2.0*threestepcdf([0.0], k; t0=t0)[1])
    else
        common = 0.0
    end
    poly = 1.0 + ktet0 * (1.0 + 0.5 * ktet0)
    dp[1] = -common * t0 + 3.0 * nc1 / k
    dp[1] -= 0.5 * (ncell - nc1 - nc0) * ktet0^3 / k / poly
    dp[2] = (ncell - nc0) * k - common * k
    dp[2] -= (ncell - nc0 - nc1) * k * (ktet0 + 1.0) / poly
    for nn in klim0:1:klim1
        tnt0 = times[nn] - t0
        dp[1] -= tnt0
        dp[2] -= 2.0 / tnt0
    end
    dp[1] = -dp[1]
    dp[2] = -dp[2]
    return dp
end


# First try global maximum likelihood up to a predetermined validity period, 
# just to see if the ML approach makes sense at all. 
function fitcdf3step(tactiv, cdfthresh)
    # Similar preparation as above
    tu, cumul, ilim1, ncells = prepcdf(tactiv, cdfthresh)
    tlim1 = tu[ilim1]
    ilim0 = findfirst(tu .> 0.0)

    lower = [1e-6, -20.0]  # slope, t0 (in s?)
    upper = [50.0 * cdfthresh / (tlim1+0.1), tu[1]-1e-3]
    init_p = [cdfthresh / (tlim1 + 0.1), -0.1]
    cost = p -> -loglhood_3step(p, tu, ilim1, ncells, ilim0)
    grads! = (G, p) -> deriv_minuslogl_3step!(G, p, tu, ilim1, ncells, ilim0)
    inner_optimizer = IPNewton()  # or NelderMead()
    results = optimize(cost, grads!,
                lower, upper, 
                init_p, inner_optimizer
            )

    # Get the results
    slope, t0 = Optim.minimizer(results)
    flag = Optim.converged(results)
    if !flag
        println("Did not converge for cdfhresh = ", cdfthresh)
        println("ilim = ", ilim1, " , ncells = ", ncells, " , init_p = ", init_p)
    end

    # Hessian not implemented, return unit variance, not used currently anyways
    hessmat = zeros(2, 2) + UniformScaling(1)
    varimat = covfromhess(hessmat)
    varislope, varit0 = abs.(diag(varimat))

    rms_i = (ilim1 > ilim0) ? ilim0 : 1
    # Compute the fitted CDF
    cdfestim = threestepcdf(tu[rms_i:ilim1], slope; t0=t0)
    sres = sum((cumul[rms_i:ilim1] .- cdfestim).^2)
    rmserr = sqrt(sres / (ilim1 - rms_i))

    result = Array{Float64}([slope, varislope, tlim1, rmserr, t0, varit0])

    return result
end



### Utilities for parameter uncertainties

# Adapted from Scipy curve_fit, covariance matrix estimate (Crámer-Rao bound)
# from the pseudo-inverse of the FIM, which is equal to -hessian 
function covfromhess(hess)
    # Do Moore-Penrose inverse, discard zero singular values.
    U, s, V = svd(-1.0 .* hess)
    VT = V'
    threshold = 1e-12 * maximum(size(hess)) * s[1]
    s = s[s .> threshold]
    sinv = diagm(1.0 ./ s)
    VT = VT[1:length(s), 1:end]
    U = U[1:end, 1:length(s)]
    pcov = U * sinv * VT
    return pcov
end