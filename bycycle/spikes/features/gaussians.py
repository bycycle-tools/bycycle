"""Compute gaussian features."""

import warnings
from functools import partial
from itertools import repeat

from multiprocessing import Pool, cpu_count
from bycycle.group.utils import progress_bar

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import stats as st
from bycycle.cyclepoints import find_extrema, find_zerox
from neurodsp.sim.cycles import sim_skewed_gaussian_cycle



###################################################################################################
###################################################################################################


def compute_gaussian_features(df_samples, sig, fs, maxfev=2000, tol=1.49e-6, n_jobs=-1, chunksize=1,
                              progress=None, z_thresh_k=0.5, z_thresh_cond=0.5, rsq_thresh=0.5):
    """Compute gaussian features.

    Parameters
    ----------
    df_samples : pandas.DataFrame
        Contains cyclepoint locations for each spike.
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    maxfev : int, optional, default: 2000
        The maximum number of calls in curve_fit.
    tol : float, optional, default: 10e-6
        Relative error desired.
    n_jobs : int, optional, default: -1
        The number of jobs to compute features in parallel.
    chunksize : int, optional, default: 1
        Number of chunks to split spikes into. Each chunk is submitted as a separate job.
        With a large number of spikes, using a larger chunksize will drastically speed up
        runtime. An optimal chunksize is typically np.ceil(n_spikes/n_jobs).
    progress : {None, 'tqdm', 'tqdm.notebook'}
        Specify whether to display a progress bar. Uses 'tqdm', if installed.
    z_thresh_k : float, optional, default: 0.5
        Potassium (k) current z-score threshold.
    z_thresh_cond : float, optional, default: 0.5
        Conductive current z-score threshold.
    rsq_thresh : float, optional, default: 0.5
        Na current r-squared threshold. Used to stop conductive/K fits in cycles
        with bad Na current fits.

    Returns
    -------
    params : dict
        Fit parameter values.
    """

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    indices = [*range(len(df_samples))]

    # Compute features in parallel
    with Pool(processes=n_jobs) as pool:

        mapping = pool.imap(partial(_compute_gaussian_features_cycle, df_samples=df_samples,
                                    sig=sig, fs=fs, maxfev=maxfev, tol=tol,
                                    z_thresh_k=0.5, z_thresh_cond=0.5, rsq_thresh=0.5),
                            indices, chunksize=chunksize)

        params = list(progress_bar(mapping, progress, len(df_samples)))

    return np.array(params)


def _compute_gaussian_features_cycle(index, df_samples=None, sig=None, fs=None,
                                     f_ranges=(300, 2000), maxfev=2000, tol=1.49e-6,
                                     z_thresh_k=0.5, z_thresh_cond=0.5, rsq_thresh=0.5):
    """Compute gaussian features for one cycle."""

    start = df_samples.iloc[index]['sample_start'].astype(int)
    end = df_samples.iloc[index]['sample_end'].astype(int)
    sample_trough = df_samples.iloc[index]['sample_trough'].astype(int)

    # Adjust samples to start at zero
    sample_trough -= start

    # Get signal and time
    sig_cyc = sig[start:end+1]
    cyc_len = len(sig_cyc)
    times_cyc = np.arange(0, cyc_len/fs, 1/fs)

    # Fit single skewed gaussian to Na current
    na_params, na_gaus = _single_gaus_fit(index, sample_trough, sig_cyc, cyc_len, times_cyc, fs,
                                          extrema_type="trough", maxfev=maxfev, tol=tol)
    if not np.isnan(na_gaus).any():

        # Get Na center and std
        na_center = int(na_params[0]*cyc_len)
        na_std = int(na_params[1]*cyc_len)

        # Determine Na current region
        upper_std = na_center + (2* na_std)
        lower_std = na_center - (2* na_std)

        # Calculate Na current r-squared
        na_rsq = calculate_r_squared(sig_cyc[lower_std:upper_std], na_gaus[lower_std:upper_std])

        # Check if Na r-squared is above threshold
        if na_rsq < rsq_thresh:
            na_rsq = np.nan
            na_params = np.append(na_params, na_rsq)

            k_params = np.array([np.nan] * len(na_params))
            cond_params = np.array([np.nan] * len(na_params))
            warnings.warn("Failed fits for index = " + str(index))

        else:

            na_params = np.append(na_params, na_rsq)

            # Substract Na current gaussian fit
            rem_sig = sig_cyc - na_gaus

            # Split remaining signal into left of Na current (K current)
            #   and right (conductive current)
            rem_sigs, times, z_scores = calculate_side_regions(na_center, rem_sig, times_cyc, fs,
             z_thresh_k, z_thresh_cond)

            side_current_region = zip(rem_sigs, [z_thresh_k, z_thresh_cond], z_scores, times)

            side_current_params = []
            side_current_gaus = []

            for rem_sig, z_thresh, z_score, times in side_current_region:

                if any(z >= z_thresh for z in z_score):
                    # Get peak of remaining signal
                    peak = get_current_peak(rem_sig, fs, f_ranges, z_thresh, z_score)

                    if peak == None:
                        params = np.array([np.nan] * len(na_params))
                        gaus = np.array([np.nan] * len(times))

                    else:
                        # Fit single skewed gaussian to K current
                        params, gaus = _single_gaus_fit(index, peak, rem_sig, len(rem_sig),
                                                        times, fs, extrema_type="peak",
                                                        maxfev=maxfev, tol=tol)

                        # Calculate r-squared
                        rsq = calculate_r_squared(rem_sig, gaus)
                        params = np.append(params, rsq)

                else:
                    params = np.array([np.nan] * len(na_params))
                    gaus =  np.array([np.nan] * len(times))

                side_current_params.append(params)
                side_current_gaus.append(gaus)

            # Unpack results
            k_params, cond_params = side_current_params
            k_gaus, cond_gaus = side_current_gaus

    else:
        na_rsq = np.nan
        na_params = np.append(na_params, na_rsq)

        k_params = np.array([np.nan] * len(na_params))
        cond_params = np.array([np.nan] * len(na_params))
        warnings.warn("Failed fits for index = " + str(index))

    all_params = [*cond_params, *na_params, *k_params]

    return all_params


def estimate_params(extrema, sig_cyc, fs, extrema_type="trough", n_decimals=2):
    """Initial gaussian parameter estimates.

    Parameters
    ----------
    extrema : int
        extrema position (peak or trough) of sig_cyc
    sig_cyc : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    extrema_type : string, optional, default: "trough"
        Type of extrema, trough or peak.
    n_decimals : int, optional, default: 2
        Number of decimals to round parameters to.

    Returns
    -------
    params : 1d array
        Estimated centers, stds, alphas, heights.
    """

    cyc_len = len(sig_cyc)

    centers = []
    stds = []
    heights = []

    # Define parameters
    if extrema_type == "trough":
        height0 =  sig_cyc[extrema] - np.mean(sig_cyc)
    else:
        height0 =  sig_cyc[extrema]

    center0 = extrema / cyc_len

    std0 = _estimate_std(sig_cyc, extrema_type=extrema_type, plot=False)

    centers.append(center0.round(n_decimals))
    stds.append(std0.round(n_decimals))
    heights.append(height0.round(n_decimals))

    # Assume no skew
    alphas = [0] * len(centers)
    params = [*centers, *stds, *alphas, *heights]

    return np.array(params)


def _estimate_bounds(sig_cyc, centers, stds, heights):
    """Estimate parameter's lower and upper bounds."""

    # Define bounds
    lower_heights = [height * .5 if height > 0 else height * 1.5 for height in heights]
    upper_heights = [height * 1.5 if height > 0 else height * .5 for height in heights]

    lower_stds = [std * .5 for std in stds]
    upper_stds = [std * 1.5 for std in stds]

    lower_alphas = [-3 for std in stds]
    upper_alphas = [3 for std in stds]

    lower_centers = [center * .5 for center in centers]
    upper_centers = [center * 1.5 for center in centers]

    upper_max = np.max(sig_cyc) - np.min((sig_cyc[0], sig_cyc[-1]))

    bounds = [
        [*lower_centers, *lower_stds, *lower_alphas, *lower_heights, 0, -1, 0],
        [*upper_centers, *upper_stds, *upper_alphas, *upper_heights, upper_max, 1, 1]
    ]

    return bounds


def _fit_gaussians(xs, ys, guess, tol, maxfev, index, bounds=None):
    """Fit gaussians with scipy's curve_fit."""

    try:
        # Fit gaussians
        warnings.filterwarnings("ignore")
        params, _ = curve_fit(_sim_gaussian_cycle, xs, ys, p0=guess)

    except:
        # Raise warning for failed fits
        warn_str = "Failed fit for index {idx}.".format(idx=index)
        warnings.warn(warn_str, RuntimeWarning)
        params = np.array([np.nan] * len(guess))

    return params


###################################################################################################
###################################################################################################


def _sim_gaussian_cycle(times, *params):
    """Proxy function for compatibility between sim_skewed_gaussian and curve_fit.

    Parameters
    ----------
    times : 1d array
        Time definition of the cycle.
    params : floats
        Variable number of centers, stds, alphas, and heights arguments, respectively. The number
        of these variable parameters determines the number of gaussians simulated. An additional
        three trailing arguments to define a sigmoid baseline as maximum, growth, midpoint.

    Returns
    -------
    sig_cycle : 1d array
        Simulated action potential.
    """
    sing_gaus = sim_skewed_gaussian_cycle(1, len(times), *params)

    return sing_gaus


def _single_gaus_fit(index, extrema, sig_cyc, cyc_len, times_cyc,
                     fs, extrema_type="trough", maxfev=2000, tol=None):
    """Calculate guassian fits for single current  """

    # Initial parameter estimation
    _params = estimate_params(extrema, sig_cyc, fs, extrema_type=extrema_type, n_decimals=2)

    # Initial bound estimation for Na current
    _bounds = _estimate_bounds(sig_cyc, *_params.reshape(4, -1)[[0, 1, 3]])

    # Fit single skewed gaussian
    _params_fit = _fit_gaussians(times_cyc, sig_cyc, _params, tol, maxfev, index, bounds=_bounds)

    if np.isnan(_params_fit).any():
        _gaus = np.array([np.nan] * len(times_cyc))

    else:
        _gaus = sim_skewed_gaussian_cycle(1, cyc_len, *_params_fit)

    return _params_fit, _gaus


def calculate_side_regions(na_center, rem_sig, times_cyc, fs, z_thresh_k, z_thresh_cond):
    """Calculate K current and conductive current regions
       of the signal based on the center of the Na current.
    """

    rem_sig_k = rem_sig[na_center:,]
    rem_sig_cond = rem_sig[:na_center,]

    times_k = times_cyc[na_center:,]
    times_cond = times_cyc[:na_center,]

    # Calculate z scores
    z_score_k = st.zscore(rem_sig_k)
    z_score_cond = st.zscore(rem_sig_cond)

    rem_sigs = [rem_sig_k, rem_sig_cond]
    times = [times_k, times_cond]
    z_scores = [z_score_k,z_score_cond]

    return [rem_sigs, times, z_scores]


###################################################################################################
###################################################################################################

def _estimate_std(spike, extrema_type='trough', plot=False):
    """Estimate std of spike"""

    spike = -spike if extrema_type == 'peak' else spike

    height, height_idx = np.min(spike), np.argmin(spike)
    half_height = height / 2

    right = spike[height_idx:]
    left = np.flip(spike[:height_idx+1])

    if plot:
        plt.plot(-spike if extrema_type=='peak' else spike)
        plt.axvline(height_idx, color='r')

    right_idx = _get_closest(right, spike, half_height)
    left_idx = _get_closest(left, spike, half_height)

    if right_idx == None:
        right_idx = left_idx

    if left_idx == None:
        left_idx = right_idx

    fwhm = (right_idx + left_idx + 1)

    std = fwhm / (2 * len(spike) * np.sqrt(2 * np.log(2)))

    return std


def _get_closest(flank, spike, half_height):

    for idx, volt in enumerate(flank):

        if volt > half_height:

            # Get closest sample left or right of half max location
            closest = np.argmin([volt - half_height,
                                 half_height - flank[idx-1]])

            idx = [idx, idx-1][closest]

            return idx


def get_current_peak(sig, fs, f_ranges, z_thresh, z_score):

    peaks, troughs = find_extrema(sig, fs, f_ranges, first_extrema=None, pass_type='bandpass')

    if len(peaks) == 0:
        return None
    elif len(peaks) > 1:
        #select highest peak
        max_volt = max( (v, i) for i, v in enumerate(sig[peaks]) )[1]
        peak = peaks[max_volt]

    else:
        peak = peaks[0]

    # check if peak is over z score threshold
    if z_score[peak] > z_thresh:
        return peak
    else:
        return None


def calculate_r_squared(sig_cyc, sig_cyc_est):

    residuals = sig_cyc - sig_cyc_est
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((sig_cyc - np.mean(sig_cyc))**2)

    r_squared = 1 - (ss_res / ss_tot)

    return r_squared
