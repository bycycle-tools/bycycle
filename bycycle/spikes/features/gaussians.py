"""Compute double gaussian features."""

import warnings
from functools import partial
from itertools import repeat

from multiprocessing import Pool, cpu_count
from bycycle.group.utils import progress_bar

import numpy as np

from scipy.optimize import curve_fit

###################################################################################################
###################################################################################################


def compute_gaussian_features(df_features, sig, fs, n_gaussians=2,
                              maxfev=2000, tol=1.49e-6, n_jobs=-1, progress=None):
    """Compute double gaussian features.

    Parameters
    ----------
    df_features : pandas.DataFrame
        Dataframe containing shape and burst features for each spike.
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    maxfev : int, optional, default: 2000
        The maximum number of calls in curve_fit.
    tol : float, optional, default: 10e-6
        Relative error desired.
    n_gaussians : {0, 2, 3}
            Fit a n number of gaussians to each spike. If zeros, no gaussian fitting occurs.
    n_jobs : int, optional, default: -1
        The number of jobs to compute features in parallel.
    progress : {None, 'tqdm', 'tqdm.notebook'}
        Specify whether to display a progress bar. Uses 'tqdm', if installed.

    Returns
    -------
    params : dict
        Fit parameter values.
    """

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    indices = [*range(len(df_features))]

    # Compute features in parallel
    with Pool(processes=n_jobs) as pool:

        mapping = pool.imap(partial(_compute_gaussian_features, df_features=df_features,
                                    sig=sig, fs=fs, maxfev=maxfev, tol=tol,
                                    n_gaussians=n_gaussians),
                            indices)

        params = list(progress_bar(mapping, progress, len(df_features)))

    return np.array(params)


def _compute_gaussian_features(index, df_features=None, sig=None,
                               fs=None, maxfev=None, tol=None, n_gaussians=None):
    """Compute gaussian features for one cycle."""

    start = df_features.iloc[index]['sample_start'].astype(int)
    end = df_features.iloc[index]['sample_end'].astype(int)

    sig_cyc = sig[start:end+1]
    times_cyc = np.arange(0, len(sig_cyc)/fs, 1/fs)

    # Initial parameter estimation
    _params = estimate_params(df_features, sig, fs, index, n_gaussians)

    # Set max to zero for single gaussians
    if len(_params) == 7:
        sig_cyc -= sig_cyc.max()

    _bounds = _estimate_bounds(sig_cyc, *_params[:-3].reshape(4, -1)[[0, 1, 3]])

    # First-pass fit
    params = _fit_gaussians(times_cyc, sig_cyc, _params, _bounds, 1e-2, maxfev, index)
    bounds = _estimate_bounds(sig_cyc, *_params[:-3].reshape(4, -1)[[0, 1, 3]])

    # Second-pass fit
    if not np.isnan(params[0]):

        params = _fit_gaussians(times_cyc, sig_cyc, params, bounds, tol, maxfev, index)

        # Insert nans where needed
        params_gaus = params[:-3].reshape(4, -1).T
        params_sigm = params[-3:]

        nan_arr = np.zeros_like(params_gaus[0])
        nan_arr[:] = np.nan

        if n_gaussians == 3 and len(params_gaus) == 2:

            if params_gaus[0][0] > params_gaus[1][0]:
                # No K current
                params_gaus = np.insert(params_gaus, 2, nan_arr, axis=0)
            else:
                # No conductive current
                params_gaus = np.insert(params_gaus, 1, nan_arr, axis=0)

        elif n_gaussians == 3 and len(params_gaus) == 1:
            # No K or conductive current
            params_gaus = np.vstack((params_gaus, [nan_arr, nan_arr]))

        elif n_gaussians == 2 and len(params_gaus) == 1:
            # No K or conductive current
            params_gaus = np.insert(params_gaus, 1, nan_arr, axis=0)

        params = np.array([*params_gaus.T.flatten(), *params_sigm])

    else:
        params = np.zeros((n_gaussians * 4) + 3)
        params[:] = np.nan

    return params


def estimate_params(df_features, sig, fs, index, n_gaussians=3, n_decimals=2):
    """Initial gaussian parameter estimates.

    Parameters
    ----------
    df_features : pandas.DataFrame
        Dataframe containing shape and burst features for each spike.
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    index : int
        The spike index in the 2d array (i.e. the spikes attribute of a Spikes class.

    Returns
    -------
    params : 1d array
        Estimated centers, stds, alphas, heights, and sigmoid max, growth, midpoint.
        The number of centers, stds, alphas, heights varies from one to three.
    bounds : list of list
        Lower and upper bounds for each parameters.
    """

    # Get sample indices
    sample_start = df_features.iloc[index]['sample_start'].astype(int)
    sample_end = df_features.iloc[index]['sample_end'].astype(int)

    sample_trough = df_features.iloc[index]['sample_trough'].astype(int)
    sample_last_peak = df_features.iloc[index]['sample_last_peak'].astype(int)
    sample_next_peak = df_features.iloc[index]['sample_next_peak'].astype(int)
    sample_decay = df_features.iloc[index]['sample_decay'].astype(int)
    sample_rise = df_features.iloc[index]['sample_rise'].astype(int)

    # Adjust samples to start at zero
    sample_trough -= sample_start
    sample_last_peak -= sample_start
    sample_next_peak -= sample_start
    sample_decay -= sample_start
    sample_rise -= sample_start

    # Get signal and time
    sig_cyc = sig[sample_start:sample_end+1]
    cyc_len = len(sig_cyc)

    if sample_last_peak == 0 and sample_next_peak == len(sig_cyc) - 1:
        # No conductive or K current
        currents = ['Na']
        sig_cyc -= np.max(sig_cyc)
    elif sample_last_peak == 0:
        # No conductive current
        currents = ['Na', 'K']
    elif sample_next_peak == len(sig_cyc) - 1:
        # No K current
        currents = ['Na', 'Conductive']
    else:
        # All 3 currents
        currents = ['Na', 'Conductive', 'K']

    centers = []
    stds = []
    heights = []

    # Define Na current estimates
    height0 =  sig_cyc[sample_trough] - np.mean((sig_cyc[0], sig_cyc[-1]))

    center0 = sample_trough / cyc_len

    extrema_idx = np.argmin(sig_cyc)
    extrema = np.min(sig_cyc)

    fwhm = (np.argmin(np.abs(sig_cyc[extrema_idx:] - (extrema * .5))) + extrema_idx) - \
            np.argmin(np.abs(sig_cyc[:extrema_idx] - (extrema * .5)))

    fwhm /= len(sig_cyc)

    fwhm_div = (2 * np.sqrt(2 * np.log(2)))
    std0 = fwhm / fwhm_div

    centers.append(center0.round(n_decimals))
    stds.append(std0.round(n_decimals))
    heights.append(height0.round(n_decimals))

    if 'Conductive' in currents:

        height1 = sig_cyc[sample_last_peak] - np.mean(sig_cyc)
        center1 = sample_last_peak / cyc_len
        std1 = len(sig_cyc[:sample_decay+1]) / (2 * fwhm_div * len(sig_cyc))

        centers.append(center1.round(n_decimals))
        stds.append(std1.round(n_decimals))
        heights.append(height1.round(n_decimals))

    if 'K' in currents:

        height2 = sig_cyc[sample_next_peak] - np.mean(sig_cyc)
        center2 = sample_next_peak / cyc_len
        std2 = len(sig_cyc[sample_rise:]) / (2 * fwhm_div * len(sig_cyc))

        centers.append(center2.round(n_decimals))
        stds.append(std2.round(n_decimals))
        heights.append(height2.round(n_decimals))

    if 'Conductive' in currents and 'K' in currents and n_gaussians == 2:

        center1 = np.mean((center1, center2))
        std1 = std1 + std2
        height1 = np.mean((height1, height2))

        centers = centers[:-2]
        centers.append(center1)
        heights = heights[:-2]
        heights.append(height1)
        stds = stds[:-2]
        stds.append(std1)

    # Assume no skew
    alphas = [0] * len(centers)

    # Sigmoid baseline
    sigmoid_max = (sig_cyc[-1] - sig_cyc[0]) * .5
    sigmoid_growth = .5
    sigmoid_mid = np.argmin(sig_cyc) / cyc_len

    if sigmoid_max < 0:
        sigmoid_max *= -1
        sigmoid_growth *= -1

    params = [*centers, *stds, *alphas, *heights, sigmoid_max, sigmoid_growth, sigmoid_mid]

    return np.array(params)


def _estimate_bounds(sig_cyc, centers, stds, heights):
    """Estimate parameters lower and upper bounds."""

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


def _fit_gaussians(xs, ys, guess, bounds, tol, maxfev, index):
    """Fit gaussians with scipy's curve_fit."""

    try:
        # Fit gaussians
        warnings.filterwarnings("ignore")
        params, _ = curve_fit(sim_action_potential, xs, ys,
                              p0=guess, bounds=bounds, ftol=tol, xtol=tol, maxfev=maxfev)
    except:
        # Raise warning for failed fits
        warn_str = "Failed fit for index {idx}.".format(idx=index)
        warnings.warn(warn_str, RuntimeWarning)
        params = np.array([np.nan] * len(guess))

    return params


def sim_action_potential(times, *params):
    """Proxy function for compatibility between _sim_ap_cycle and curve_fit.

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

    gaussian_params = params[:-3]

    sigmoid_params = params[-3:]

    if len(gaussian_params) == 4:
        pass
    elif len(gaussian_params) % 3 == 0:
        gaussian_params = np.array(gaussian_params).reshape((-1, 3))
    elif len(gaussian_params) % 2 == 0:
        gaussian_params = np.array(gaussian_params).reshape((-1, 2))

    sig_cycle = _sim_ap_cycle(1, len(times), *gaussian_params)

    xs = np.arange(-np.ceil(len(times)/2), np.floor(len(times)/2))

    sigmoid = _sim_sigmoid(xs, *sigmoid_params)

    return sig_cycle + sigmoid


def _sim_gaussian_cycle(n_seconds, fs, std, center=.5, height=1.):
    """Simulate a gaussian cycle."""

    xs = np.linspace(0, 1, int(np.ceil(n_seconds * fs)))
    cycle = np.exp(-(xs-center)**2 / (2*std**2))

    return cycle


def _sim_skewed_gaussian_cycle(n_seconds, fs, center, std, alpha, height=1):
    """Simulate a skewed gaussian cycle."""

    from scipy.stats import norm

    n_samples = int(np.ceil(n_seconds * fs))

    # Gaussian distribution
    cycle = _sim_gaussian_cycle(n_seconds, fs, std, center, height)

    # Skewed cumulative distribution function.
    #   Assumes time are centered around 0. Adjust to center around 0.5.
    times = np.linspace(-1, 1, n_samples)
    cdf = norm.cdf(alpha * ((times - ((center * 2) - 1)) / std))

    # Skew the gaussian
    cycle = cycle * cdf

    # Rescale height
    cycle = (cycle / np.max(cycle)) * height

    return cycle


def _sim_ap_cycle(n_seconds, fs, centers, stds, alphas, heights):
    """Simulate an action potential as the sum of two inverse, skewed gaussians.

    Parameters
    ----------
    n_seconds : float
        Length of cycle window in seconds.
    fs : float
        Sampling frequency of the cycle simulation.
    centers : array-like or float
        Times where the peak occurs in the pre-skewed gaussian.
    stds : array-like or float
        Standard deviations of the gaussian kernels, in seconds.
    alphas : array-like or float
        Magnitiude and direction of the skew.
    heights : array-like or float
        Maximum value of the cycles.

    Returns
    -------
    cycle : 1d array
        Simulated spike cycle.
    """

    # Determine number of parameters
    n_params = []

    params, n_params = _make_iterable([centers, stds, alphas, heights])
    centers, stds, alphas, heights = params

    # Parameter checking
    if len(n_params) != 0:
        for param_len in n_params[1:]:
            if param_len != n_params[0]:
                raise ValueError('Unequal lengths between two or more of {centers, stds, heights}')

    # Simulate
    if len(n_params) > 0:

        n_samples = int(np.ceil(n_seconds * fs))
        cycle = np.zeros((n_params[0], n_samples))

        for idx, (center, std, alpha, height) in enumerate(zip(centers, stds, alphas, heights)):
            cycle[idx] = _sim_skewed_gaussian_cycle(n_seconds, fs, center, std, alpha, height)

        cycle = np.sum(cycle, axis=0)

    else:

        cycle = _sim_skewed_gaussian_cycle(n_seconds, fs, next(centers),
                                           next(stds), next(alphas), next(heights))

    return cycle

def _make_iterable(params):

    n_params = []
    for idx, param in enumerate(params):

        if isinstance(param, (tuple, list, np.ndarray)):
            n_params.append(len(param))
        else:
            params[idx] = repeat(param)

    return params, n_params


def _sim_sigmoid(xs, maximum, growth, mid):

    mid = xs[round((len(xs)-1) * mid)]

    return maximum / (1.0 + np.exp(-growth*(xs-mid)))
