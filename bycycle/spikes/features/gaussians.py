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
    r_squared : 1d array
        R-squared values from fits.
    """

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    params = np.zeros((len(df_features), 10))
    r_squared = np.zeros(len(df_features))

    indices = [*range(len(df_features))]

    # Compute features in parallel
    with Pool(processes=n_jobs) as pool:

        mapping = pool.imap(partial(_compute_gaussian_features, df_features=df_features,
                                    sig=sig, fs=fs, maxfev=2000, tol=tol, n_gaussians=n_gaussians),
                            indices)

        results = list(progress_bar(mapping, progress, len(df_features)))

    # Unpack results
    params = np.array([result[0] for result in results])
    r_squared = np.array([result[1] for result in results])

    return params, r_squared


def _compute_gaussian_features(index, df_features=None, sig=None,
                               fs=None, maxfev=None, tol=1.49e-6, n_gaussians=None):
    """Compute gaussian features for one cycle."""

    start = df_features.iloc[index]['sample_start'].astype(int)
    end = df_features.iloc[index]['sample_end'].astype(int)

    sig_cyc = sig[start:end+1]
    times_cyc = np.arange(0, len(sig_cyc)/fs, 1/fs)

    # Initial parameter estimation
    centers, stds, alphas, heights, sigmoid_max, sigmoid_growth, sigmoid_mid = \
        estimate_params(df_features, sig, fs, index, n_gaussians)

    # Height and std must be > 0
    lower_heights = [0.00001 if height - 10 <= 0 else height - 10 for height in heights]
    upper_heights = [height+10 for height in heights]
    lower_heights[0] = heights[0] - 10
    upper_heights[0] = heights[0] + 10

    lower_stds = [0.00001 if std - 2.5 < 0 else std - 2.5 for std in stds]
    upper_stds = [std + 2.5 for std in stds]

    lower_alphas = [-25 for std in stds]
    upper_alphas = [25 for std in stds]

    lower_centers = [0, .25] if len(centers) == 2 else [0, 0, .25]
    upper_centers = [1, 1.25] if len(centers) == 2 else [1, 1, 1.25]

    upper_max = sig_cyc.max() - sig_cyc.min()

    # Organize bounds and guess
    bounds = (
        np.array([*lower_centers, *lower_stds, *lower_alphas, *lower_heights, 0, -1, 0]),
        np.array([*upper_centers, *upper_stds, *upper_alphas, *upper_heights, upper_max, 1, 1])
    )

    guess = [*centers, *stds, *alphas, *heights, sigmoid_max, sigmoid_growth, sigmoid_mid]

    try:
        # Fit double gaussian
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        params, _ = curve_fit(sim_action_potential, times_cyc, sig_cyc,
                              p0=guess, bounds=bounds, ftol=tol, xtol=tol, maxfev=maxfev)
    except:
        # Raise warning for failed fits
        warn_str = "Failed fit for index {idx}.".format(idx=index)
        warnings.warn(warn_str, RuntimeWarning)
        params = [np.nan] * len(guess)

    if np.isnan(params[0]):
        r_squared = np.nan
    else:
        # Calculate r-squared
        residuals = sig_cyc - sim_action_potential(times_cyc, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((sig_cyc - np.mean(sig_cyc))**2)

        r_squared = 1 - (ss_res / ss_tot)

    return params, r_squared


def estimate_params(df_features, sig, fs, index, n_gaussians):
    """Initial double gaussian parameter estimates.

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
    n_gaussian : {2, 3}
        The number of gaussians to fit each spike.

    Returns
    -------
    centers : tuple of float
        Gaussian centers, as a proportion of the spike length.
    stds :  tuple of float
        Gaussian widths, as a proportion of the spike length.
    alphas : tuple of float
        Gaussian skews.
    heights: tuple of float
        Gaussian heights, in units of sig.
    sigmoid_max : float
        Height of the sigmoid curve.
    sigmoid_growth : float
        Stepness or growth rate of the sigmoid curve.
    sigmoid_mid:
        Midpoint of the sigmoid curve.
    """

    # Get sample indices
    start = df_features.iloc[index]['sample_start'].astype(int)
    end = df_features.iloc[index]['sample_end'].astype(int)
    sample_trough = df_features.iloc[index]['sample_trough'].astype(int)
    sample_next_peak = df_features.iloc[index]['sample_next_peak'].astype(int)
    sample_last_peak = start + np.argmax(sig[start:sample_trough])
    volt_next_peak = df_features.iloc[index]['volt_peak']

    # Get signal and time
    sig_cyc = sig[start:end+1]
    cyc_len = len(sig_cyc)

    # Estimate height
    height0 = -(np.mean((sig[start], sig[end])) - \
        df_features.iloc[index]['volt_trough'].astype(int))

    height0 = height0 if height0 < 0 else 0.00001

    # Estimate center
    trough_loc = sample_trough - start
    center0 = trough_loc / cyc_len

    # Estimate standard deviation
    sample_decay = df_features.iloc[index]['sample_decay'].astype(int)
    sample_rise = df_features.iloc[index]['sample_rise'].astype(int)

    sig_trough = sig[sample_decay:sample_rise+1]
    std0 = len(sig_trough) / (2 * len(sig_cyc))

    if n_gaussians == 2:

        height1 = volt_next_peak - sig[end]
        height1 = height1 if height1 > 0 else 0.00001

        next_peak_loc = sample_next_peak - start
        center1 = next_peak_loc / cyc_len

        if np.argmax(sig[start:sample_rise+1]) == 0:
            std1 = sig[sample_decay:].std() / (2 * cyc_len)
        else:
            std1 = np.mean((sig[:sample_decay+1].std(),
                            sig[sample_rise:].std()))  / (2 * cyc_len)

        # Organize params
        centers = (center0, center1)
        stds = (std0, std1)
        alphas = (0, 0)
        heights = (height0, height1)

    elif n_gaussians == 3:

        if np.argmax(sig[start:sample_trough]) == 0:
            height1 = 0.00001
        else:
            height1 = np.max(sig[start:sample_trough]) - sig[start]

        height2 = volt_next_peak - sig_cyc[-1]
        height2 = height2 if height2 > 0 else 0.00001

        center1 = (sample_last_peak - start) / cyc_len
        center2 = (sample_next_peak - start) / cyc_len

        std1 = sig[start:sample_decay+1].std() / (2 * cyc_len)
        std2 = sig[sample_rise:].std() / (2 * cyc_len)

        # Organize params
        centers = (center0, center1, center2)
        stds = (std0, std1, std2)
        alphas = (0, 0, 0)
        heights = (height0, height1, height2)

    sigmoid_max = sig[end] - sig[start]
    sigmoid_growth = .5
    sigmoid_mid = .5

    if sigmoid_max < 0:
        sigmoid_max *= -1
        sigmoid_growth *= -1

    return centers, stds, alphas, heights, sigmoid_max, sigmoid_growth, sigmoid_mid


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

    if len(gaussian_params) % 3 == 0:
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

    if isinstance(centers, (tuple, list, np.ndarray)):
        n_params.append(len(centers))
    else:
        centers = repeat(centers)

    if isinstance(stds, (tuple, list, np.ndarray)):
        n_params.append(len(stds))
    else:
        stds = repeat(stds)

    if isinstance(heights, (tuple, list, np.ndarray)):
        n_params.append(len(heights))
    else:
        heights = repeat(heights)

    # Parameter checking
    if len(n_params) == 0:
        raise ValueError('Array-like expected for one of {centers, stds, heights}.')

    for param_len in n_params[1:]:
        if param_len != n_params[0]:
            raise ValueError('Unequal lengths between two or more of {centers, stds, heights}')

    # Initialize cycle array
    n_samples = int(np.ceil(n_seconds * fs))

    n_params = n_params[0]

    cycle = np.zeros((n_params, n_samples))

    # Simulate
    for idx, (center, std, alpha, height) in enumerate(zip(centers, stds, alphas, heights)):
        cycle[idx] = _sim_skewed_gaussian_cycle(n_seconds, fs, center, std, alpha, height)

    cycle = np.sum(cycle, axis=0)

    return cycle


def _sim_sigmoid(xs, maximum, growth, mid):

    mid = xs[round((len(xs)-1) * mid)]

    return maximum / (1.0 + np.exp(-growth*(xs-mid)))
