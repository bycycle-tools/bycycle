"""Compute double gaussian features."""

import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
from bycycle.group.utils import progress_bar

import numpy as np

from scipy.optimize import curve_fit

###################################################################################################
###################################################################################################


def compute_gaussian_features(df_features, sig, fs, maxfev=2000, n_jobs=-1, progress=None):
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
                                    sig=sig, fs=fs),
                            indices)

        results = list(progress_bar(mapping, progress, len(df_features)))

    # Unpack results
    params = np.array([result[0] for result in results])
    r_squared = np.array([result[1] for result in results])

    param_labels = ['center0', 'center1', 'std0', 'std1', 'alpha0', 'alpha1',
                    'height0', 'height1', 'shift0', 'shift1']

    params = {k: v for k, v in zip(param_labels, params.transpose())}

    return params, r_squared


def _compute_gaussian_features(index, df_features=None, sig=None, fs=None, maxfev=None):
    """Compute gaussian features for one cycle."""

    start = df_features.iloc[index]['sample_start'].astype(int)
    end = df_features.iloc[index]['sample_end'].astype(int)

    sig_cyc = sig[start:end+1]
    times_cyc = np.arange(0, len(sig_cyc)/fs, 1/fs)

    # Initial parameter estimation
    center0, center1, std0, std1, alpha0, alpha1, height0, height1 = \
        estimate_params(df_features, sig, fs, index)

    # Height and std must be > 0
    height0_lower = 0.00001 if height0 - 10 <= 0 else height0 - 10
    height1_lower = 0.00001 if height1 - 10 <= 0 else height1 - 10
    std0_lower = 0.00001 if std0 - 2.5 < 0 else std0 - 2.5
    std1_lower = 0.00001 if std1 - 2.5 < 0 else std1 - 2.5

    # Oranize bounds and guess
    bounds = (
        np.array([0, 0.25,  std0_lower, std1_lower,  -50,  -50,
                  height0_lower, height1_lower, -20, -20]),
        np.array([1, 1.25,  std0+2.5, std1+2.5, 50, 50, height0+10, height1+10, 20, 20])
    )

    guess = [center0, center1, std0, std1, alpha0, alpha1, height0, height1, 0, 0]

    try:
        # Fit double gaussian
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        params, _ = curve_fit(sim_action_potential, times_cyc, sig_cyc,
                               p0=guess, bounds=bounds, maxfev=maxfev)
    except:
        # Raise warning for failed fits
        warn_str = "Failed fit for index {idx}.".format(idx=index)
        warnings.warn(warn_str, RuntimeWarning)
        params = [np.nan] * 10

    if np.isnan(params[0]):
        r_squared = np.nan
    else:
        # Calculate r-squared
        residuals = sig_cyc - sim_action_potential(times_cyc, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((sig_cyc - np.mean(sig_cyc))**2)

        r_squared = 1 - (ss_res / ss_tot)

    return params, r_squared


def estimate_params(df_features, sig, fs, index):
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

    Returns
    -------
    center0 : float
        First gaussian center, as a proportion of the spike length.
    center1 : float
        Second gaussian center, as a proportion of the spike length.
    std0 : float
        First gaussian width, as a proportion of the spike length.
    std1 : float
        Second gaussian width, as a proportion of the spike length.
    alpha0 : float
        First gaussian skew.
    alpha1 : float
        Second gaussian skew.
    height0: float
        First gaussian height, in units of sig.
    height1: float
        Second gaussian height, in units of sig.
    """

    # Get sample indices
    start = df_features.iloc[index]['sample_start'].astype(int)
    end = df_features.iloc[index]['sample_end'].astype(int)
    sample_trough = df_features.iloc[index]['sample_trough'].astype(int)
    sample_peak = df_features.iloc[index]['sample_next_peak'].astype(int)

    # Get signal and times
    sig_cyc = sig[start:end+1]
    cyc_len = len(sig_cyc)
    times_cyc = np.arange(0, len(sig_cyc)/fs, 1/fs)

    baseline_volt = np.mean((sig[start], sig[end]))

    # Estimate heights
    height0 = baseline_volt - df_features.iloc[index]['volt_trough']
    height0 = height0 if height0 > 0 else 0.00001

    height1 = df_features.iloc[index]['volt_peak'] - baseline_volt
    height1 = height1 if height1 > 0 else 0.00001

    # Estimate centers
    trough_loc = sample_trough - start
    peak_loc = sample_peak - start

    center0 = trough_loc / cyc_len
    center1 = peak_loc / cyc_len

    # Estimate standard deviations
    sig_rise = sig[sample_trough:sample_peak+1]

    zerox_mid = np.argmin(np.abs(sig_rise - baseline_volt))
    zerox_mid = zerox_mid + sample_trough

    sig_trough = sig[start:zerox_mid+1]
    sig_peak = sig[zerox_mid:end+1]

    std0 = sig_trough.std() / (2 * cyc_len)
    std1 = sig_peak.std() / (2 * cyc_len)

    # Set alphas (skew parameters)
    alpha0 = 5
    alpha1 = .5

    return center0, center1, std0, std1, alpha0, alpha1, height0, height1


def sim_action_potential(times, center0, center1, std0, std1, alpha0,
                         alpha1, height0, height1, shift0, shift1):
    """Proxy function for compatibility between _sim_ap_cycle and curve_fit.

    Parameters
    ----------
    times : 1d array
        Time definition of the cycle.
    center0 : float
        First gaussian center, as a proportion of the spike length.
    center1 : float
        Second gaussian center, as a proportion of the spike length.
    std0 : float
        First gaussian width, as a proportion of the spike length.
    std1 : float
        Second gaussian width, as a proportion of the spike length.
    alpha0 : float
        First gaussian skew.
    alpha1 : float
        Second gaussian skew.
    height0: float
        First gaussian height, in units of sig.
    height1: float
        Second gaussian height, in units of sig.

    Returns
    -------
    sig_cycle : 1d array
        Simulated action potential.
    """

    centers = (center0, center1)
    stds = (std0, std1)
    alphas = (alpha0, alpha1)
    heights = (height0, height1)
    shifts = (shift0, shift1)

    sig_cycle = _sim_ap_cycle(1, len(times), centers, stds, alphas, heights, shifts,
                              max_extrema='trough')

    return sig_cycle


def _sim_gaussian_cycle(n_seconds, fs, std, center=.5, height=1.):
    """Simulate a gaussian cycle."""

    xs = np.linspace(0, 1, int(np.ceil(n_seconds * fs)))
    cycle = height * np.exp(-(xs-center)**2 / (2*std**2))

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


def _sim_ap_cycle(n_seconds, fs, centers, stds, alphas, heights, shifts, max_extrema='trough'):
    """Simulate an action potential as the sum of two skewed gaussians."""

    polar = _sim_skewed_gaussian_cycle(n_seconds, fs, centers[0], stds[0],
                                       alphas[0], height=heights[0]) + shifts[0]

    repolar = _sim_skewed_gaussian_cycle(n_seconds, fs, centers[1], stds[1],
                                         alphas[1], height=heights[1]) + shifts[1]

    cycle = polar - repolar

    if max_extrema == 'trough':
        cycle = -cycle

    return cycle
