"""Compute multi-electrode array features."""

from functools import partial

import numpy as np

from multiprocessing import Pool, cpu_count
from bycycle.group.utils import progress_bar


###################################################################################################
###################################################################################################


def compute_mea_features(df_samples, sigs, n_jobs=-1, chunksize=1, progress=None):
    """Compute correlation and relative variance features.

    Parameters
    ----------
    df_samples : pandas.DataFrame
        Contains cyclepoint locations for each spike.
    sigs : 2d array
        Voltage time series.
    n_jobs : int, optional, default: -1
        The number of jobs to compute features in parallel.
    chunksize : int, optional, default: 1
        Number of chunks to split spikes into. Each chunk is submitted as a separate job.
        With a large number of spikes, using a larger chunksize will drastically speed up
        runtime. An optimal chunksize is typically np.ceil(n_spikes/n_jobs).
    progress : {None, 'tqdm', 'tqdm.notebook'}
        Specify whether to display a progress bar. Uses 'tqdm', if installed.

    Returns
    -------
    params : 1d array
        Fit parameter values, the first parameters are correlation coeffeicients, followed by
        relative variances.
    """

    starts = df_samples['sample_start']
    ends = df_samples['sample_end']
    locs = np.vstack((starts, ends)).T

    with Pool(processes=n_jobs) as pool:

        mapping = pool.imap(partial(mea, sigs=sigs), locs, chunksize=chunksize)

        params = list(progress_bar(mapping, progress, len(df_samples)))

    return np.array(params)


def _compute_mea_features(locs, sigs):
    """Compute correlation and relative variance for each electrode."""

    start, end = locs[0], locs[1]

    coeffs = np.corrcoef(sigs[:, start:end+1])
    coeffs = np.hstack([coeffs[idx, idx+1:] for idx in range(len(coeffs))])

    var = np.var(sigs[:, start:end+1], axis=1)
    var /= var.sum()

    return np.append(coeffs, var)
