"""Functions to compute features across 2 dimensional arrays of data."""

from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

from bycycle.features import compute_features
from bycycle.group.utils import progress_bar

###################################################################################################
###################################################################################################

def compute_features_2d(sigs, fs, f_range, compute_features_kwargs=None,
                        return_samples=True, n_jobs=-1, progress=None):
    """Compute shape and burst features for a 2 dimensional array of signals.

    Parameters
    ----------
    sigs : 2d array
        Voltage time series, with shape [n_signals, n_points].
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest, in Hz.
    compute_features_kwargs : dict or list of dict
        Keyword arguments used in :func:`~.compute_features`.
    return_samples : bool, optional, default: True
        Whether to return a dataframe of cyclepoint sample indices.
    n_jobs : int, optional, default: -1
        The number of jobs, one per cpu, to compute features in parallel.
    progress: {None, 'tqdm', 'tqdm.notebook'}, optional, default: None
        Specify whether to display a progress bar. Use 'tqdm' if installed.

    Returns
    -------
    df_features : list of pandas.DataFrame
        Dataframes containing shape and burst features for each cycle.
        Each dataframe is computed using the :func:`~.compute_features` function.
    df_samples : list of pandas.DataFrame, optional
        Dataframes containing cyclepoints for each cycle.
        Only returned if ``return_samples`` is True.
        Each dataframe is computed using the :func:`~.compute_features` function.

    Notes
    -----

    - The order of ``df_features`` and ``df_samples`` corresponds to the order of ``sigs``.
    - If ``compute_features_kwargs`` is a dictionary, the same kwargs are applied applied across
      the first axis of ``sigs``. Otherwise, a list of dictionaries equal in length to the
      first axis of ``sigs`` is required to apply unique kwargs to each signal.
    - ``return_samples`` is controlled from the kwargs passed in this function. If
      ``return_samples`` is a key in ``compute_features_kwargs``, it's value will be ignored.

    """

    if isinstance(compute_features_kwargs, list) and len(compute_features_kwargs) != len(sigs):
        raise ValueError("When compute_features_kwargs is a list, it's length must be equal to "
                         "sigs. Use a dictionary when applying the same kwargs to each signal.")
    elif compute_features_kwargs is None:
        compute_features_kwargs = {}

    # Remove return_samples from compute_features_kwargs
    #   This kwarg is set directly in the function call
    if isinstance(compute_features_kwargs, list):

        for kwargs in compute_features_kwargs:
            if 'return_samples' in kwargs.keys():
                kwargs.pop('return_samples')

    elif (isinstance(compute_features_kwargs, dict) and
         'return_samples' in compute_features_kwargs.keys()):

        compute_features_kwargs.pop('return_samples')

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    with Pool(processes=n_jobs) as pool:

        if isinstance(compute_features_kwargs, list):
            # Map iterable sigs and kwargs together
            mapping = pool.imap(partial(_proxy, fs=fs, f_range=f_range,
                                        return_samples=return_samples),
                                zip(sigs, compute_features_kwargs))

        else:
            # Only map sigs, kwargs are the same for each mapping
            mapping = pool.imap(partial(compute_features, fs=fs, f_range=f_range,
                                        return_samples=return_samples, **compute_features_kwargs),
                                sigs)

        if return_samples is True:
            df_features, df_samples = zip(*progress_bar(mapping, progress, len(sigs)))

            df_features = list(df_features)
            df_samples = list(df_samples)

        else:
            df_features = list(progress_bar(mapping, progress, len(sigs)))

    if return_samples is True:
        return df_features, df_samples

    return df_features


def _proxy(args, fs=None, f_range=None, return_samples=None):
    """Proxy function to map kwargs and sigs together."""

    sig, kwargs = args[0], args[1:]
    return compute_features(sig, fs=fs, f_range=f_range,
                            return_samples=return_samples, **kwargs[0])
