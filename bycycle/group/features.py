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

    Examples
    --------
    Compute the features of a 2d array in parrallel using the same compute_features kwargs:

    >>> import numpy as np
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sigs = np.array([sim_bursty_oscillation(10, fs, 10) for i in range(10)])
    >>> compute_kwargs = {'burst_method': 'amp', 'threshold_kwargs':{'burst_fraction_threshold': 1}}
    >>> df_features = compute_features_2d(sigs, fs, f_range=(8, 12), return_samples=False, n_jobs=2,
    ...                                   compute_features_kwargs=compute_kwargs)

    Compute the features of a 2d array in parallel using using individualized settings per signal to
    examine the effect of various amplitude consistency thresholds:

    >>> sigs =  np.array([sim_bursty_oscillation(10, fs, freq=10)] * 10)
    >>> compute_kwargs = [{'threshold_kwargs': {'amp_consistency_threshold': thresh*.1}}
    ...                   for thresh in range(1, 11)]
    >>> df_features = compute_features_2d(sigs, fs, f_range=(8, 12), return_samples=False,
    ...                                   n_jobs=2, compute_features_kwargs=compute_kwargs)
    """

    if isinstance(compute_features_kwargs, list) and len(compute_features_kwargs) != len(sigs):
        raise ValueError("When compute_features_kwargs is a list, it's length must be equal to "
                         "sigs. Use a dictionary when applying the same kwargs to each signal.")
    elif compute_features_kwargs is None:
        compute_features_kwargs = {}

    # Drop `return_samples` argument, as this is set directly in the function call
    compute_features_kwargs = [compute_features_kwargs] if \
        isinstance(compute_features_kwargs, dict) else compute_features_kwargs

    [kwargs.pop('return_samples', None) for kwargs in compute_features_kwargs]

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    with Pool(processes=n_jobs) as pool:

        if len(compute_features_kwargs) > 1:
            # Map iterable sigs and kwargs together
            mapping = pool.imap(partial(_proxy, fs=fs, f_range=f_range,
                                        return_samples=return_samples),
                                zip(sigs, compute_features_kwargs))

        else:
            # Only map sigs, kwargs are the same for each mapping
            mapping = pool.imap(partial(compute_features, fs=fs, f_range=f_range,
                                        return_samples=return_samples,
                                        **compute_features_kwargs[0]),
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


def compute_features_3d(sigs, fs, f_range, compute_features_kwargs=None,
                        return_samples=True, n_jobs=-1, progress=None):
    """Compute shape and burst features for a 3 dimensional array of signals.

    Parameters
    ----------
    sigs : 3d array
        Voltage time series, with shape [n_groups, n_signals, n_points].
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest, in Hz.
    compute_features_kwargs : dict or 1d list of dict or 2d list of dict
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
      all signals. A 1d list, equal in length to the first dimensions of sigs, may be applied to
      each set of signals along the first dimensions. A 2d list, the same shape as the first two
      dimensions of ``sigs`` may also be used to applied unique parameters to each signal.
    - ``return_samples`` is controlled from the kwargs passed in this function. The
      ``return_samples`` value in ``compute_features_kwargs`` will be ignored.

    Examples
    --------
    Compute the features of a 3d array, in parallel, with a shape of
    (n_channels=2, n_epochs=3, n_signals=5000) using the same compute_features kwargs:

    >>> import numpy as np
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sigs = np.array([[sim_bursty_oscillation(10, fs, freq=10) for epoch in range(3)]
    ...                 for ch in range(2)])
    >>> threshold_kwargs = {'amp_consistency_threshold': .5, 'period_consistency_threshold': .5,
    ...                     'monotonicity_threshold': .8, 'min_n_cycles': 3}
    >>> compute_feature_kwargs = {'threshold_kwargs': threshold_kwargs, 'center_extrema': 'trough'}
    >>> features = compute_features_3d(sigs, fs, f_range= (8, 12), return_samples=False, n_jobs=2,
    ...                                compute_features_kwargs=compute_feature_kwargs)

    Compute the features of a 3d array, in parallel, with a shape of
    (n_channels=2, n_epochs=3, n_signals=5000) using channel-specific compute_features kwargs:

    >>> threshold_kwargs_ch1 = {'amp_consistency_threshold': .25, 'monotonicity_threshold': .25,
    ...                         'period_consistency_threshold': .25, 'min_n_cycles': 3}
    >>> threshold_kwargs_ch2 = {'amp_consistency_threshold': .5, 'monotonicity_threshold': .5,
    ...                         'period_consistency_threshold': .5, 'min_n_cycles': 3}
    >>> compute_kwargs = [{'threshold_kwargs': threshold_kwargs_ch1, 'center_extrema': 'trough'},
    ...                   {'threshold_kwargs': threshold_kwargs_ch2, 'center_extrema': 'trough'}]
    >>> features = compute_features_3d(sigs, fs, f_range= (8, 12), return_samples=False, n_jobs=2,
    ...                                compute_features_kwargs=compute_kwargs)
    """

    # Convert list of kwargs to array to check dimensions
    kwargs = compute_features_kwargs

    if isinstance(kwargs, list):
        kwargs = np.array(kwargs)
    elif kwargs is None:
        kwargs = {}

    # Ensure compute_features corresponds to sigs
    if isinstance(kwargs, np.ndarray):

        if kwargs.ndim == 1 and np.shape(kwargs)[0] != np.shape(sigs)[0]:

            raise ValueError("When compute_features_kwargs is a 1d list, it's length must "
                             "be equal to the first dimension of sigs. Use a dictionary "
                             "when applying the same kwargs to each signal.")

        elif ((kwargs.ndim == 2 and np.shape(kwargs)[0] != np.shape(sigs)[0]) or
              (kwargs.ndim == 2 and np.shape(kwargs)[1] != np.shape(sigs)[1])):

            raise ValueError("When compute_features_kwargs is a 2d list, it's shape must "
                             "be equal to the shape of the first two dimensions of sigs. "
                             "Use a dictionary when applying the same kwargs to each signal.")

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    # Reshape sigs into 2d aray
    sigs_2d = sigs.reshape(np.shape(sigs)[0]*np.shape(sigs)[1], np.shape(sigs)[2])

    # Reshape kwargs
    if isinstance(kwargs, np.ndarray) and kwargs.ndim == 1:
        # Repeat each kwargs for each signal along the second dimension
        kwargs = np.array([[kwarg]*np.shape(sigs)[1] for kwarg in kwargs])

    # Flatten kwargs
    if isinstance(kwargs, np.ndarray):
        kwargs = list(kwargs.flatten())

    df_features = \
        compute_features_2d(sigs_2d, fs, f_range, compute_features_kwargs=kwargs,
                            return_samples=return_samples, n_jobs=n_jobs, progress=progress)

    if return_samples:

        df_features, df_samples = df_features[0], df_features[1]
        df_samples = _reshape_df(df_samples, sigs)

    # Reshape returned features to match the first two dimensions of sigs
    df_features = _reshape_df(df_features, sigs)

    if return_samples:
        return df_features, df_samples

    return df_features


def _proxy(args, fs=None, f_range=None, return_samples=None):
    """Proxy function to map kwargs and sigs together."""

    sig, kwargs = args[0], args[1:]
    return compute_features(sig, fs=fs, f_range=f_range,
                            return_samples=return_samples, **kwargs[0])


def _reshape_df(df_features, sigs_3d):
    """Reshape a list of dataframes."""

    df_reshape = []

    dim_b = np.shape(sigs_3d)[1]

    max_idx = [i for i in range(dim_b, len(df_features)*dim_b, dim_b)]
    min_idx = [i for i in range(0, len(df_features), dim_b)]

    for lower, upper in zip(min_idx, max_idx):
        df_reshape.append(df_features[lower:upper])

    return df_reshape
