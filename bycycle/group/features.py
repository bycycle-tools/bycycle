"""Functions to compute features across 2 dimensional arrays of data."""

import warnings
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

from bycycle.features import compute_features
from bycycle.burst import detect_bursts_cycles, detect_bursts_amp
from bycycle.group.utils import progress_bar

###################################################################################################
###################################################################################################

def compute_features_2d(sigs, fs, f_range, compute_features_kwargs=None, global_features=False,
                        return_samples=True, n_jobs=-1, progress=None):
    """Compute shape and burst features for a 2 dimensional array of signals.

    Parameters
    ----------
    sigs : 2d array
        Voltage time series, i.e. (n_channels, n_samples) or (n_epochs, n_samples).
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest, in Hz.
    compute_features_kwargs : dict or list of dict
        Keyword arguments used in :func:`~.compute_features`.
    global_features : bool, optional default: False
        Calculates features across a flattened, 1d array if True. This is recommended when the 2d
        array was recorded continuously, i.e. (n_epochs, n_signals).
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

    Notes
    -----

    - When ``global_features = True`` parallel computation may not be performed due to the
      requirement of flattening the array into one dimension.
    - The order of ``df_features`` corresponds to the order of ``sigs``.
    - If ``compute_features_kwargs`` is a dictionary, the same kwargs are applied applied across
      the first axis of ``sigs``. Otherwise, a list of dictionaries equal in length to the
      first axis of ``sigs`` is required to apply unique kwargs to each signal.
    - ``return_samples`` is controlled from the kwargs passed in this function. If
      ``return_samples`` is a key in ``compute_features_kwargs``, it's value will be ignored.

    Examples
    --------
    Compute the features of a 2d array (n_epochs=10, n_samples=5000) containing epoched data:

    >>> import numpy as np
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sigs = np.array([sim_bursty_oscillation(10, fs, 10) for i in range(10)])
    >>> compute_kwargs = {'burst_method': 'amp', 'threshold_kwargs':{'burst_fraction_threshold': 1}}
    >>> df_features = compute_features_2d(sigs, fs, f_range=(8, 12), global_features=True,
    ...                                   compute_features_kwargs=compute_kwargs)

    Compute the features of a 2d array in parallel using the same compute_features kwargs. Note each
    signal's features are computed separately in this case:

    >>> compute_kwargs = {'burst_method': 'amp', 'threshold_kwargs':{'burst_fraction_threshold': 1}}
    >>> df_features = compute_features_2d(sigs, fs, f_range=(8, 12), n_jobs=2,
    ...                                   compute_features_kwargs=compute_kwargs)

    Compute the features of a 2d array in parallel using using individualized settings per signal to
    examine the effect of various amplitude consistency thresholds. This is recommended when working
    with a signal of shape (n_channels, n_samples):

    >>> sigs =  np.array([sim_bursty_oscillation(10, fs, freq=10)] * 10)
    >>> compute_kwargs = [{'threshold_kwargs': {'amp_consistency_threshold': thresh*.1}}
    ...                   for thresh in range(1, 11)]
    >>> df_features = compute_features_2d(sigs, fs, f_range=(8, 12), return_samples=False,
    ...                                   n_jobs=2, compute_features_kwargs=compute_kwargs)
    """

    if isinstance(compute_features_kwargs, list) and len(compute_features_kwargs) != len(sigs):
        raise ValueError(""""
            When compute_features_kwargs is a list, its length must be equal to sigs. Use a
            dictionary when applying the same kwargs to each signal.
        """)
    elif compute_features_kwargs is None:
        compute_features_kwargs = {}

    # Drop `return_samples` argument, as this is set directly in the function call
    compute_features_kwargs = [compute_features_kwargs] if \
        isinstance(compute_features_kwargs, dict) else compute_features_kwargs

    [kwargs.pop('return_samples', None) for kwargs in compute_features_kwargs]

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs


    if global_features is False:
        # Compute each signal's independently and in paralllel

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

            df_features = list(progress_bar(mapping, progress, len(sigs)))

    else:

        # Compute features after flattening the 2d array (i.e. calculated across a 1d signal)
        sig_flat = sigs.flatten()

        center_extrema = compute_features_kwargs[0].pop('center_extrema', 'peak')

        df_flat = compute_features(sig_flat, fs=fs, f_range=f_range, return_samples=True,
                                   center_extrema=center_extrema, **compute_features_kwargs[0])

        # Reshape the dataframe into original sigs shape
        last_sample = 'sample_next_trough' if center_extrema == 'peak' else 'sample_next_peak'

        df_features = []
        sig_last_idxs = np.arange(len(sigs[0]), len(sig_flat) + len(sigs[0]), len(sigs[0]))
        sig_first_idxs = np.append(0, sig_last_idxs[:-1])

        for first_idx, last_idx in zip(sig_first_idxs, sig_last_idxs):

            # Get the range for each df
            idx_range = np.where((df_flat[last_sample].values <= last_idx) & \
                                 (df_flat[last_sample].values > first_idx))[0]

            df_single = df_flat.iloc[idx_range]
            df_single.reset_index(drop=True, inplace=True)

            # Shift sample indices
            sample_cols = [col for col in df_single.columns if 'sample_' in col]

            for col in sample_cols:
                df_single[col] = df_single[col] - last_idx

            df_features.append(df_single)

        # Apply different thresholds if specified
        if len(compute_features_kwargs) > 0:

            for idx, compute_kwargs in enumerate(compute_features_kwargs):

                burst_method = compute_kwargs.pop('burst_method', 'cycles')
                thresholds = compute_kwargs.pop('threshold_kwargs', {})

                compute_kwargs_next = compute_kwargs.pop('center_extrema', None)
                if idx > 0 and not compute_kwargs_next and compute_kwargs_next != center_extrema:

                    warnings.warn('''
                        The same center extrema must be used when using global_features with a list
                        of compute_features_kwargs. Using the first center_extrema: {extrema}.
                        '''.format(extrema=center_extrema))

                if burst_method == 'cycles':
                    df_features[idx] = detect_bursts_cycles(df_features[idx], **thresholds)

                elif burst_method == 'amp':
                    df_features[idx] = detect_bursts_amp(df_features[idx], **thresholds)

    return df_features


def compute_features_3d(sigs, fs, f_range, compute_features_kwargs=None, global_features=False,
                        return_samples=True, n_jobs=-1, progress=None):
    """Compute shape and burst features for a 3 dimensional array of signals.

    Parameters
    ----------
    sigs : 3d array
        Voltage time series, with 3d shape, i.e. (n_groups, n_epochs, n_samples).
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest, in Hz.
    compute_features_kwargs : dict or 1d list of dict or 2d list of dict
        Keyword arguments used in :func:`~.compute_features`.
    global_features : bool, optional default: False
        Calculates features across a flattened, 1d array if True. This is recommended when the 3d
        3d array was recorded continuously.
    return_samples : bool, optional, default: True
        Whether to return a dataframe of cyclepoint sample indices.
    n_jobs : int, optional, default: -1
        The number of jobs, one per cpu, to compute features in parallel.
    progress : {None, 'tqdm', 'tqdm.notebook'}, optional, default: None
        Specify whether to display a progress bar. Use 'tqdm' if installed.

    Returns
    -------
    df_features : list of pandas.DataFrame
        Dataframes containing shape and burst features for each cycle.
        Each dataframe is computed using the :func:`~.compute_features` function.

    Notes
    -----

    - The order of ``df_features`` corresponds to the order of ``sigs``.
    - If ``compute_features_kwargs`` is a dictionary, the same kwargs are applied applied across
      all signals. A 1d list, equal in length to the first dimensions of sigs, may be applied to
      each set of signals along the first dimensions. A 2d list, the same shape as the first two
      dimensions of ``sigs`` may also be used to applied unique parameters to each signal.
    - ``return_samples`` is controlled from the kwargs passed in this function. The
      ``return_samples`` value in ``compute_features_kwargs`` will be ignored.
    - When ``global_features = True`` parallel computation may not be performed due to the
      requirement of flattening the array into one dimension.

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

            raise ValueError("""
                When compute_features_kwargs is a 1d list, it's length must be equal to the first
                dimension of sigs. Use a dictionary when applying the same kwargs to each signal.
            """)

        elif ((kwargs.ndim == 2 and np.shape(kwargs)[0] != np.shape(sigs)[0]) or
              (kwargs.ndim == 2 and np.shape(kwargs)[1] != np.shape(sigs)[1])):

            raise ValueError("""
                When compute_features_kwargs is a 2d list, it's shape must be equal to the shape of
                the first two dimensions of sigs. Use a dictionary when applying the same kwargs to
                each signal.
            """)

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
                            global_features=global_features, return_samples=return_samples,
                            n_jobs=n_jobs, progress=progress)

    # Reshape returned features to match the first two dimensions of sigs
    df_features = _reshape_df(df_features, sigs)

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
