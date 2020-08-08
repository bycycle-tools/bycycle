"""Functions to compute features across epoched data.
"""

from functools import partial
from multiprocessing import Pool, cpu_count

from bycycle.features import compute_features
from bycycle.group.utils import progress_bar

###################################################################################################
###################################################################################################

def compute_features_2d(sigs, fs, f_range, compute_features_kwargs=None,
                        return_samples=True, n_cpus=-1, progress=None):
    """Compute shape and burst features for epoched signals.

    Parameters
    ----------
    sigs : 2d array
        Voltage time series for each epoch with shape (i.e. n_epochs, n_points).
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest (Hz).
    compute_features_kwargs : dict or list of dict
        Keyword arguments used in :func:`~.compute_features`.
    return_samples : bool, optional, default: True
        Whether to return a dataframe of cyclepoint sample indices.
    n_cpus : int, optional, default: -1
        The number of jobs, one per cpu, to compute features in parallel.
    progress: str, optional, default: None
        Specificy whether to display a progress. Use 'tqdm' if installed.

    Returns
    -------
    df_features : list
        A list of dataframes containing shape and burst features for each cycle.
        Each dataframe is computed using the :func:`~.compute_features` function.
    df_samples : list, optional
        A list of dataframes containing cyclepoints for each cycle returned when ``return_samples``
        is True. Each dataframe is computed using the :func:`~.compute_features` function.

    Notes
    -----

    - The order of ``df_features`` and ``df_samples`` corresponds to the order of ``sigs``.
    - If ``compute_features_kwargs`` is a dictionary, the same kwargs are applied applied across
      the first axis of ``sigs``. Otherwise, a list of dictionaries equal in length to the
      first axis of ``sigs`` is required to apply unique kwargs to each signal.

    """

    if isinstance(compute_features_kwargs, list) and len(compute_features_kwargs) != len(sigs):
        raise ValueError('When compute_features_kwargs is a list, it\'s length must be equal to '
                         'sigs. Use a dictionary when applying the same kwargs to each signal.')

    n_cpus = cpu_count() if n_cpus == -1 else n_cpus

    return_samples = compute_features_kwargs['return_samples'] if 'return_samples' in \
        compute_features_kwargs.keys() else True

    with Pool(processes=n_cpus) as pool:

        def _proxy(args, fs=None, f_range=None):
            """Proxy function to map kwargs and sigs together."""

            sig, kwargs = args[0], args[1:]
            return compute_features(sig, fs=fs, f_range=f_range, **kwargs)

        if isinstance(compute_features_kwargs, list):
            # Map iterable sigs and kwargs together
            mapping = pool.imap(partial(_proxy, fs=fs, f_range=f_range),
                                zip(sigs, compute_features_kwargs))

        else:
            # Only map sigs, kwargs are the same for each mapping
            mapping = pool.imap(partial(compute_features, fs=fs, f_range=f_range,
                                        **compute_features_kwargs),
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
