"""Functions to compute features across epoched data.
"""

from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from bycycle.features import compute_features

###################################################################################################
###################################################################################################

def compute_features_epochs(sigs, fs, f_range, compute_features_kwargs=None,
                            return_samples=True, n_cpus=-1, progress=None):
    """Compute shape and burst features for epoched signals.

    Parameters
    ----------
    sigs : 2d array
        Voltage time series for each epoch.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest (Hz).
    compute_features_kwargs : dict
        Keyword arguments used in :func:`~.compute_features`
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
    The order of ``df_features`` and ``df_samples`` corresponds to the order of ``sigs``.
    """

    n_cpus = cpu_count() if n_cpus == -1 else n_cpus

    return_samples = compute_features_kwargs['return_samples'] if 'return_samples' in \
        compute_features_kwargs.keys() else True

    with Pool(processes=n_cpus) as pool:

        if return_samples is True:
            df_features, df_samples = zip(*_progress(pool.imap(partial(compute_features, fs=fs,
                                                                       f_range=f_range,
                                                                       **compute_features_kwargs),
                                                               sigs),
                                                     progress, len(sigs)))

            df_features = list(df_features)
            df_samples = list(df_samples)

        else:
            df_features = list(_progress(pool.imap(partial(compute_features, fs=fs,
                                                           f_range=f_range,
                                                           **compute_features_kwargs),
                                                   sigs),
                                         progress, len(sigs)))

    if return_samples is True:
        return df_features, df_samples

    return df_features


def _progress(iterable, progress, n_to_run):
    """Add a progress bar to an iterable to be processed.

    Parameters
    ----------
    iterable : list or iterable
        Iterable object to potentially apply progress tracking to.
    progress : {None, 'tqdm'}, optional
        Which kind of progress bar to use. If None, no progress bar is used.
    n_to_run : int
        Number of jobs to complete.

    Returns
    -------
    pbar : iterable or tqdm object
        Iterable object, with tqdm progress functionality, if requested.

    Raises
    ------
    ValueError
        If the input for `progress` is not understood.

    Notes
    -----
    The explicit `n_to_run` input is required as tqdm requires this in the parallel case.
    The `tqdm` object that is potentially returned acts the same as the underlying iterable,
    with the addition of printing out progress every time items are requested.
    """

    # Check progress specifier is okay
    tqdm_options = ['tqdm', 'tqdm.notebook']
    if progress is not None and progress not in tqdm_options:
        raise ValueError("Progress bar option not understood.")

    # Set the display text for the progress bar
    pbar_desc = 'Computing Bycycle Features'

    # Use a tqdm, progress bar, if requested
    if progress:

        # Try loading the tqdm module
        try:
            from tqdm import tqdm

            # If tqdm loaded, apply the progress bar to the iterable
            pbar = tqdm(iterable, desc=pbar_desc, total=n_to_run, dynamic_ncols=True)

        except ModuleNotFoundError:

            # If tqdm isn't available, proceed without a progress bar
            print(("A progress bar requiring the 'tqdm' module was requested, "
                   "but 'tqdm' is not installed. \nProceeding without using a progress bar."))
            pbar = iterable

    # If progress is None, return the original iterable without a progress bar applied
    else:
        pbar = iterable

    return pbar
