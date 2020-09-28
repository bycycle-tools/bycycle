"""Functions to compute features across epoched data."""

from importlib import import_module

###################################################################################################
###################################################################################################

def progress_bar(iterable, progress, n_to_run):
    """Add a progress bar to an iterable to be processed.

    Parameters
    ----------
    iterable : list or iterable
        Iterable object to potentially apply progress tracking to.
    progress : {None, 'tqdm', 'tqdm.notebook'}, optional
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

    - ``tqdm`` must be installed separately from bycycle.
    - The explicit `n_to_run` input is required as tqdm requires this in the parallel case.
      The `tqdm` object that is potentially returned acts the same as the underlying iterable,
      with the addition of printing out progress every time items are requested.


    Examples
    --------
    Use a ``tqdm`` progress bar, which must me installed separately from ``bycycle``,
    when computing the features for 10 signals:

    >>> from multiprocessing import Pool
    >>> from functools import partial
    >>> from bycycle.features import compute_features
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> sigs = [sim_bursty_oscillation(10, fs=500, freq=10)] * 10
    >>> mapping = Pool(1).imap(partial(compute_features, fs=500, f_range=(8, 12),
    ...                                return_samples=False), sigs)
    >>> df_features = list(progress_bar(mapping, progress='tqdm', n_to_run=len(sigs)))
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
            tqdm = import_module(progress)

            # If tqdm loaded, apply the progress bar to the iterable
            pbar = tqdm.tqdm(iterable, desc=pbar_desc, total=n_to_run, dynamic_ncols=True)

        except ImportError:

            # If tqdm isn't available, proceed without a progress bar
            print(("A progress bar requiring the 'tqdm' module was requested, "
                   "but 'tqdm' is not installed. \nProceeding without using a progress bar."))
            pbar = iterable

    # If progress is None, return the original iterable without a progress bar applied
    else:
        pbar = iterable

    return pbar
