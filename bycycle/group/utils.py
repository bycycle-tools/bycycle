"""Functions to compute features across epoched data."""

from importlib import import_module
import numpy as np

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


def check_kwargs_shape(sigs, compute_features_kwargs, axis):
    """Raise an error when compute_features_kwargs and the shape of sigs mismatch.

    Parameters
    ----------
    sigs : 2d or 3d array
        Voltage time series.
    compute_features_kwargs : dict or 1d list of dict or 2d list of dict
        Keyword arguments used in :func:`~.compute_features`.
    axis : {None, 0, 1, 2}
        Which axes to calculate features across.

    Raises
    ------
    ValueError
        If the shape compute_features_kwargs and sigs are not compatible.
    """

    kwargs = compute_features_kwargs

    # Don't raise error when kwargs is None or a dict
    if isinstance(kwargs, dict) or kwargs is None:
        return

    # Ensure kwargs match to sigs
    kwargs_dim0 = np.shape(kwargs)[0]
    kwargs_dim1 = np.shape(kwargs)[1] if kwargs.ndim == 2 else None

    sigs_dim0 = np.shape(sigs)[0]
    sigs_dim1 = np.shape(sigs)[1] if sigs.ndim == 3 else None

    # 2D checks
    if axis != 2 and kwargs_dim1 is not None:
        raise ValueError("When compute_features_kwargs is a 2d list, axis must be 2.")
    elif axis == 0 and sigs_dim1 is None:
        raise ValueError("When sigs is 2D, axis must be either 0 or None.")
    elif (axis == 1 or axis == None) and sigs_dim1 is None and kwargs_dim0 != sigs_dim0:
        axis_str = "zeroth"

    # 3D checks
    elif axis == None and sigs_dim1 is not None:
        raise ValueError("When sigs is 3D, axis must be either 0, 1, or 2.")
    elif axis == 0 and kwargs_dim0 != sigs_dim0:
        axis_str = 'zeroth'
    elif axis == 1 and sigs_dim1 is not None and kwargs_dim0 != sigs_dim1:
        axis_str = 'first'
    elif axis == 2 and (kwargs_dim0 != sigs_dim0 or kwargs_dim1 != sigs_dim1):
        axis_str = 'sum of the zeroth and first'
    elif axis == 2 and kwargs_dim1 is None:
        raise ValueError("When axis=2, compute_features_kwargs must be a 2d list.")
    else:
        return

    error_str = """
        When compute_features_kwargs is a {dim}d list and axis={axis_int}, its length must be
        equal to the {axis_str} dimension of sigs.
    """.format(dim=kwargs.ndim, axis_int=axis, axis_str=axis_str)

    raise ValueError(error_str)
