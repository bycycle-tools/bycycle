"""Detect bursts: amplitude threshold approach."""

from bycycle.utils.checks import check_param
from bycycle.burst.utils import check_min_burst_cycles

###################################################################################################
###################################################################################################

def detect_bursts_amp(df_features, burst_fraction_threshold=1, min_n_cycles=3):
    """Detect bursts based on an amplitude thresholding.

    Parameters
    ----------
    df_features : pandas.DataFrame
        Waveform features for individual cycles from :func:`~.compute_burst_features`.
    burst_fraction_threshold : int or float, optional, default: 1
        Minimum fraction of a cycle to be identified as a burst.
    min_n_cycles : int, optional, default: 3
        The minimum number of cycles of consecutive cycles required to be considered a burst.

    Returns
    -------
    df_features : pandas.DataFrame
        Dataframe updated, with a additional column to indicate if the cycle is part of a burst.

    Examples
    --------
    Apply thresholding for dual amplitude burst detection:

    >>> from bycycle.features import compute_burst_features, compute_shape_features
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> df_shapes, df_samples = compute_shape_features(sig, fs, f_range=(8, 12))
    >>> df_burst = compute_burst_features(df_shapes, df_samples, sig, burst_method='amp',
    ...                                   burst_kwargs={'fs': fs, 'f_range': (8, 12)})
    >>> df_burst = detect_bursts_amp(df_burst)
    """

    # Ensure arguments are within valid ranges
    check_param(burst_fraction_threshold, 'burst_fraction_threshold', (0, 1))

    # Determine cycles that are defined as bursting throughout the whole cycle
    is_burst = [frac >= burst_fraction_threshold for frac in df_features['burst_fraction']]

    df_features['is_burst'] = check_min_burst_cycles(is_burst, min_n_cycles=min_n_cycles)

    return df_features
