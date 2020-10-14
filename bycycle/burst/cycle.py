"""Detect bursts: cycle consistency approach."""

import pandas as pd

from bycycle.utils.checks import check_param
from bycycle.burst.utils import check_min_burst_cycles

###################################################################################################
###################################################################################################

pd.options.mode.chained_assignment = None

def detect_bursts_cycles(df_features, amp_fraction_threshold=0., amp_consistency_threshold=.5,
                         period_consistency_threshold=.5, monotonicity_threshold=.8,
                         min_n_cycles=3):
    """Detects bursts based on consistency between cycles.

    Parameters
    ----------
    df_features : pandas.DataFrame
        Waveform features for individual cycles from :func:`~.compute_burst_features`.
    amp_fraction_threshold : float, optional, default: 0.
        The minimum normalized amplitude a cycle must have in order to be considered in an
        oscillation. Must be between 0 and 1.

        - 0 = the minimum amplitude across all cycles
        - .5 = the median amplitude across all cycles
        - 1 = the maximum amplitude across all cycles

    amp_consistency_threshold : float, optional, default: 0.5
        The minimum normalized difference in rise and decay magnitude to be considered as in an
        oscillatory mode. Must be between 0 and 1.

        - 1 = the same amplitude for the rise and decay
        - .5 = the rise (or decay) is half the amplitude of the decay (rise)

    period_consistency_threshold : float, optional, default: 0.5
        The minimum normalized difference in period between two adjacent cycles to be considered
        as in an oscillatory mode. Must be between 0 and 1.

        - 1 = the same period for both cycles
        - .5 = one cycle is half the duration of another cycle

    monotonicity_threshold : float, optional, default: 0.8
        The minimum fraction of time segments between samples that must be going in the same
        direction. Must be between 0 and 1.

        - 1 = rise and decay are perfectly monotonic
        - .5 = both rise and decay are rising half of the time and decay half the time
        - 0 = rise period is all decaying and decay period is all rising

    min_n_cycles : int, optional, default: 3
        The minimum number of cycles of consecutive cycles required to be considered a burst.

    Returns
    -------
    df_features : pandas.DataFrame
        Same df as input, with an additional column (`is_burst`) to indicate if the cycle is part
        of an oscillatory burst, with additional columns indicating the burst detection parameters.

    Notes
    -----
    * The first and last period cannot be considered oscillating if the consistency measures are
      used.

    Examples
    --------
    Apply thresholding for consistency burst detection:

    >>> from bycycle.features import compute_burst_features, compute_shape_features
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> df_shapes, df_samples = compute_shape_features(sig, fs, f_range=(8, 12))
    >>> df_burst = compute_burst_features(df_shapes, df_samples, sig)
    >>> df_burst = detect_bursts_cycles(df_burst)
    """

    # Ensure arguments are within valid ranges
    check_param(amp_fraction_threshold, 'amp_fraction_threshold', (0, 1))
    check_param(amp_consistency_threshold, 'amp_consistency_threshold', (0, 1))
    check_param(period_consistency_threshold, 'period_consistency_threshold', (0, 1))
    check_param(monotonicity_threshold, 'monotonicity_threshold', (0, 1))

    # Compute if each period is part of an oscillation
    amp_fraction = df_features['amp_fraction'] > amp_fraction_threshold
    amp_consistency = df_features['amp_consistency'] > amp_consistency_threshold
    period_consistency = df_features['period_consistency'] > period_consistency_threshold
    monotonicity = df_features['monotonicity'] > monotonicity_threshold

    # Set the burst status for each cycle as the answer across all criteria
    is_burst = amp_fraction & amp_consistency & period_consistency & monotonicity

    # Set the first and last cycles to not be part of a burst
    is_burst[0] = False
    is_burst[-1] = False

    df_features['is_burst'] = check_min_burst_cycles(is_burst, min_n_cycles=min_n_cycles)

    return df_features
