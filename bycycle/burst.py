"""Analyze periods of oscillatory bursting in neural signals."""

import numpy as np
import pandas as pd

###################################################################################################
###################################################################################################

pd.options.mode.chained_assignment = None

def detect_bursts_cycles(df_features, amplitude_fraction_threshold=0.,
                         amplitude_consistency_threshold=.5,
                         period_consistency_threshold=.5,
                         monotonicity_threshold=.8,
                         n_cycles_min=3):
    """Compute consistency between cycles and determine which are truly oscillating.

    Parameters
    ----------
    df_features : pandas DataFrame
        Waveform features for individual cycles from :func:`~.compute_burst_features`.
    amplitude_fraction_threshold : float, optional, default: 0.
        The minimum normalized amplitude a cycle must have in order to be considered in an
        oscillation. Must be between 0 and 1.

        - 0 = the minimum amplitude across all cycles
        - .5 = the median amplitude across all cycles
        - 1 = the maximum amplitude across all cycles

    amplitude_consistency_threshold : float, optional, default: 0.5
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

    n_cycles_min : int, optional, default: 3
        Minimum number of cycles to be identified as truly oscillating needed in a row in order
        for them to remain identified as truly oscillating.

    Returns
    -------
    df_features : pandas DataFrame
        Same df as input, with an additional column (`is_burst`) to indicate if the cycle is part
        of an oscillatory burst, with additional columns indicating the burst detection parameters.

    Notes
    -----
    * The first and last period cannot be considered oscillating if the consistency measures are
      used.
    """

    # Compute if each period is part of an oscillation
    cycle_amp_burst = df_features['amplitude_fraction'] > amplitude_fraction_threshold
    cycle_amp_consist_burst = df_features['amplitude_consistency'] > amplitude_consistency_threshold
    cycle_period_consist_burst = df_features['period_consistency'] > period_consistency_threshold
    cycle_monotonicity_burst = df_features['monotonicity'] > monotonicity_threshold

    is_burst = cycle_amp_burst & cycle_amp_consist_burst & \
        cycle_period_consist_burst & cycle_monotonicity_burst
    is_burst[0] = False
    is_burst[-1] = False

    df_features['is_burst'] = is_burst
    df_features = _min_consecutive_cycles(df_features, n_cycles_min=n_cycles_min)
    df_features['is_burst'] = df_features['is_burst'].astype(bool)

    return df_features


def detect_bursts_df_amp(df_features, burst_fraction_threshold=1, n_cycles_min=3):
    """Determine which cycles in a signal are part of an oscillatory
    burst using an amplitude thresholding approach.

    Parameters
    ----------
    df_features : pandas DataFrame
        Waveform features for individual cycles from :func:`~.compute_burst_features`.
    burst_fraction_threshold : int or float, optional, default: 1
        Minimum fraction of a cycle to be indentified as a burst.
    n_cycles_min : int, optional, default: 3
        Minimum number of cycles to be identified as truly oscillating needed in a row in order
        for them to remain identified as truly oscillating.

    Returns
    -------
    df : pandas DataFrame
        Same df as input, with an additional column to indicate
        if the cycle is part of an oscillatory burst.
    """

    # Determine cycles that are defined as bursting throughout the whole cycle
    df_features['is_burst'] = [frac >= burst_fraction_threshold for frac in
                               df_features['burst_fraction']]

    df_features = _min_consecutive_cycles(df_features, n_cycles_min=n_cycles_min)
    df_features['is_burst'] = df_features['is_burst'].astype(bool)

    return df_features


def _min_consecutive_cycles(df_shape, n_cycles_min=3):
    """Enforce minimum number of consecutive cycles."""

    is_burst = np.copy(df_shape['is_burst'].values)
    temp_cycle_count = 0

    for idx, bursting in enumerate(is_burst):

        if bursting:
            temp_cycle_count += 1

        else:

            if temp_cycle_count < n_cycles_min:
                for c_rm in range(temp_cycle_count):
                    is_burst[idx - 1 - c_rm] = False

            temp_cycle_count = 0

    df_shape['is_burst'] = is_burst

    return df_shape
