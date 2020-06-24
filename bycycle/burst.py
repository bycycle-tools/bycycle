"""Analyze periods of oscillatory bursting in neural signals."""

import numpy as np
import pandas as pd

from neurodsp.burst import detect_bursts_dual_threshold

###################################################################################################
###################################################################################################

pd.options.mode.chained_assignment = None

def detect_bursts_cycles(df, sig, amp_fraction_threshold=0.,
                         amp_consistency_threshold=.5,
                         period_consistency_threshold=.5,
                         monotonicity_threshold=.8,
                         n_cycles_min=3):
    """Compute consistency between cycles and determine which are truly oscillating.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe of waveform features for individual cycles, trough-centered.
    sig : 1d array
        Signal used to compute monotonicity.
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

    n_cycles_min : int, optional, default: 3
        Minimum number of cycles to be identified as truly oscillating needed in a row in order
        for them to remain identified as truly oscillating.

    Returns
    -------
    df : pandas DataFrame
        Same df as input, with an additional column (`is_burst`) to indicate if the cycle is part
        of an oscillatory burst, with additional columns indicating the burst detection parameters.

    Notes
    -----
    * The first and last period cannot be considered oscillating if the consistency measures are used.
    """

    # Compute normalized amplitude for all cycles
    df['amp_fraction'] = df['volt_amp'].rank()/len(df)

    # Compute amplitude consistency
    cycles = len(df)
    amp_consists = np.ones(cycles) * np.nan
    rises = df['volt_rise'].values
    decays = df['volt_decay'].values

    for cyc in range(1, cycles-1):

        consist_current = np.min([rises[cyc], decays[cyc]]) / np.max([rises[cyc], decays[cyc]])

        if 'sample_peak' in df.columns:
            consist_last = np.min([rises[cyc], decays[cyc-1]]) / np.max([rises[cyc], decays[cyc-1]])
            consist_next = np.min([rises[cyc+1], decays[cyc]]) / np.max([rises[cyc+1], decays[cyc]])

        else:
            consist_last = np.min([rises[cyc-1], decays[cyc]]) / np.max([rises[cyc-1], decays[cyc]])
            consist_next = np.min([rises[cyc], decays[cyc+1]]) / np.max([rises[cyc], decays[cyc+1]])

        amp_consists[cyc] = np.min([consist_current, consist_next, consist_last])

    df['amp_consistency'] = amp_consists

    # Compute period consistency
    period_consists = np.ones(cycles) * np.nan
    periods = df['period'].values

    for cyc in range(1, cycles-1):

        consist_last = np.min([periods[cyc], periods[cyc-1]]) / \
            np.max([periods[cyc], periods[cyc-1]])
        consist_next = np.min([periods[cyc+1], periods[cyc]]) / \
            np.max([periods[cyc+1], periods[cyc]])

        period_consists[cyc] = np.min([consist_next, consist_last])

    df['period_consistency'] = period_consists

    # Compute monotonicity
    monotonicity = np.ones(cycles) * np.nan

    for idx, row in df.iterrows():

        if 'sample_peak' in df.columns:
            rise_period = sig[int(row['sample_last_trough']):int(row['sample_peak'])]
            decay_period = sig[int(row['sample_peak']):int(row['sample_next_trough'])]

        else:
            decay_period = sig[int(row['sample_last_peak']):int(row['sample_trough'])]
            rise_period = sig[int(row['sample_trough']):int(row['sample_next_peak'])]

        decay_mono = np.mean(np.diff(decay_period) < 0)
        rise_mono = np.mean(np.diff(rise_period) > 0)
        monotonicity[idx] = np.mean([decay_mono, rise_mono])

    df['monotonicity'] = monotonicity

    # Compute if each period is part of an oscillation
    cycle_good_amp = df['amp_fraction'] > amp_fraction_threshold
    cycle_good_amp_consist = df['amp_consistency'] > amp_consistency_threshold
    cycle_good_period_consist = df['period_consistency'] > period_consistency_threshold
    cycle_good_monotonicity = df['monotonicity'] > monotonicity_threshold

    is_burst = cycle_good_amp & cycle_good_amp_consist & \
        cycle_good_period_consist & cycle_good_monotonicity
    is_burst[0] = False
    is_burst[-1] = False

    df['is_burst'] = is_burst
    df = _min_consecutive_cycles(df, n_cycles_min=n_cycles_min)
    df['is_burst'] = df['is_burst'].astype(bool)

    return df


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


def detect_bursts_df_amp(df, sig, fs, f_range, amp_threshes=(1, 2),
                         n_cycles_min=3, filter_kwargs=None):
    """Determine which cycles in a signal are part of an oscillatory
    burst using an amplitude thresholding approach.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe of waveform features for individual cycles, trough-centered.
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, Hz.
    f_range : tuple of (float, float)
        Frequency range (Hz) for oscillator of interest.
    amp_threshes : tuple (low, high), optional, default: (1, 2)
        Threshold values for determining timing of bursts.
        These values are in units of amplitude (or power, if specified) normalized to
        the median amplitude (value 1).
    n_cycles_min : int, optional, default: 3
        Minimum number of cycles to be identified as truly oscillating needed in a row in
        order for them to remain identified as truly oscillating.
    filter_kwargs : dict, optional
        Keyword arguments to :func:`~neurodsp.filt.filter.filter_signal`.

    Returns
    -------
    df : pandas DataFrame
        Same df as input, with an additional column to indicate
        if the cycle is part of an oscillatory burst.
    """

    # Detect bursts using the dual amplitude threshold approach
    sig_burst = detect_bursts_dual_threshold(sig, fs, amp_threshes, f_range,
                                             min_n_cycles=n_cycles_min, **filter_kwargs)

    # Compute fraction of each cycle that's bursting
    burst_fracs = []
    for _, row in df.iterrows():
        fraction_bursting = np.mean(sig_burst[int(row['sample_last_trough']):
                                              int(row['sample_next_trough'] + 1)])
        burst_fracs.append(fraction_bursting)

    # Determine cycles that are defined as bursting throughout the whole cycle
    df['is_burst'] = [frac == 1 for frac in burst_fracs]

    df = _min_consecutive_cycles(df, n_cycles_min=n_cycles_min)
    df['is_burst'] = df['is_burst'].astype(bool)

    return df
