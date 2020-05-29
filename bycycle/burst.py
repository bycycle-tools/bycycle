"""Analyze periods of oscillatory bursting in neural signals."""

import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt

from neurodsp.burst import detect_bursts_dual_threshold

###################################################################################################
###################################################################################################

pd.options.mode.chained_assignment = None

def detect_bursts_cycles(df, sig, amplitude_fraction_threshold=0.,
                         amplitude_consistency_threshold=.5,
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
    cycle_good_amp = df['amp_fraction'] > amplitude_fraction_threshold
    cycle_good_amp_consist = df['amp_consistency'] > amplitude_consistency_threshold
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


def plot_burst_detect_params(sig, fs, df_shape, osc_kwargs, tlims=None,
                             figsize=(16, 3), plot_only_result=False):
    """Create a plot to study how the cycle-by-cycle burst detection
    algorithm determine bursting periods of a signal.

    Parameters
    ----------
    sig : 1d array
        Time series analyzed to compute `df_shape`.
    fs : float
        Sampling rate, in Hz.
    df_shape : pandas DataFrame
        Dataframe output of `features.compute_features()`.
    osc_kwargs : dict
        Dictionary of thresholds for burst detection used in the function
        `features.compute_features()` using the kwarg `burst_detection_kwargs`.
    tlims : tuple of (float, float), optional
        Start and stop times for plot.
    figsize : tuple of (float, float), optional
        Size of figure.
    plot_only_result : bool, optional, default: False
        If True, do not plot the subplots showing the parameters.

    Returns
    -------
    ax : matplotlib axis handle or list of axis handles
        If `plot_only_result` = True: return a plot of the burst
        detection in which periods with bursts are denoted in red.

        If `plot_only_result` = False: return a list of the fig
        handle followed by the 5 axes.

        In the top plot, the raw signal is plotted in black, and the
        red line indicates periods defined as oscillatory bursts.
        The highlighted regions indicate when each burst requirement
        was violated, color-coded consistently with the plots below.

        - blue: amplitude_fraction_threshold,
        - red: amplitude_consistency_threshold
        - yellow: period_consistency_threshold
        - green: monotonicity_threshold
    """

    # Normalize signal
    sig = zscore(sig)

    # Determine time array
    times = np.arange(0, len(sig) / fs, 1 /fs)

    if tlims is None:
        tlims = (times[0], times[-1])

    # Determine extrema labels
    if 'sample_trough' in df_shape.columns:
        center_e = 'trough'
        side_e = 'peak'
    else:
        center_e = 'peak'
        side_e = 'trough'

    # Limit to time periods of interest
    tidx = np.logical_and(times >= tlims[0], times < tlims[1])
    sig = sig[tidx]
    times = times[tidx]
    df_shape = df_shape[(df_shape['sample_last_' + side_e] > int(fs * tlims[0])) &
                        (df_shape['sample_next_' + side_e] < int(fs * tlims[1]))]
    df_shape['sample_last_' + side_e] = df_shape['sample_last_' + side_e] - int(fs * tlims[0])
    df_shape['sample_next_' + side_e] = df_shape['sample_next_' + side_e] - int(fs * tlims[0])
    df_shape['sample_' + center_e] = df_shape['sample_' + center_e] - int(fs * tlims[0])

    # Determine which samples are defined as bursting
    is_osc = np.zeros(len(sig), dtype=bool)
    df_osc = df_shape[df_shape['is_burst']]
    for _, cyc in df_osc.iterrows():
        samp_start_burst = cyc['sample_last_' + side_e]
        samp_end_burst = cyc['sample_next_' + side_e] + 1
        is_osc[samp_start_burst:samp_end_burst] = True

    if plot_only_result:

        # Plot the time series and indicate peaks and troughs
        _, ax = plt.subplots(figsize=figsize)

        ax.plot(times, sig, 'k')
        ax.plot(times[is_osc], sig[is_osc], 'r.')
        ax.set_xlim(tlims)
        ax.set_title('Raw z-scored signal. Red trace indicates periods of bursting', size=15)
        ax.set_xlabel('Time (s)', size=12)
        ax.set_ylabel('Voltage (normalized)', size=12)

        return ax

    else:

        # Plot the time series and indicate peaks and troughs
        fig = plt.figure(figsize=(figsize[0], 5*figsize[1]))
        ax1 = fig.add_subplot(5, 1, 1)
        ax1.plot(times, sig, 'k')
        ax1.plot(times[is_osc], sig[is_osc], 'r', linewidth=2)
        ax1.plot(times[df_shape['sample_' + center_e]], sig[df_shape['sample_' + center_e]],
                 'm.', ms=10)
        ax1.plot(times[df_shape['sample_last_' + side_e]], sig[df_shape['sample_last_' + side_e]],
                 'c.', ms=10)
        ax1.set_xlim(tlims)
        ax1.set_xticks([])
        ax1.set_ylabel('Black: Raw signal\nRed: oscillatory periods')
        ax1.set_ylim((-4, 4))

        # Highlight where burst detection parameters were violated
        # Use a different color for each burst detection parameter
        ax1.fill_between(times[df_shape['sample_last_' + side_e]], -4, 400,
                         where=df_shape['amp_fraction'] < \
                            osc_kwargs['amplitude_fraction_threshold'],
                         interpolate=True, facecolor='blue', alpha=0.5, )
        ax1.fill_between(times[df_shape['sample_last_' + side_e]], -4, 400,
                         where=df_shape['amp_consistency'] < \
                            osc_kwargs['amplitude_consistency_threshold'],
                         interpolate=True, facecolor='red', alpha=0.5)
        ax1.fill_between(times[df_shape['sample_last_' + side_e]], -4, 400,
                         where=df_shape['period_consistency'] < \
                            osc_kwargs['period_consistency_threshold'],
                         interpolate=True, facecolor='yellow', alpha=0.5)
        ax1.fill_between(times[df_shape['sample_last_' + side_e]], -4, 400,
                         where=df_shape['monotonicity'] < \
                            osc_kwargs['monotonicity_threshold'],
                         interpolate=True, facecolor='green', alpha=0.5)

        ax2 = fig.add_subplot(5, 1, 2)
        ax2.plot(times[df_shape['sample_' + center_e]], df_shape['amp_fraction'], 'k.-')
        ax2.plot(tlims, [osc_kwargs['amplitude_fraction_threshold'],
                         osc_kwargs['amplitude_fraction_threshold']], 'k--')
        ax2.set_xlim(tlims)
        ax2.set_xticks([])
        ax2.set_ylim((-.02, 1.02))
        ax2.set_ylabel('Band amplitude fraction\nthreshold={:.02f}'
                       .format(osc_kwargs['amplitude_fraction_threshold']))
        ax2.fill_between(times[df_shape['sample_last_' + side_e]], 0, 100,
                         where=df_shape['amp_fraction'] < \
                         osc_kwargs['amplitude_fraction_threshold'],
                         interpolate=True, facecolor='blue', alpha=0.5)

        ax3 = fig.add_subplot(5, 1, 3)
        ax3.plot(times[df_shape['sample_' + center_e]], df_shape['amp_consistency'], 'k.-')
        ax3.plot(tlims, [osc_kwargs['amplitude_consistency_threshold'],
                         osc_kwargs['amplitude_consistency_threshold']], 'k--')
        ax3.set_xlim(tlims)
        ax3.set_xticks([])
        ax3.set_ylim((-.02, 1.02))
        ax3.set_ylabel('Amplitude consistency\nthreshold={:.02f}'.
                       format(osc_kwargs['amplitude_consistency_threshold']))
        ax3.fill_between(times[df_shape['sample_last_' + side_e]], 0, 100,
                         where=df_shape['amp_consistency'] < \
                            osc_kwargs['amplitude_consistency_threshold'],
                         interpolate=True, facecolor='red', alpha=0.5)

        ax4 = fig.add_subplot(5, 1, 4)
        ax4.plot(times[df_shape['sample_' + center_e]], df_shape['period_consistency'], 'k.-')
        ax4.plot(tlims, [osc_kwargs['period_consistency_threshold'],
                         osc_kwargs['period_consistency_threshold']], 'k--')
        ax4.set_xlim(tlims)
        ax4.set_xticks([])
        ax4.set_ylabel('Period consistency\nthreshold={:.02f}'
                       .format(osc_kwargs['period_consistency_threshold']))
        ax4.set_ylim((-.02, 1.02))
        ax4.fill_between(times[df_shape['sample_last_' + side_e]], 0, 100,
                         where=df_shape['period_consistency'] < \
                         osc_kwargs['period_consistency_threshold'],
                         interpolate=True, facecolor='yellow', alpha=0.5)

        ax5 = fig.add_subplot(5, 1, 5)
        ax5.plot(times[df_shape['sample_' + center_e]], df_shape['monotonicity'], 'k.-')
        ax5.plot(tlims, [osc_kwargs['monotonicity_threshold'],
                         osc_kwargs['monotonicity_threshold']], 'k--')
        ax5.set_xlim(tlims)
        ax5.set_ylabel('Monotonicity\nthreshold={:.02f}'
                       .format(osc_kwargs['monotonicity_threshold']))
        ax5.set_ylim((-.02, 1.02))
        ax5.set_xlabel('Time (s)', size=20)
        ax5.fill_between(times[df_shape['sample_last_' + side_e]], 0, 100,
                         where=df_shape['monotonicity'] < osc_kwargs['monotonicity_threshold'],
                         interpolate=True, facecolor='green', alpha=0.5)

        return [fig, ax1, ax2, ax3, ax4, ax5]


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
