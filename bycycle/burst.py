"""
burst.py
Analyze periods of oscillatory bursting in a neural signal
"""

import numpy as np
import matplotlib.pyplot as plt
from bycycle.filt import amp_by_time, bandpass_filter
from scipy.stats import zscore


def detect_bursts_cycles(df, x, amplitude_fraction_threshold=0,
                         amplitude_consistency_threshold=.5,
                         period_consistency_threshold=.5,
                         monotonicity_threshold=.8,
                         N_cycles_min=3):
    """
    Compute consistency between cycles and determine which are truly oscillating

    Parameters
    ----------
    df : pandas DataFrame
        dataframe of waveform features for individual cycles, trough-centered
    x : numpy array
        trace used to compute monotonicity
    amplitude_fraction_threshold : float (0 to 1)
        the minimum normalized amplitude a cycle must have
        in order to be considered in an oscillation.
        0 = the minimum amplitude across all cycles
        .5 = the median amplitude across all cycles
        1 = the maximum amplitude across all cycles
    amplitude_consistency_threshold : float (0 to 1)
        the minimum normalized difference in rise and decay magnitude
        to be considered as in an oscillatory mode
        1 = the same amplitude for the rise and decay
        .5 = the rise (or decay) is half the amplitude of the decay (rise)
    period_consistency_threshold : float (0 to 1)
        the minimum normalized difference in period between two adjacent cycles
        to be considered as in an oscillatory mode
        1 = the same period for both cycles
        .5 = one cycle is half the duration of another cycle
    monotonicity_threshold : float (0 to 1)
        the minimum fraction of time segments between samples that must be
        going in the same direction.
        1 = rise and decay are perfectly monotonic
        .5 = both rise and decay are rising half of the time
             and decay half the time
        0 = rise period is all decaying and decay period is all rising
    N_cycles_min : int
        minimum number of cycles to be identified as truly oscillating
        needed in a row in order for them to remain identified as
        truly oscillating

    Returns
    -------
    df : pandas DataFrame
        same df as input, with an additional column to indicate
        if the cycle is part of an oscillatory burst.
        Also additional columns indicating the burst detection
        parameters.

    Notes
    -----
    * The first and last period cannot be considered oscillating
    if the consistency measures are used.
    """

    # Compute normalized amplitude for all cycles
    df['amp_fraction'] = df['volt_amp'].rank()/len(df)

    # Compute amplitude consistency
    C = len(df)
    amp_consists = np.ones(C) * np.nan
    rises = df['volt_rise'].values
    decays = df['volt_decay'].values
    for p in range(1, C - 1):
        consist_current = np.min([rises[p], decays[p]]) / np.max([rises[p], decays[p]])
        consist_last = np.min([rises[p], decays[p - 1]]) / np.max([rises[p], decays[p - 1]])
        consist_next = np.min([rises[p + 1], decays[p]]) / np.max([rises[p + 1], decays[p]])
        amp_consists[p] = np.min([consist_current, consist_next, consist_last])
    df['amp_consistency'] = amp_consists

    # Compute period consistency
    period_consists = np.ones(C) * np.nan
    periods = df['period'].values
    for p in range(1, C - 1):
        consist_last = np.min([periods[p], periods[p - 1]]) / np.max([periods[p], periods[p - 1]])
        consist_next = np.min([periods[p + 1], periods[p]]) / np.max([periods[p + 1], periods[p]])
        period_consists[p] = np.min([consist_next, consist_last])
    df['period_consistency'] = period_consists

    # Compute monotonicity
    monotonicity = np.ones(C) * np.nan
    for i, row in df.iterrows():
        rise_period = x[int(row['sample_last_trough']):int(row['sample_peak'])]
        decay_period = x[int(row['sample_peak']):int(row['sample_next_trough'])]
        decay_mono = np.mean(np.diff(decay_period) < 0)
        rise_mono = np.mean(np.diff(rise_period) > 0)
        monotonicity[i] = np.mean([decay_mono, rise_mono])
    df['monotonicity'] = monotonicity

    # Compute if each period is part of an oscillation
    cycle_good_amp = df['amp_fraction'] > amplitude_fraction_threshold
    cycle_good_amp_consist = df['amp_consistency'] > amplitude_consistency_threshold
    cycle_good_period_consist = df['period_consistency'] > period_consistency_threshold
    cycle_good_monotonicity = df['monotonicity'] > monotonicity_threshold
    is_burst = cycle_good_amp & cycle_good_amp_consist & cycle_good_period_consist & cycle_good_monotonicity
    is_burst[0] = False
    is_burst[-1] = False
    df['is_burst'] = is_burst
    df = _min_consecutive_cycles(df, N_cycles_min=N_cycles_min)
    df['is_burst'] = df['is_burst'].astype(bool)
    return df


def _min_consecutive_cycles(df_shape, N_cycles_min=3):
    '''Enforce minimum number of consecutive cycles'''
    is_burst = np.copy(df_shape['is_burst'].values)
    temp_cycle_count = 0
    for i, c in enumerate(is_burst):
        if c:
            temp_cycle_count += 1
        else:
            if temp_cycle_count < N_cycles_min:
                for c_rm in range(temp_cycle_count):
                    is_burst[i - 1 - c_rm] = False
            temp_cycle_count = 0
    df_shape['is_burst'] = is_burst
    return df_shape


def plot_burst_detect_params(x, Fs, df_shape, osc_kwargs,
                             tlims=None, figsize=(16, 3),
                             plot_only_result=False):
    """
    Create a plot to study how the cycle-by-cycle burst detection
    algorithm determine bursting periods of a signal.

    Parameters
    ----------
    x : numpy array
        signal analyzed
    Fs : float
        sampling rate
    df_shape : pd.DataFrame
        dataframe output of `features_by_cycle`
    osc_kwargs : dict
        dictionary of thresholds for burst detection
        used in the function `features_by_cycle` using
        the kward `burst_detection_kwargs`
    tlims : tuple, length 2
        start and stop times for plot
    figsize : tuple, length 2
        size of figure

    Returns
    -------
    A figure with 5 subplots.
    In the top plot, the raw signal is plotted in black, and the
    red line indicates periods defined as oscillatory bursts.
    The highlighted regions indicate when each burst requirement
    was violated, color-coded consistently with the plots below.

    * blue: amplitude_fraction_threshold,
    * red: amplitude_consistency_threshold
    * yellow: period_consistency_threshold
    * green: monotonicity_threshold
    """

    # Normalize signal
    x = zscore(x)

    # Determine time array
    t = np.arange(0, len(x) / Fs, 1 / Fs)

    if tlims is None:
        tlims = (t[0], t[-1])

    # Determine extrema strs
    if 'sample_trough' in df_shape.columns:
        center_e = 'trough'
        side_e = 'peak'
    else:
        center_e = 'peak'
        side_e = 'trough'

    # Determine which samples are defined as bursting
    is_osc = np.zeros(len(x), dtype=bool)
    df_osc = df_shape[df_shape['is_burst']]
    for _, cyc in df_osc.iterrows():
        is_osc[cyc['sample_last_' + side_e]:cyc['sample_next_' + side_e] + 1] = True

    if plot_only_result:
        # Plot the time series and indicate peaks and troughs
        plt.figure(figsize=figsize)
        plt.plot(t, x, 'k')
        plt.plot(t[is_osc], x[is_osc], 'r.')
        plt.xlim(tlims)
        plt.tight_layout()
        plt.title('Raw z-scored signal. Red trace indicates periods of bursting', size=15)
        plt.ylim((-3, 3))
        plt.xlabel('Time (s)')
        plt.show()

    else:
        # Plot the time series and indicate peaks and troughs
        plt.figure(figsize=figsize)
        plt.plot(t, x, 'k')
        plt.plot(t[is_osc], x[is_osc], 'r', linewidth=2)
        plt.plot(t[df_shape['sample_' + center_e]], x[df_shape['sample_' + center_e]], 'm.', ms=10)
        plt.plot(t[df_shape['sample_last_' + side_e]], x[df_shape['sample_last_' + side_e]], 'c.', ms=10)
        plt.xlim(tlims)
        plt.tight_layout()
        plt.title('Raw signal with highlights indicating violations of oscillatory burst requirements')
        plt.ylim((-3, 3))
        plt.xlabel('Time (s)')

        # Highlight where burst detection parameters were violated
        # Use a different color for each burst detection parameter
        plt.fill_between(t[df_shape['sample_last_' + side_e]], min(x), max(x) + (max(x) - min(x)) * 100,
                         where=df_shape['amp_fraction'] < osc_kwargs['amplitude_fraction_threshold'],
                         interpolate=True, facecolor='blue', alpha=0.5, )
        plt.fill_between(t[df_shape['sample_last_' + side_e]], min(x), max(x) + (max(x) - min(x)) * 100,
                         where=df_shape['amp_consistency'] < osc_kwargs['amplitude_consistency_threshold'],
                         interpolate=True, facecolor='red', alpha=0.5)
        plt.fill_between(t[df_shape['sample_last_' + side_e]], min(x), max(x) + (max(x) - min(x)) * 100,
                         where=df_shape['period_consistency'] < osc_kwargs['period_consistency_threshold'],
                         interpolate=True, facecolor='yellow', alpha=0.5)
        plt.fill_between(t[df_shape['sample_last_' + side_e]], min(x), max(x) + (max(x) - min(x)) * 100,
                         where=df_shape['monotonicity'] < osc_kwargs['monotonicity_threshold'],
                         interpolate=True, facecolor='green', alpha=0.5)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=figsize)
        plt.plot(t[df_shape['sample_' + center_e]], df_shape['amp_fraction'], 'k.-')
        plt.plot(tlims, [osc_kwargs['amplitude_fraction_threshold'],
                         osc_kwargs['amplitude_fraction_threshold']], 'k--')
        plt.xlim(tlims)
        plt.ylim((-.02, 1.02))
        plt.title('Band amplitude fraction, threshold={:.02f}'.format(osc_kwargs['amplitude_fraction_threshold']))
        plt.fill_between(t[df_shape['sample_last_' + side_e]], 0, 100,
                         where=df_shape['amp_fraction'] < osc_kwargs['amplitude_fraction_threshold'],
                         interpolate=True, facecolor='blue', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=figsize)
        plt.plot(t[df_shape['sample_' + center_e]], df_shape['amp_consistency'], 'k.-')
        plt.plot(tlims, [osc_kwargs['amplitude_consistency_threshold'],
                         osc_kwargs['amplitude_consistency_threshold']], 'k--')
        plt.xlim(tlims)
        plt.ylim((-.02, 1.02))
        plt.title('Amplitude consistency, threshold={:.02f}'.format(osc_kwargs['amplitude_consistency_threshold']))
        plt.fill_between(t[df_shape['sample_last_' + side_e]], 0, 100,
                         where=df_shape['amp_consistency'] < osc_kwargs['amplitude_consistency_threshold'],
                         interpolate=True, facecolor='red', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=figsize)
        plt.plot(t[df_shape['sample_' + center_e]], df_shape['period_consistency'], 'k.-')
        plt.plot(tlims, [osc_kwargs['period_consistency_threshold'],
                         osc_kwargs['period_consistency_threshold']], 'k--')
        plt.xlim(tlims)
        plt.title('Period consistency, threshold={:.02f}'.format(osc_kwargs['period_consistency_threshold']))
        plt.fill_between(t[df_shape['sample_last_' + side_e]], 0, 100,
                         where=df_shape['period_consistency'] < osc_kwargs['period_consistency_threshold'],
                         interpolate=True, facecolor='yellow', alpha=0.5)
        plt.ylim((-.02, 1.02))
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=figsize)
        plt.plot(t[df_shape['sample_' + center_e]], df_shape['monotonicity'], 'k.-')
        plt.plot(tlims, [osc_kwargs['monotonicity_threshold'],
                         osc_kwargs['monotonicity_threshold']], 'k--')
        plt.xlim(tlims)
        plt.title('Monotonicity, threshold={:.02f}'.format(osc_kwargs['monotonicity_threshold']))
        plt.fill_between(t[df_shape['sample_last_' + side_e]], 0, 100,
                         where=df_shape['monotonicity'] < osc_kwargs['monotonicity_threshold'],
                         interpolate=True, facecolor='green', alpha=0.5)
        plt.ylim((-.02, 1.02))
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()


def detect_bursts_df_amp(df, x, Fs, f_range,
                         amp_threshes=(1, 2), N_cycles_min=3,
                         filter_kwargs=None):
    """

    Determine which cycles in a signal are part of an oscillatory
    burst using an amplitude thresholding approach

    Parameters
    ----------
    df : pandas DataFrame
        dataframe of waveform features for individual cycles, trough-centered
    x : numpy array
    	trace used to compute monotonicity
    Fs : float
        The sampling rate in Hz
    f_range : tuple (low, high), Hz
        frequency range for oscillator of interest
    amp_threshes : tuple (low, high)
        Threshold values for determining timing of bursts.
        These values are in units of amplitude
        (or power, if specified) normalized to the median
        amplitude (value 1).
    N_cycles_min : int
        minimum number of cycles to be identified as truly oscillating
        needed in a row in order for them to remain identified as
        truly oscillating
    filter_kwargs : dict
        keyword arguments to filt.bandpass_filter

    Returns
    -------
    df : pandas DataFrame
        same df as input, with an additional column to indicate
        if the cycle is part of an oscillatory burst
    """

    # Detect bursts using the dual amplitude threshold approach
    x_burst = twothresh_amp(x, Fs, f_range, amp_threshes,
                            N_cycles_min=N_cycles_min,
                            filter_kwargs=filter_kwargs)

    # Compute fraction of each cycle that's bursting
    burst_fracs = []
    for i, row in df.iterrows():
        fraction_bursting = np.mean(x_burst[int(row['sample_last_trough']):
                                            int(row['sample_next_trough'] + 1)])
        burst_fracs.append(fraction_bursting)

    # Determine cycles that are defined as bursting throughout the whole cycle
    df['is_burst'] = [x == 1 for x in burst_fracs]

    df = _min_consecutive_cycles(df, N_cycles_min=N_cycles_min)
    df['is_burst'] = df['is_burst'].astype(bool)

    return df


def twothresh_amp(x, Fs, f_range, amp_threshes, N_cycles_min=3,
                  magnitude_type='amplitude',
                  return_amplitude=False,
                  filter_kwargs=None):
    """
    Detect periods of oscillatory bursting in a neural signal
    by using two amplitude thresholds.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        The sampling rate in Hz
    f_range : tuple (low, high), Hz
        frequency range for oscillator of interest
    amp_threshes : tuple (low, high)
        Threshold values for determining timing of bursts.
        These values are in units of amplitude
        (or power, if specified) normalized to the median
        amplitude (value 1).
    N_cycles_min : float
        minimum burst duration in terms of number of cycles of f_range[0]
    magnitude_type : string in ('power', 'amplitude')
        metric of magnitude used for thresholding
    filter_kwargs : dict
        keyword arguments to filt.bandpass_filter
    """

    # Set default filtering parameters
    if filter_kwargs is None:
        filter_kwargs = {}

    # Assure the amp_threshes is a tuple of length 2
    if len(amp_threshes) != 2:
        raise ValueError(
            "Invalid number of elements in 'amp_threshes' parameter")

    # Compute amplitude time series
    x_amplitude = amp_by_time(x, Fs, f_range, filter_kwargs=filter_kwargs,
                              remove_edge_artifacts=False)

    # Set magnitude as power or amplitude
    if magnitude_type == 'power':
        x_magnitude = x_amplitude**2
    elif magnitude_type == 'amplitude':
        x_magnitude = x_amplitude
    else:
        raise ValueError("Invalid 'magnitude' parameter")

    # Rescale magnitude by median
    x_magnitude = x_magnitude / np.median(x_magnitude)

    # Identify time periods of oscillation using the 2 thresholds
    isosc = _2threshold_split(x_magnitude, amp_threshes[1], amp_threshes[0])

    # Remove short time periods of oscillation
    min_period_length = int(np.ceil(N_cycles_min * Fs / f_range[0]))
    isosc_noshort = _rmv_short_periods(isosc, min_period_length)

    if return_amplitude:
        return isosc_noshort, x_magnitude
    else:
        return isosc_noshort


def _2threshold_split(x, thresh_hi, thresh_lo):
    """
    Identify periods of a time series that are above thresh_lo and have at
    least one value above thresh_hi
    """

    # Find all values above thresh_hi
    # To avoid bug in later loop, do not allow first or last index to start
    # off as 1
    x[[0, -1]] = 0
    idx_over_hi = np.where(x >= thresh_hi)[0]

    # Initialize values in identified period
    positive = np.zeros(len(x))
    positive[idx_over_hi] = 1

    # Iteratively test if a value is above thresh_lo if it is not currently in
    # an identified period
    lenx = len(x)
    for i in idx_over_hi:
        j_down = i - 1
        if positive[j_down] == 0:
            j_down_done = False
            while j_down_done is False:
                if x[j_down] >= thresh_lo:
                    positive[j_down] = 1
                    j_down -= 1
                    if j_down < 0:
                        j_down_done = True
                else:
                    j_down_done = True

        j_up = i + 1
        if positive[j_up] == 0:
            j_up_done = False
            while j_up_done is False:
                if x[j_up] >= thresh_lo:
                    positive[j_up] = 1
                    j_up += 1
                    if j_up >= lenx:
                        j_up_done = True
                else:
                    j_up_done = True

    return positive


def _rmv_short_periods(x, N):
    """Remove periods that ==1 for less than N samples"""

    if np.sum(x) == 0:
        return x

    osc_changes = np.diff(1 * x)
    osc_starts = np.where(osc_changes == 1)[0]
    osc_ends = np.where(osc_changes == -1)[0]

    if len(osc_starts) == 0:
        osc_starts = [0]
    if len(osc_ends) == 0:
        osc_ends = [len(osc_changes)]

    if osc_ends[0] < osc_starts[0]:
        osc_starts = np.insert(osc_starts, 0, 0)
    if osc_ends[-1] < osc_starts[-1]:
        osc_ends = np.append(osc_ends, len(osc_changes))

    osc_length = osc_ends - osc_starts
    osc_starts_long = osc_starts[osc_length >= N]
    osc_ends_long = osc_ends[osc_length >= N]

    is_osc = np.zeros(len(x))
    for osc in range(len(osc_starts_long)):
        is_osc[osc_starts_long[osc]:osc_ends_long[osc]] = 1
    return is_osc
