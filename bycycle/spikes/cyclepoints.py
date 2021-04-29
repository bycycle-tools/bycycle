"""Compute spike cyclespoints functions."""

import numpy as np

from bycycle.cyclepoints import find_extrema, find_zerox
from bycycle.spikes.utils import create_cyclepoints_df

###################################################################################################
###################################################################################################

def compute_spike_cyclepoints(sig, fs, f_range, std=2, prune=False):
    """Find spike cyclepoints.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest (Hz).
    std : float or int, optional, default: 1.5
        The standard deviation threshold used to identify spikes.
    prune : bool, optional, default: False
        Remove spikes with high variablility in non-trough peaks.

    Returns
    -------
    df_samples : pd.DataFrame
        Contains cycle points locations, in samples:

        - sample_start : spike start location.
        - sample_decay : spike zero-crossing decay location.
        - sample_trough : spike minima location.
        - sample_rise : spike zero-crossing rise location.
        - sample_next_peak : spike maxima location.
        - sample_next_decay : spike zero-crossing decay location, after the peak.
        - sample_end : spike end location.

    """

    times = np.arange(0, len(sig)/fs, 1/fs)

    # Find troughs
    _, _troughs = find_extrema(sig, fs, f_range, first_extrema=None, pass_type='bandpass')

    # Threshold troughs
    thresh = np.mean(sig) - np.std(sig) * std

    volt_troughs = sig[_troughs]

    idxs = np.where(volt_troughs < thresh)[0]

    if len(idxs) == 0:
        raise ValueError('No spikes found outside of std. Try reducing std.')

    troughs = _troughs[idxs]
    volt_troughs = sig[troughs]

    drop_idxs = np.zeros_like(troughs, dtype='bool')

    # Determine spike starting points
    starts = np.zeros_like(troughs)
    for idx, trough in enumerate(troughs):

        sig_reflect = np.flip(sig[:trough])

        # Ignore the previous extrema
        last_extrema = np.where(np.diff(sig_reflect) < 0)[0]
        if len(last_extrema) == 0:
            drop_idxs[idx] = True
            continue

        sig_reflect = sig_reflect[last_extrema[0]:]

        reflect_diff = np.where(np.diff(sig_reflect) > 0)[0]

        if len(reflect_diff) != 0:
            starts[idx] = trough - reflect_diff[0] - last_extrema[0] - 1
        else:
            drop_idxs[idx] = True
            continue

        # Trim outlier start voltages (i.e. a two extrema spike vs three extrema spike)
        volt_start = sig[starts[idx]]
        volt_last_peak = sig[trough - last_extrema[0] - 1]

        if abs(volt_troughs[idx] - volt_start) < abs(volt_last_peak- volt_start):
            starts[idx] = trough - last_extrema[0] - 1

    # Determine next peak and next decay (end) points
    next_peaks = np.zeros_like(troughs)
    ends = np.zeros_like(troughs)

    right_edges = np.append(starts[1:], len(sig)).astype(int)

    for idx, right_edge in enumerate(right_edges):

        if drop_idxs[idx]:
            continue

        sig_post_trough = sig[troughs[idx]:right_edge+2]
        forward_diff = np.diff(sig_post_trough)

        if len(forward_diff) == 0:
            drop_idxs[idx] = True
            continue

        # Find next peaks: volt_thresh prevents small diffences between peak and inflection point
        #   by requiring a slope < -.75 between the two points
        volt_slope_thresh = -.75

        post_trough_diff = np.where(forward_diff < 0)[0]
        post_trough_decay = np.split(post_trough_diff,
                                     np.where(np.diff(post_trough_diff) != 1)[0]+1)

        if len(post_trough_decay[0]) == 0:
            ends[idx] = right_edge
            next_peaks[idx] = right_edge
            continue

        post_trough_slopes = np.zeros(len(post_trough_decay))
        for decay_idx, decay in enumerate(post_trough_decay):

            if decay[0] == decay[-1]:
                post_trough_slopes[decay_idx] = 0
            else:
                volt_next_peak = sig_post_trough[decay[0]]
                volt_end = sig_post_trough[decay[-1]+1]

                post_trough_slopes[decay_idx] = (volt_end - volt_next_peak) / \
                    ((troughs[idx] + decay[-1]+1) - (troughs[idx] + decay[0]))

        next_peak_idx = np.where(post_trough_slopes < volt_slope_thresh)[0]

        if len(next_peak_idx) == 0:
            ends[idx] = post_trough_decay[0][0] + troughs[idx]
            next_peaks[idx] = ends[idx]
            continue

        next_peak_idx = next_peak_idx[0]
        next_peak = post_trough_decay[next_peak_idx][0]

        next_peaks[idx] = troughs[idx] + next_peak

        # Find next decays
        next_decay = forward_diff[next_peak+1:]
        next_decay = np.where(next_decay >= 0)[0]

        if len(next_decay) == 0:
            ends[idx] = troughs[idx] + next_peak
        elif len(next_decay) > 0:

            volt_next_decay = sig[troughs[idx] + next_peak + next_decay[0]]
            volt_next_peak = sig[troughs[idx] + next_peak]
            volt_trough = volt_troughs[idx]

            if abs(volt_next_decay- volt_trough) > abs(volt_next_decay - volt_next_peak):
                ends[idx] = troughs[idx] + next_peak + next_decay[0] + 1
            else:
                ends[idx] = troughs[idx] + next_peak

        # Current spike overlaps with next spike, take the larger of the two
        if idx < len(starts)-1 and troughs[idx] + next_peak == starts[idx+1]:
            if volt_troughs[idx] < volt_troughs[idx+1]:
                drop_idxs[idx+1] = True
                ends[idx] = troughs[idx+1]
            else:
                drop_idxs[idx] = True
            continue

    # Drop invalid spikes
    starts = starts[~drop_idxs]
    troughs = troughs[~drop_idxs]
    next_peaks = next_peaks[~drop_idxs]
    ends = ends[~drop_idxs]

    decays, rises, next_decays = _compute_spike_zerox(-sig, starts, troughs, next_peaks, ends)

    # Oraganize points into a dataframe
    df_samples = create_cyclepoints_df(sig, starts, decays, troughs, rises,
                                       next_peaks, next_decays, ends)

    # Remove spikes with high variablility in non-trough peaks.
    if prune:
        drop_idxs = _prune_spikes(df_samples, sig, std)
        df_samples = df_samples.iloc[~drop_idxs]
        df_samples.reset_index(inplace=True, drop=True)

    return df_samples


def _compute_spike_zerox(sig, starts, troughs, next_peaks, ends):
    """Find spike zero-crossings"""

    decays = np.zeros_like(troughs)
    rises = np.zeros_like(troughs)
    next_decays = np.zeros_like(troughs)

    for idx in range(len(troughs)):

        decay, rise = find_zerox(-sig, [troughs[idx], ends[idx]],
                                 [starts[idx], next_peaks[idx]])

        decays[idx] = decay[0]
        rises[idx] = rise[0]
        next_decays[idx] = decay[1]

    return decays, rises, next_decays


def _prune_spikes(df_samples, sig, std=2):
    """Remove spikes with outlier cyclepoints."""

    drop_idxs = np.zeros(len(df_samples), dtype=bool)

    for idx, row in df_samples.iterrows():

        # Check non-trough voltages are closer to one another than the trough
        trough_volt = sig[int(row['sample_trough'])]

        side_pts = [int(row['sample_start']), int(row['sample_next_peak']),
                    int(row['sample_end'])]

        for cyc_idx, sample in enumerate(side_pts):

            curr_volt = sig[sample]

            others = [0, 1, 2]
            del others[cyc_idx]

            for other in others:

                other_volt = sig[side_pts[other]]

                if np.abs(curr_volt - trough_volt) < np.abs(curr_volt - other_volt):
                    drop_idxs[idx] = True
                    continue

    return drop_idxs
