""""""

import numpy as np

from bycycle.cyclepoints import find_extrema, find_zerox
from bycycle.spikes.utils import create_cyclepoints_df

###################################################################################################
###################################################################################################

def compute_spike_cyclepoints(sig, fs, f_range, std=2):
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

    # Find troughs
    _, _troughs = find_extrema(sig, fs, f_range, first_extrema=None, pass_type='bandpass')

    # Threshold troughs
    thresh = np.mean(sig) - np.std(sig) * std

    volt_troughs = sig[_troughs]

    idxs = np.where(volt_troughs < thresh)[0]

    if len(idxs) == 0:
        raise ValueError('No spikes found outside of std. Try reducing std.')

    troughs = _troughs[idxs]

    drop_idxs = np.zeros_like(troughs, dtype='bool')

    # Determine spike starting points
    starts = np.zeros_like(troughs)
    for idx, trough in enumerate(troughs):

        sig_reflect = np.flip(sig[:trough])

        reflect_diff = np.where(np.diff(sig_reflect) < 0)[0]

        if len(reflect_diff) != 0:
            starts[idx] = trough - reflect_diff[0]
        else:
            drop_idxs[idx] = True

    # Determine next peak and next decay points
    next_peaks = np.zeros_like(troughs)
    ends = np.zeros_like(troughs)

    right_edges = np.append(starts[1:], len(sig)).astype(int)

    for idx, right_edge in enumerate(right_edges):

        forward_diff = np.diff(sig[troughs[idx]:right_edge])

        if len(forward_diff) == 0:
            drop_idxs[idx] = True
            continue

        # Find next peaks
        next_peak = np.where(forward_diff < 0)[0]
        if len(next_peak) == 0:
            drop_idxs[idx] = True
            continue
        next_peaks[idx] = troughs[idx] + next_peak[0]

        # Find next decays
        next_peak_idx = np.argwhere(forward_diff < 0)[0][0]
        next_decay = forward_diff[next_peak_idx:]

        next_decay = np.where(next_decay > 0)[0]
        if len(next_decay) == 0:
            drop_idxs[idx] == True
            continue

        ends[idx] = troughs[idx] + next_peak_idx + next_decay[0]

    # Drop monotonic spikes
    starts = starts[~drop_idxs]
    troughs = troughs[~drop_idxs]
    next_peaks = next_peaks[~drop_idxs]
    ends = ends[~drop_idxs]

    decays, rises, next_decays = _compute_spike_zerox(-sig, starts, troughs, next_peaks, ends)

    # Oraganize points into a dataframe
    df_samples = create_cyclepoints_df(sig, starts, decays, troughs, rises,
                                       next_peaks, next_decays, ends)

    # Apply standard deviation thresholding
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

        # Check center point is outside std
        spike_mean = sig[side_pts[0]:side_pts[-1]].mean()
        spike_std = sig[side_pts[0]:side_pts[-1]].std()
        spike_thresh = spike_mean - (std * spike_std)

        if trough_volt >= spike_thresh:
            drop_idxs[idx] = True

    return drop_idxs
