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

    # Find troughs
    _peaks, _troughs = find_extrema(sig, fs, f_range, first_extrema='trough', pass_type='bandpass')

    # Threshold troughs
    thresh = np.mean(sig) - np.std(sig) * std

    volt_troughs = sig[_troughs]

    idxs = np.where(volt_troughs < thresh)[0]

    if len(idxs) == 0:
        raise ValueError('No spikes found outside of std. Try reducing std.')

    # Index cyclepoints
    troughs = _troughs[idxs]
    volt_troughs = sig[troughs]

    last_peaks = _peaks[idxs-1]
    next_peaks = _peaks[idxs]

    last_troughs = _troughs[idxs-1]
    next_troughs = _troughs[idxs+1]

    # Determine spike starting points
    starts = np.zeros_like(troughs)
    ends = np.zeros_like(troughs)

    std_sig = np.std(sig)

    volt_troughs = sig[troughs]
    volt_last_peaks = sig[last_peaks]
    volt_last_troughs = sig[last_troughs]
    volt_next_troughs = sig[next_troughs]
    volt_next_peaks = sig[next_peaks]
    volts = zip(volt_last_troughs, volt_last_peaks, volt_troughs, volt_next_peaks, volt_next_troughs)

    for idx, (last_trough, last_peak, trough, next_peak, next_trough) in enumerate(volts):

        # Criteria to defining start/end inflections:
        #   a) Height from sideextrema to infecltions > 0.5 stdev
        #   b) Inflection voltage must be closer to side peak than trough

        pre_norm_diff = (last_peak - last_trough) / std_sig
        post_norm_diff = (next_trough - next_peak) / std_sig

        pre_criteria_b = (last_trough - trough) > (last_peak - last_trough)
        post_criteria_b = (next_trough - trough) > (next_peak - next_trough)

        starts[idx] = last_troughs[idx] if pre_norm_diff > 0.5 and pre_criteria_b \
            else last_peaks[idx]
        ends[idx] =  next_troughs[idx] if post_norm_diff < 0.5 and post_criteria_b \
            else next_peaks[idx]

    decays, rises, next_decays = _compute_spike_zerox(-sig, starts, troughs, next_peaks, ends)

    # Oraganize points into a dataframe
    df_samples = create_cyclepoints_df(sig, starts, decays, troughs, rises,
                                       last_peaks, next_peaks, next_decays, ends)

    # Drop overlapping spikes, favoring larger voltage
    drop_spikes = np.ones(len(df_samples), dtype=bool)

    for idx in range(len(df_samples)-1):

        if drop_spikes[idx]:
            continue

        curr_end = df_samples.iloc[idx]['sample_end']
        curr_volt_trough = volt_troughs[idx]

        next_start = df_samples.iloc[idx+1]['sample_start']
        next_volt_trough = volt_troughs[idx + 1]

        if curr_end > next_start and curr_volt_trough > next_volt_trough:
            drop_spikes[idx] = False
        elif curr_end > next_start and curr_volt_trough < next_volt_trough:
            drop_spikes[idx+1] = False

    df_samples = df_samples[drop_spikes]
    df_samples.reset_index(inplace=True, drop=True)

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
