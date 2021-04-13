"""Dataframe and spike segmentation functions."""

import numpy as np
import pandas as pd

from bycycle.cyclepoints import find_zerox
from bycycle.utils.dataframes import get_extrema_df

###################################################################################################
###################################################################################################

def slice_spikes(bm, std=2):
    """Create a samples dataframe from isolated spikes.

    Parameters
    ----------
    bm : bycycle.Bycycle
        A bycycle model that has been successfully fit.
    std : float or int
        The standard deviation used to identify spikes.

    Returns
    -------
    df_features : pd.DataFrame
        A dataframe containing cyclepoint locations, as samples indices, for each spike.
    spikes : 2d array
        The signal associated with each spike (row in the ``df_features``).
    """

    # Infer extrema
    center_e, side_e = get_extrema_df(bm.df_features)

    # Determine sub-(standard deviation) extrema (i.e. spikes as large deviations from mean)
    volt_center = bm.df_features['volt_' + center_e].values

    if center_e == 'trough':
        thresh = volt_center.mean() - (volt_center.std() * std)
        cycle_idxs = np.array([[idx-1, idx, idx+1] for idx in np.where(volt_center < thresh)[0]
                               if idx-1 >= 0 and idx+1 < len(bm.df_features)])
    elif center_e == 'peak':
        thresh = volt_center.mean() + (volt_center.std() * std)
        cycle_idxs = np.array([[idx-1, idx, idx+1] for idx in np.where(volt_center > thresh)[0]
                               if idx-1 >= 0 and idx+1 < len(bm.df_features)])

    # Infer cyclespoints from bycycle dataframe
    starts = bm.df_features.iloc[cycle_idxs[:, 0]]['sample_zerox_rise'].values
    centers =  bm.df_features.iloc[cycle_idxs[:, 1]]['sample_' + center_e].values
    ends = bm.df_features.iloc[cycle_idxs[:, 2]]['sample_' + center_e].values
    next_sides = bm.df_features.iloc[cycle_idxs[:, 1]]['sample_next_' + side_e].values

    # Maximum pre-extrema and post-extrema lengths
    left_max = (centers-starts).max()
    right_max = (ends-centers).max()

    # Init 2d spike array
    spikes = np.zeros((len(starts), sum([left_max, right_max])))
    spikes[:] = np.nan

    # Store cyclepoints as 2d array of (start, center, left side, end, decay, rise, next_decay)
    cps = zip(starts, centers,  next_sides, ends)
    cps_adjusted = np.zeros((len(centers), 7), dtype=int)

    # Store where final spikes are subthresh
    drop_idxs = np.zeros(len(starts), dtype='bool')

    for idx, (start, center, side, end) in enumerate(cps):

        # Fill the spike from the signal
        pad_left = left_max - (center-start)
        pad_right = len(spikes[idx]) - (right_max - (end-center))
        spikes[idx][pad_left:pad_right] = bm.sig[start:end]

        # Determine the right inflection point
        #   (first derivative == 0) of the repolarization phase
        compare = np.greater_equal if center_e == 'trough' else np.less_equal

        extrema_right = pad_right - (end-side)
        trail_flank = np.diff(spikes[idx][extrema_right:pad_right])
        trail_flank = np.where(compare(trail_flank, 0))[0]

        inflection_pt = pad_right-extrema_right-1 if len(trail_flank) == 0 else trail_flank[0]
        inflection_pt += extrema_right

        # Adjust the next side point to match the location in the 2d spike array
        adj_next_side = pad_left + (side - start)
        adj_center = pad_left + (center - start)

        # Determine the left rising/decaying point
        spike_inv = np.flip(spikes[idx][:adj_center+1])
        diff_sign = np.sign(np.diff(spike_inv))

        compare = (-1, 1) if center_e == 'trough' else (1, -1)
        new_start = np.where((diff_sign[:-1] == compare[0]) &
                             (diff_sign[1:] == compare[1]))[0]

        # Minus one to account for diff slicing
        adj_start = pad_left if len(new_start) == 0 else adj_center - new_start[0] - 1

        # Everything to the right/left of the inflection/start gets nan
        spikes[idx][inflection_pt+1:] = np.nan
        spikes[idx][:adj_start] = np.nan

        cps_adjusted[idx] = [adj_start, adj_center, adj_next_side, inflection_pt,
                             0, 0, 0]

        # Mark poor spikes (i.e. the center point should be the only voltage outlier)
        spike_mean = np.nanmean(spikes[idx])
        spike_std = np.nanstd(spikes[idx])

        spike_thresh = spike_mean - (std * spike_std) if center_e == 'trough' \
            else spike_mean + (std * spike_std)

        compare = np.greater if center_e == 'peak' else np.less
        for sample in [adj_start, adj_next_side, inflection_pt]:
            if compare(spikes[idx][sample], spike_thresh):
                drop_idxs[idx] = True
                break

        compare = np.less_equal if center_e == 'peak' else np.greater_equal
        if compare(spikes[idx][adj_center], spike_thresh):
            drop_idxs[idx] = True

    # Remove unstable spikes
    spikes = spikes[~drop_idxs]
    cps_adjusted = cps_adjusted[~drop_idxs]

    mask = ~np.isnan(spikes).all(axis=0)
    spikes = spikes[:, mask]

    # Shift to account for spike removal
    mask_shift = len(mask[:np.where(mask)[0][0]])
    cps_adjusted = cps_adjusted - mask_shift

    # Get zero-crossings
    for idx, (spike, cps) in enumerate(zip(spikes, cps_adjusted)):

        rises, decays = get_spike_zerox(spike, [cps[0], cps[2]], [cps[1], cps[3]], center='trough')

        cps[-3:] = [decays[0], rises[0], decays[1]]

    # Move cps to a dataframe
    df_features = pd.DataFrame()

    df_features['sample_last_rise'] = cps_adjusted[:, 0]
    df_features['sample_trough'] = cps_adjusted[:, 1]
    df_features['sample_next_peak'] = cps_adjusted[:, 2]
    df_features['sample_next_decay'] = cps_adjusted[:, 3]
    df_features['sample_zerox_decay'] = cps_adjusted[:, 4]
    df_features['sample_zerox_rise'] = cps_adjusted[:, 5]
    df_features['sample_next_zerox_decay'] = cps_adjusted[:, 6]

    return df_features, spikes


def get_spike_zerox(spike, peaks, troughs, center='extrema'):
    """Utility function to find zero-crossings in spikes."""

    # Get zero-crossings
    #   Note: order of params is flipped when inverting signal
    if center == 'trough':
        decays, rises = find_zerox(-spike, troughs, peaks)
    elif center == 'peak':
        rises, decays = find_zerox(spike, troughs, peaks)

    return rises, decays
