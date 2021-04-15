"""Dataframe and spike segmentation functions."""

import numpy as np
import pandas as pd

from bycycle.cyclepoints import find_zerox
from bycycle.utils.dataframes import get_extrema_df

###################################################################################################
###################################################################################################

def slice_spikes(bm, std=1.5):
    """Create a samples dataframe from isolated spikes.

    Parameters
    ----------
    bm : bycycle.Bycycle
        A bycycle model that has been successfully fit.
    std : float or int, optional, default: 1.5
        The standard deviation used to identify spikes.

    Returns
    -------
    df_features : pd.DataFrame
        A dataframe containing cyclepoint locations, as samples indices, for each spike.
    spikes : 2d array
        The signal associated with each spike (row in the ``df_features``).

    Notes
    -----
    This function assumes bm has been fit with ``center_extrema=trough``.
    """

    # Infer extrema
    center_e, side_e = get_extrema_df(bm.df_features)

    # Determine sub-(standard deviation) extrema (i.e. spikes as large deviations from mean)
    volt_center = bm.df_features['volt_' + center_e].values

    thresh = volt_center.mean() - (volt_center.std() * std)
    cycle_idxs = np.array([[idx-1, idx, idx+1] for idx in np.where(volt_center < thresh)[0]
                            if idx-1 >= 0 and idx+1 < len(bm.df_features)])

    # Get cyclespoints from bycycle dataframe
    starts = bm.df_features.iloc[cycle_idxs[:, 0]]['sample_zerox_rise'].values
    centers =  bm.df_features.iloc[cycle_idxs[:, 1]]['sample_trough'].values
    ends = bm.df_features.iloc[cycle_idxs[:, 2]]['sample_trough'].values
    next_sides = bm.df_features.iloc[cycle_idxs[:, 1]]['sample_next_peak'].values

    # Maximum pre-extrema and post-extrema lengths
    left_max = (centers-starts).max()
    right_max = (ends-centers).max()

    # Init 2d spike array
    spikes = np.zeros((len(starts), sum([left_max, right_max])))
    spikes[:] = np.nan
    # Store cyclepoints as 2d array of (defined for trough-centered spikes):
    #   [first rise, trough, peak, inflection, first decay, rise, second decay]
    cps = zip(starts, centers,  next_sides, ends)
    cps_adjusted = np.zeros((len(centers), 7), dtype=int)

    # Keep track of how much the start of a spike changes
    start_shifts = np.zeros_like(starts)

    # Store where final spikes are subthresh
    drop_idxs = np.zeros(len(starts), dtype='bool')

    for idx, (start, center, side, end) in enumerate(cps):

        # Fill the spike from the signal
        pad_left = left_max - (center-start)
        pad_right = len(spikes[idx]) - (right_max - (end-center))
        spikes[idx][pad_left:pad_right] = bm.sig[start:end]

        # Determine the inflection point
        #   (first derivative == 0) of the repolarization phase
        extrema_right = pad_right - (end-side)
        trail_flank = np.diff(spikes[idx][extrema_right:pad_right])
        trail_flank = np.where(np.greater_equal(trail_flank, 0))[0]

        inflection_pt = pad_right-extrema_right-1 if len(trail_flank) == 0 else trail_flank[0]
        inflection_pt += extrema_right

        # Adjust the next side point to match the location in the 2d spike array
        adj_next_side = pad_left + (side - start)
        adj_center = pad_left + (center - start)

        # Determine the first rising point
        spike_inv = np.flip(spikes[idx][:adj_center+1])
        diff_sign = np.sign(np.diff(spike_inv))

        new_start = np.where((diff_sign[:-1] == -1) & (diff_sign[1:] == 1))[0]

        # Minus one to account for diff slicing
        adj_start = pad_left if len(new_start) == 0 else adj_center - new_start[0] - 1
        start_shifts[idx] = adj_start - pad_left

        # Everything to the right/left of the inflection/start gets nan
        spikes[idx][inflection_pt+1:] = np.nan
        spikes[idx][:adj_start] = np.nan

        # Zeros are place holders for zerox
        cps_adjusted[idx] = [adj_start, adj_center, adj_next_side, inflection_pt, 0, 0, 0]

        # Mark poor spikes (i.e. the center point should be the only voltage outlier)
        spike_mean = np.nanmean(spikes[idx])
        spike_std = np.nanstd(spikes[idx])

        spike_thresh = spike_mean - (std * spike_std)

        # Check non-center point voltages are closer to one another than the trough
        trough_volt = spikes[idx][adj_center]

        cyc_pts = [adj_start, adj_next_side, inflection_pt]

        for cyc_idx, sample in enumerate(cyc_pts):

            curr_volt = spikes[idx][sample]

            others = [0, 1, 2]
            del others[cyc_idx]

            for other in others:

                other_volt = spikes[idx][cyc_pts[other]]

                if np.abs(curr_volt - trough_volt) < np.abs(curr_volt - other_volt):
                    drop_idxs[idx] = True
                    break

            if drop_idxs[idx]:
                break

        # Check center point is outside std
        if  trough_volt >= spike_thresh:
            drop_idxs[idx] = True

    # Remove std violations
    spikes = spikes[~drop_idxs]
    cps_adjusted = cps_adjusted[~drop_idxs]

    mask = ~np.isnan(spikes).all(axis=0)
    spikes = spikes[:, mask]

    # Shift to account for spike removal
    mask_shift = len(mask[:np.where(mask)[0][0]])
    cps_adjusted = cps_adjusted - mask_shift

    # Get zero-crossings
    for idx, (spike, cps) in enumerate(zip(spikes, cps_adjusted)):

        # Order of params is flipped since signal is inverted
        decays, rises = find_zerox(-spike, [cps[1], cps[3]], [cps[0], cps[2]])
        cps[-3:] = [decays[0], rises[0], decays[1]]

    # Move cps to a dataframe
    df_features = pd.DataFrame()

    df_features['sample_spike'] = starts[~drop_idxs] + start_shifts[~drop_idxs]
    df_features['sample_last_rise'] = cps_adjusted[:, 0]
    df_features['sample_trough'] = cps_adjusted[:, 1]
    df_features['sample_next_peak'] = cps_adjusted[:, 2]
    df_features['sample_next_decay'] = cps_adjusted[:, 3]
    df_features['sample_zerox_decay'] = cps_adjusted[:, 4]
    df_features['sample_zerox_rise'] = cps_adjusted[:, 5]
    df_features['sample_next_zerox_decay'] = cps_adjusted[:, 6]

    return df_features, spikes


def rename_df(df_features):
    """Rename the columns of a peak-centered dataframe.

    Parameters
    ----------
    df_features : pd.DataFrame
        A dataframe containing cyclepoint locations, as samples indices, for each spike.

    Returns
    -------
    df_features : pd.DataFrame
        A renamed dataframe containing updated column names.
    """

    mapping = {}

    orig_keys = ['peak', 'trough', 'rise', 'decay']
    new_keys = ['trough', 'peak', 'decay', 'rise']

    for key in df_features.columns:
        for orig, new in zip(orig_keys, new_keys):
            if orig in key:
                mapping[key] = key.replace(orig, new)

    df_features.rename(columns=mapping, inplace=True)

    return df_features

