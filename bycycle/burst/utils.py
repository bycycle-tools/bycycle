"""Utilities for burst detection."""

from copy import deepcopy
import numpy as np
from bycycle.utils.checks import check_param

###################################################################################################
###################################################################################################

def check_min_burst_cycles(is_burst, min_n_cycles=3):
    """Enforce minimum number of consecutive cycles to be considered a burst.

    Parameters
    ----------
    is_burst : 1d array
        Boolean array indicating which cycles are bursting.
    min_n_cycles : int, optional, default: 3
        The minimum number of cycles of consecutive cycles required to be considered a burst.

    Returns
    -------
    is_burst : 1d array
        Updated burst array.

    Examples
    --------
    Remove bursts with less than 3 consectutive cycles:

    >>> is_burst = np.array([False, True, True, False, True, True, True, True, False])
    >>> check_min_burst_cycles(is_burst)
    array([False, False, False, False,  True,  True,  True,  True, False])
    """

    # Ensure argument is within valid range
    check_param(min_n_cycles, 'min_n_cycles', (0, np.inf))

    temp_cycle_count = 0

    for idx, bursting in enumerate(is_burst):

        if bursting:
            temp_cycle_count += 1

        else:

            if temp_cycle_count < min_n_cycles:
                for c_rm in range(temp_cycle_count):
                    is_burst[idx - 1 - c_rm] = False

            temp_cycle_count = 0

    return is_burst


def recompute_edges(df_features, threshold_kwargs):
    """Recompute the is_burst column for cycles on the edges of bursts.

    Parameters
    ----------
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for each cycle.
    threshold_kwargs : dict, optional, default: None
        Feature thresholds for cycles to be considered bursts, matching keyword arguments for:

        - :func:`~.detect_bursts_cycles` for consistency burst detection
          (i.e. when burst_method == 'cycles')

    Returns
    -------
    df_features_edges : pandas.DataFrame
        An cycle feature dataframe with an updated ``is_burst`` column for edge cycles.

    Notes
    -----

    - `df_features` must be computed using consistency burst detection.

    Examples
    --------
    Lower the amplitude consistency threshold to zero for cycles on the edges of bursts:

    >>> sig = sim_combined(n_seconds=4, fs=1000, components={'sim_bursty_oscillation': {'freq': 10},
    ...                                                      'sim_powerlaw': {'exp': 2}})
    >>> threshold_kwargs = {'amp_fraction_threshold': 0., 'amp_consistency_threshold': .5,
    ...                     'period_consistency_threshold': .5, 'monotonicity_threshold': .4,
    ...                     'min_n_cycles': 3}
    >>> df_features, _ = compute_features(sig, fs=1000, f_range=(8, 12), threshold_kwargs=threshold_kwargs)
    >>> threshold_kwargs['amp_consistency_threshold'] = 0
    >>> df_features_edges = recompute_edges(df_features, threshold_kwargs)
    """

    # Prevent circular import between burst.utils and burst.cycle
    from bycycle.burst import detect_bursts_cycles

    # Prevent overwriting the orignal dataframe
    df_features_edges = df_features.copy()

    # Identify all cycles where is_burst changes on the following cycle
    #   Use copy to keep dataframe columns unlinked
    is_burst = deepcopy(df_features_edges['is_burst'].values)
    burst_edges = np.where(is_burst[1:] == ~is_burst[:-1])[0]

    # Adjust odd edges such that all edges fall on is_burst == False
    burst_edges = np.array([edge if idx % 2 == 0 else edge+1 for idx, edge in
                            enumerate(burst_edges)])

    # Recompute is_burst
    df_features_edges =  detect_bursts_cycles(
        df_features_edges,
        amp_fraction_threshold=threshold_kwargs['amp_fraction_threshold'],
        amp_consistency_threshold=threshold_kwargs['amp_consistency_threshold'],
        period_consistency_threshold=threshold_kwargs['period_consistency_threshold'],
        monotonicity_threshold=threshold_kwargs['monotonicity_threshold'],
        min_n_cycles=threshold_kwargs['min_n_cycles']
    )

    # Confine recomputed is_burst to edges
    is_burst[burst_edges] = df_features_edges.iloc[burst_edges]['is_burst'].values

    df_features_edges['is_burst'] = is_burst

    return df_features_edges
