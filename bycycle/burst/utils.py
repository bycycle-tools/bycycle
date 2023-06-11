"""Utilities for burst detection."""

from copy import deepcopy
import numpy as np
from bycycle.utils.checks import check_param_range, check_param_options

####################################################################################################
####################################################################################################

def check_min_burst_cycles(is_burst, min_n_cycles=3):
    """Enforce minimum number of consecutive cycles to be considered a burst.

    Parameters
    ----------
    is_burst : 1d array-like
        Boolean array indicating which cycles are bursting.
    min_n_cycles : int, optional, default: 3
        The minimum number of cycles of consecutive cycles required to be considered a burst.

    Returns
    -------
    is_burst : 1d array-like
        Updated burst array with same type as input.

    Examples
    --------
    Remove bursts with less than 3 consecutive cycles:

    >>> is_burst = np.array([False, True, True, False, True, True, True, True, False])
    >>> check_min_burst_cycles(is_burst)
    array([False, False, False, False,  True,  True,  True,  True, False])
    """

    # Ensure argument is within valid range
    check_param_range(min_n_cycles, 'min_n_cycles', (0, np.inf))

    # extract transition indices
    diff = np.diff(is_burst, prepend=0, append=0)
    transitions = np.flatnonzero(diff)
    ons, offs = transitions[0::2], transitions[1::2]

    # select only segments with long enough duration
    durations = offs - ons
    long_enough = durations >= min_n_cycles
    ons, offs = ons[long_enough], offs[long_enough]

    # construct bool time series from transition indices
    is_burst[:] = False
    for turn_on, turn_off in zip(ons, offs):
        is_burst[turn_on:turn_off] = True

    return is_burst


def recompute_edges(df_features, threshold_kwargs, burst_method='cycles', burst_kwargs=None):
    """Recompute the is_burst column for cycles on the edges of bursts.

    Parameters
    ----------
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for each cycle.
    threshold_kwargs : dict
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

    >>> from neurodsp.sim import sim_combined
    >>> from bycycle.features import compute_features
    >>> sig = sim_combined(n_seconds=4, fs=1000, components={'sim_bursty_oscillation': {'freq': 10},
    ...                                                      'sim_powerlaw': {'exp': 2}})
    >>> threshold_kwargs = {'amp_fraction_threshold': 0., 'amp_consistency_threshold': .5,
    ...                     'period_consistency_threshold': .5, 'monotonicity_threshold': .4,
    ...                     'min_n_cycles': 3}
    >>> df_features = compute_features(sig, fs=1000, f_range=(8, 12),
    ...                                threshold_kwargs=threshold_kwargs)
    >>> threshold_kwargs['amp_consistency_threshold'] = 0
    >>> df_features_edges = recompute_edges(df_features, threshold_kwargs)
    """

    # Prevent circular imports between burst.utils and burst.cycle
    from bycycle.burst import detect_bursts_cycles

    # Prevent overwriting the original dataframe
    df_features_edges = df_features.copy()

    # Identify all cycles where is_burst changes on the following cycle
    #   Use copy to keep dataframe columns unlinked
    is_burst = deepcopy(df_features_edges['is_burst'].values)
    burst_edges = np.where(is_burst[1:] == ~is_burst[:-1])[0]

    # Get cycles outside of bursts
    burst_starts = np.array([edge for idx, edge in enumerate(burst_edges) if idx % 2 == 0])
    burst_ends = np.array([edge+1 for idx, edge in enumerate(burst_edges) if idx % 2 == 1 ])

    # Recompute is_burst for cycles at the edge
    for start_idx, end_idx in zip(burst_starts, burst_ends):

        df_features_edges = recompute_edge(df_features_edges, start_idx, 'next')
        df_features_edges = recompute_edge(df_features_edges, end_idx, 'last')

    # Apply thresholding
    df_features_edges = detect_bursts_cycles(df_features_edges, **threshold_kwargs)

    return df_features_edges


def recompute_edge(df_features, cyc_idx, direction):
    """Recompute consistency features at the edge of a cycle.

    Parameters
    ----------
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for each cycle.
    cyc_idx : int
        The dataframe index of the edge.
    direction : {'both', 'next', 'last'}
        The direction to compute consistency.

    Returns
    -------
    df_features : pandas.DataFrame
        A dataframe with updated consistency features.
    """

    # Check that param validity
    check_param_options(direction, 'direction', ['both', 'next', 'last'])

    # Prevent circular imports between burst.utils and burst.cycle
    from bycycle.features.burst import compute_amp_consistency, compute_period_consistency

    # Slice edges rows and recompute burst features
    lower = max(cyc_idx-1, 0)
    upper = min(cyc_idx+2, len(df_features))
    edge_range = range(lower, upper)

    edge = df_features.iloc[edge_range].copy()

    # Update dataframe with recomputed consistency features
    df_features['amp_consistency'][cyc_idx] = \
        compute_amp_consistency(edge, direction=direction)[1]

    df_features['period_consistency'][cyc_idx] = \
        compute_period_consistency(edge, direction=direction)[1]

    return df_features
