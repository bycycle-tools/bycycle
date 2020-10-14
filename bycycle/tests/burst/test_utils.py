"""Test burst.utils."""

import numpy as np

from bycycle.features import compute_features
from bycycle.burst.utils import *


###################################################################################################
###################################################################################################

def test_check_min_burst_cycles():

    is_burst = np.array([True, True, True, False])
    is_burst_check = check_min_burst_cycles(is_burst, min_n_cycles=3)

    assert (is_burst == is_burst_check).all()

    is_burst = np.array([True, False, True, False])
    is_burst_check = check_min_burst_cycles(is_burst, min_n_cycles=3)

    assert not any(is_burst_check)


def test_recompute_edges(sim_args_comb):

    # Grab sim arguments from fixture
    sig = sim_args_comb['sig']
    threshold_kwargs = sim_args_comb['threshold_kwargs']
    fs = sim_args_comb['fs']
    n_seconds = len(sig) / fs
    f_range = sim_args_comb['f_range']
    df_features = sim_args_comb['df_features']

    # Case 1: use the same thresholds should result in the same dataframe
    df_features_edges = recompute_edges(df_features, threshold_kwargs)
    assert (df_features['is_burst'].values == df_features['is_burst'].values).all()

    # Case 2: ensure the number of burst increases when edge thresholds are lowered

    # This guarantees at least one non-burst cycle on the edge will be recomputed
    #   as a burst by duplicating the signal and adding zero-padding between
    sig_force_recomp = np.zeros(2*len(sig) + fs)
    sig_force_recomp[:int(fs *  n_seconds)] = sig
    sig_force_recomp[int(fs * n_seconds) + fs:] = sig

    df_features = compute_features(sig, fs, f_range, threshold_kwargs=threshold_kwargs,
                                   return_samples=False)

    # Update thresholds to give all edge cycles is_burst = True, except for the first and
    #   last cycles, since these cycles contain np.nan values for consistency measures
    threshold_kwargs['amp_fraction_threshold'] = 0
    threshold_kwargs['amp_consistency_threshold'] = 0
    threshold_kwargs['period_consistency_threshold'] = 0
    threshold_kwargs['monotonicity_threshold'] = 0

    df_features_edges = recompute_edges(df_features, threshold_kwargs)

    # Ensure that at least one cycle was added to is_burst after recomputed
    is_burst_orig = len(np.where(df_features['is_burst'] == True)[0])
    is_burst_recompute = len(np.where(df_features_edges['is_burst'] == True)[0])

    assert is_burst_recompute > is_burst_orig
