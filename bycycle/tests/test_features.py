"""Tests the main cycle-by-cycle feature computation function."""

import os
import numpy as np

from bycycle.features import compute_shapes, compute_features

# Set data path
DATA_PATH = os.getcwd() + '/tutorials/data/'

###################################################################################################
###################################################################################################

def test_compute_shapes():

     # Load signal
    sig = np.load(DATA_PATH + 'sim_stationary.npy')

    fs = 1000
    f_range = (6, 14)

    # Compute cycle shapes
    df_shapes = compute_shapes(sig, fs, f_range)

    # Check inverted signal gives appropriately opposite data
    df_opp = compute_shapes(-sig, fs, f_range, center_extrema='trough')

    np.testing.assert_allclose(df_shapes['sample_peak'], df_opp['sample_trough'])
    np.testing.assert_allclose(df_shapes['sample_last_trough'], df_opp['sample_last_peak'])
    np.testing.assert_allclose(df_shapes['time_peak'], df_opp['time_trough'])
    np.testing.assert_allclose(df_shapes['time_rise'], df_opp['time_decay'])
    np.testing.assert_allclose(df_shapes['volt_rise'], df_opp['volt_decay'])
    np.testing.assert_allclose(df_shapes['volt_amp'], df_opp['volt_amp'])
    np.testing.assert_allclose(df_shapes['period'], df_opp['period'])
    np.testing.assert_allclose(df_shapes['time_rdsym'], 1 - df_opp['time_rdsym'])
    np.testing.assert_allclose(df_shapes['time_ptsym'], 1 - df_opp['time_ptsym'])


def test_compute_features():
    """Test cycle-by-cycle feature computation."""

    # Load signal
    sig = np.load(DATA_PATH + 'sim_stationary.npy')

    fs = 1000
    f_range = (6, 14)

    # Compute cycle shapes
    df_shapes = compute_shapes(sig, fs, f_range)

    # Compute cycle features
    df_features = compute_features(df_shapes, sig)

    assert len(df_shapes) == len(df_features)
    assert np.nan not in df_features['amplitude_fraction']
    assert np.nan not in df_features['amplitude_consistency']
    assert np.nan not in df_features['period_consistency']
    assert np.nan not in df_features['monotonicity']



