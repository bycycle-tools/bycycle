"""Tests plotting cycle features."""

import numpy as np

from bycycle.burst import detect_bursts_cycles
from bycycle.plts.features import  plot_feature_hist, plot_feature_categorical
from bycycle.tests.utils import plot_test
from bycycle.tests.settings import TEST_PLOTS_PATH

###################################################################################################
###################################################################################################

@plot_test
def test_plot_feature_hist(sim_args):

    df_features = sim_args['df_features']

     # Apply consistency burst detection for consistency detection
    burst_detection_kwargs = {'amplitude_fraction_threshold': 0.,
                              'amplitude_consistency_threshold': .5,
                              'period_consistency_threshold': .5,
                              'monotonicity_threshold': .5,
                              'n_cycles_min': 3}

    df_features = detect_bursts_cycles(df_features, **burst_detection_kwargs)

    plot_feature_hist(df_features, 'amplitude_consistency', xlim=(0, 1), save_fig=True,
                      file_name='test_plot_feature_hist', file_path=TEST_PLOTS_PATH)

@plot_test
def test_plot_feature_categorical(sim_args):

    df_features = sim_args['df_features']

    # Apply consistency burst detection for consistency detection
    burst_detection_kwargs = {'amplitude_fraction_threshold': 0.,
                              'amplitude_consistency_threshold': .5,
                              'period_consistency_threshold': .5,
                              'monotonicity_threshold': .5,
                              'n_cycles_min': 3}

    df_features = detect_bursts_cycles(df_features, **burst_detection_kwargs)

    # Compare the first to second half of a signal.
    #   The distribution should be the same since sim_oscillation is used.
    group = np.array(['First Half' for row in range(len(df_features))])
    group[:round(len(group)/2)] = 'Second Half'
    df_features['group'] = group

    plot_feature_categorical(df_features, 'amplitude_consistency', group_by='group', save_fig=True,
                            file_name='test_plot_feature_hist', file_path=TEST_PLOTS_PATH)
