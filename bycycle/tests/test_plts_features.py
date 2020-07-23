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
    threshold_kwargs = sim_args['threshold_kwargs']

    # Apply consistency burst detection
    df_features = detect_bursts_cycles(df_features, **threshold_kwargs)

    plot_feature_hist(df_features, 'amp_consistency', xlim=(0, 1), save_fig=True,
                      file_name='test_plot_feature_hist', file_path=TEST_PLOTS_PATH)

@plot_test
def test_plot_feature_categorical(sim_args):

    df_features = sim_args['df_features']
    threshold_kwargs = sim_args['threshold_kwargs']

    # Apply consistency burst detection
    df_features = detect_bursts_cycles(df_features, **threshold_kwargs)

    # Compare the first to second half of a signal.
    #   The distribution should be the same since sim_oscillation is used.
    group = np.array(['First Half' for row in range(len(df_features))])
    group[:round(len(group)/2)] = 'Second Half'
    df_features['group'] = group

    plot_feature_categorical(df_features, 'amp_consistency', group_by='group', save_fig=True,
                             file_name='test_plot_feature_hist', file_path=TEST_PLOTS_PATH)
