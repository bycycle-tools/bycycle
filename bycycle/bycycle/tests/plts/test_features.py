"""Tests plts.features."""

import numpy as np

from bycycle.burst import detect_bursts_cycles

from bycycle.tests.tutils import plot_test
from bycycle.tests.settings import TEST_PLOTS_PATH

from bycycle.plts.features import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_feature_hist(sim_args):

    df_features = sim_args['df_features']
    threshold_kwargs = sim_args['threshold_kwargs']

    df_features = detect_bursts_cycles(df_features, **threshold_kwargs)

    plot_feature_hist(df_features, 'amp_consistency', xlim=(0, 1), save_fig=True,
                      file_name='test_plot_feature_hist', file_path=TEST_PLOTS_PATH)

@plot_test
def test_plot_feature_categorical(sim_args):

    df_features = sim_args['df_features']
    threshold_kwargs = sim_args['threshold_kwargs']

    df_features = detect_bursts_cycles(df_features, **threshold_kwargs)

    # Compare first & second halves of the signal - distributions should be the same
    group = np.array(['First Half' for row in range(len(df_features))])
    group[:round(len(group)/2)] = 'Second Half'
    df_features['group'] = group

    plot_feature_categorical(df_features, 'amp_consistency', group_by='group', save_fig=True,
                             file_name='test_plot_feature_hist', file_path=TEST_PLOTS_PATH)
