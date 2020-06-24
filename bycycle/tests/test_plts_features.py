"""Tests plotting cycle features."""

import numpy as np

from bycycle.plts.features import  plot_feature_hist, plot_feature_catplot
from bycycle.tests.utils import plot_test
from bycycle.tests.settings import TEST_PLOTS_PATH

###################################################################################################
###################################################################################################

@plot_test
def test_plot_feature_hist(sim_args):

    df = sim_args['df']

    plot_feature_hist(df, 'volt_amp', xlim=(0, 1), save_fig=True,
                      file_name='test_plot_feature_hist', file_path=TEST_PLOTS_PATH)

@plot_test
def test_plot_feature_catplot(sim_args):

    df = sim_args['df']

    # Compare the first to second half of a signal.
    #   The distribution should be the same since sim_oscillation is used.
    group = np.array(['First Half' for row in range(len(df))])
    group[:round(len(group)/2)] = 'Second Half'
    df['group'] = group

    plot_feature_catplot(df, 'amp_consistency', group_by='group', save_fig=True,
                         file_name='test_plot_feature_hist', file_path=TEST_PLOTS_PATH)
