"""Tests for plts.burst."""

import pytest

import numpy as np

from bycycle.burst import detect_bursts_cycles

from bycycle.tests.tutils import plot_test
from bycycle.tests.settings import TEST_PLOTS_PATH

from bycycle.plts import *

###################################################################################################
###################################################################################################

@plot_test
@pytest.mark.parametrize("interp", [True, False])
def test_plot_burst_detect_param(sim_args, interp):

    df_samples = sim_args['df_samples']
    df_features = sim_args['df_features']
    sig = sim_args['sig']
    fs = sim_args['fs']

    thresh = np.nanmin(df_features['amp_consistency'].values) + 0.1

    plot_burst_detect_param(df_features, df_samples, sig, fs, 'amp_consistency',
                            thresh, interp=interp, save_fig=True,
                            file_path=TEST_PLOTS_PATH, file_name='test_plot_burst_detect_param')


@plot_test
@pytest.mark.parametrize("plot_only_result", [True, False])
def test_plot_burst_detect_summary(sim_args, plot_only_result):

    burst_detection_kwargs = {'amp_fraction_threshold': 1,
                              'amp_consistency_threshold': .5,
                              'period_consistency_threshold': .5,
                              'monotonicity_threshold': .8,
                              'min_n_cycles': 3}

    df_samples = sim_args['df_samples']
    df_features = sim_args['df_features']
    sig = sim_args['sig']
    fs = sim_args['fs']

    df_features = detect_bursts_cycles(df_features, **burst_detection_kwargs)

    plot_burst_detect_summary(df_features, df_samples, sig, fs, burst_detection_kwargs,
                              plot_only_result=plot_only_result,
                              save_fig=True, file_path=TEST_PLOTS_PATH,
                              file_name='test_plot_burst_detect_summary')
