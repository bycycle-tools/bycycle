"""Tests plotting bursts."""

import numpy as np
import pytest

from bycycle.plts import plot_burst_detect_param, plot_burst_detect_summary
from bycycle.tests.utils import plot_test

###################################################################################################
###################################################################################################

@plot_test
@pytest.mark.parametrize("interp", [True, False])
def test_plot_burst_detect_param(sim_args, interp):

    df = sim_args['df']
    sig = sim_args['sig']
    fs = sim_args['fs']

    thresh = np.nanmin(df['amp_consistency'].values) + 0.1

    plot_burst_detect_param(df, sig, fs, 'amp_consistency', thresh, interp=interp)


@plot_test
@pytest.mark.parametrize("plot_only_result", [True, False])
def test_plot_burst_detect_summary(sim_args, plot_only_result):

    osc_kwargs = {'amplitude_fraction_threshold': 1.1,
                  'amplitude_consistency_threshold': .5,
                  'period_consistency_threshold': .5,
                  'monotonicity_threshold': .8}

    plot_burst_detect_summary(**sim_args, osc_kwargs=osc_kwargs, plot_only_result=plot_only_result)
