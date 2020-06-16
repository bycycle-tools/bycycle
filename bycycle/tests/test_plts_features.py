"""Tests plotting cycle features."""

import pytest

from bycycle.plts.features import  plot_feature_hist
from bycycle.tests.utils import plot_test

###################################################################################################
###################################################################################################

@plot_test
def test_plot_feature_hist(sim_args):

    df = sim_args['df']

    plot_feature_hist(df, 'volt_amp', xlim=(0, 1))
