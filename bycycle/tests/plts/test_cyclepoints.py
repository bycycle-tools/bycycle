"""Tests for plts.cyclepoints."""

from bycycle.cyclepoints import find_extrema, find_zerox

from bycycle.tests.tutils import plot_test
from bycycle.tests.settings import TEST_PLOTS_PATH

from bycycle.plts.cyclepoints import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_cyclepoints_df(sim_args):

    plot_cyclepoints_df(sim_args['df_samples'], sim_args['sig'], sim_args['fs'], save_fig=True,
                        file_name='test_plot_cyclepoints_df', file_path=TEST_PLOTS_PATH)


@plot_test
def test_plot_cyclepoints_array(sim_args):

    peaks, troughs = find_extrema(sim_args['sig'], sim_args['fs'], (6, 14))
    rises, decays = find_zerox(sim_args['sig'], peaks, troughs)

    plot_cyclepoints_array(sim_args['sig'], sim_args['fs'], peaks=peaks, troughs=troughs,
                           rises=rises, decays=decays, save_fig=True,
                           file_name='test_plot_cyclepoints_array', file_path=TEST_PLOTS_PATH)
