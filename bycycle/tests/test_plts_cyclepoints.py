"""Tests plotting cyclepoints."""

from bycycle.plts.cyclepoints import plot_cyclepoints_df, plot_cyclepoints_array
from bycycle.cyclepoints import find_extrema, find_zerox
from bycycle.tests.utils import plot_test
from bycycle.tests.settings import TEST_PLOTS_PATH

###################################################################################################
###################################################################################################

@plot_test
def test_plot_cyclepoints_df(sim_args):

    plot_cyclepoints_df(sim_args['df_samples'], sim_args['sig'], sim_args['fs'], save_fig=True,
                        file_name='test_plot_cyclepoints_df', file_path=TEST_PLOTS_PATH)


@plot_test
def test_plot_cyclepoints_array(sim_args):

    ps, ts = find_extrema(sim_args['sig'], sim_args['fs'], (6, 14))
    rises, decays = find_zerox(sim_args['sig'], ps, ts)

    plot_cyclepoints_array(sim_args['sig'], sim_args['fs'], ps=ps, ts=ts,
                           rises=rises, decays=decays, save_fig=True,
                           file_name='test_plot_cyclepoints_array', file_path=TEST_PLOTS_PATH)
