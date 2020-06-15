"""Tests plotting cyclepoints."""

import pytest

from bycycle.plts.cyclepoints import plot_cyclepoints
from bycycle.tests.utils import plot_test

###################################################################################################
###################################################################################################

@plot_test
@pytest.mark.parametrize("tlims", [None, (0, 1)])
def test_plot_cyclepoints(tlims, sim_args):
    """Test plotting extrema/zero-crossings."""

    plot_cyclepoints(sim_args['df'], sim_args['sig'], sim_args['fs'], tlims=tlims)
