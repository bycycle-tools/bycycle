"""Test functions for the Bycycle class."""

from pytest import raises

import pandas as pd

from bycycle import Bycycle
from bycycle.tests.tutils import plot_test

###################################################################################################
###################################################################################################

def test_bycycle():
    """Test initializing a Bycycle object."""

    bm = Bycycle()

    assert bm.center_extrema == 'peak'
    assert bm.burst_method == 'cycles'
    assert isinstance(bm.thresholds, dict)
    assert bm.return_samples

    defaults = [bm.burst_kwargs, bm.find_extrema_kwargs, bm.df_features, bm.sig, bm.fs, bm.f_range]
    assert defaults == [None] * len(defaults)


def test_bycycle_fit(sim_args):
    """Test the fit method of a Bycycle object."""

    bm = Bycycle()

    sig = sim_args['sig']
    fs = sim_args['fs']
    f_range = sim_args['f_range']

    bm.fit(sig, fs, f_range)

    assert isinstance(bm.df_features, pd.DataFrame)
    assert bm.fs == fs
    assert bm.f_range == f_range
    assert (bm.sig == sig).all()


@plot_test
def test_bycycle_plot(sim_args):
    """Test the plot method of a Bycycle object."""

    bm = Bycycle()

    sig = sim_args['sig']
    fs = sim_args['fs']
    f_range = sim_args['f_range']

    with raises(ValueError):
        bm.plot()

    bm.fit(sig, fs, f_range)
    bm.plot()
