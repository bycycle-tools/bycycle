"""Test functions for the Bycycle class."""

from inspect import ismethod

import pytest

import pandas as pd
import numpy as np

from bycycle.objs.fit import Bycycle, BycycleGroup, BycycleBase
from bycycle.tests.tutils import plot_test

###################################################################################################
###################################################################################################

@pytest.mark.parametrize('Base', [BycycleBase, Bycycle, BycycleGroup])
@pytest.mark.parametrize('thresholds', [True, False])
@pytest.mark.parametrize('burst_method', ['cycles', 'amp'])
@pytest.mark.parametrize('find_extrema_kwargs', [True, False])
def test_base_init(Base, thresholds, burst_method, find_extrema_kwargs):
    """Test initializing a Bycycle object."""

    if not thresholds:
        thresholds = None
    elif burst_method == 'cycles':
        thresholds = {
            'amp_fraction': 0.,
            'amp_consistency': .5,
            'period_consistency': .5,
            'monotonicity': .8,
            'min_n_cycles': 3
        }
    elif burst_method == 'amp':
        thresholds = {
            'burst_fraction': 1,
            'min_n_cycles': 3
        }

    if find_extrema_kwargs:
        find_extrema_kwargs = {'filter_kwargs': {'n_cycles': 3}}
    else:
        find_extrema_kwargs = None

    bm = Base(thresholds=thresholds, burst_method=burst_method,
              find_extrema_kwargs=find_extrema_kwargs)

    assert bm.center_extrema == 'peak'
    assert bm.burst_method == burst_method
    assert isinstance(bm.thresholds, dict)
    assert isinstance(bm.find_extrema_kwargs, dict)
    assert bm.burst_kwargs == {}
    assert bm.return_samples

    defaults = [bm.df_features, bm.sig, bm.fs, bm.f_range]
    assert defaults == [None] * len(defaults)


@pytest.mark.parametrize('recompute_edges', [True, False])
def test_bycycle_fit(sim_args, recompute_edges):
    """Test the fit method of a Bycycle object."""

    bm = Bycycle()

    sig = sim_args['sig']
    fs = sim_args['fs']
    f_range = sim_args['f_range']

    with pytest.raises(ValueError):
        bm.fit(np.array([sig, sig]), fs, f_range)

    bm.fit(sig, fs, f_range, recompute_edges=recompute_edges)

    assert isinstance(bm.df_features, pd.DataFrame)
    assert bm.fs == fs
    assert bm.f_range == f_range
    assert (bm.sig == sig).all()

    # test getting attribute from dataframe
    assert (bm.df_features['time_peak'].values == bm.time_peak).all()


@plot_test
def test_bycycle_plot(sim_args):
    """Test the plot method of a Bycycle object."""

    bm = Bycycle()

    sig = sim_args['sig']
    fs = sim_args['fs']
    f_range = sim_args['f_range']

    with pytest.raises(ValueError):
        bm.plot()

    with pytest.raises(AttributeError):
        bm.df_features.time_peak

    bm.fit(sig, fs, f_range)
    bm.plot()
    bm.df_features.time_peak


def test_bycyclegroup_fit(sim_args):
    """Test the fit method of a BycycleGroup object."""

    # 2d case
    sigs = np.array([sim_args['sig']] * 2)
    fs = sim_args['fs']
    f_range = sim_args['f_range']
    thresholds = {
        'amp_fraction_threshold': 0.,
        'amp_consistency_threshold': .5,
        'period_consistency_threshold': .5,
        'monotonicity_threshold': .8,
        'min_n_cycles': 3
    }

    bg = BycycleGroup(thresholds=thresholds)

    with pytest.raises(ValueError):
        bg.fit(sigs[0], fs, f_range)

    bg.fit(sigs, fs, f_range)

    assert isinstance(bg.df_features, list)
    for bm in bg:
        assert isinstance(bm, Bycycle)
    assert isinstance(bg.df_features[0], pd.DataFrame)
    assert ismethod(bg.models[0].fit)
    assert ismethod(bg.models[0].plot)
    assert len(bg) == 2
    assert bg.fs == fs
    assert bg.f_range == f_range
    assert (bg.sigs == sigs).all()

    # 3d case
    sigs = np.array([sigs, sigs])
    bg = BycycleGroup()

    with pytest.raises(ValueError):
        bg.fit(sigs[0][0], fs, f_range)

    bg.fit(sigs, fs, f_range)

    assert isinstance(bg.df_features, list)
    assert isinstance(bg.df_features[0], list)
    assert isinstance(bg[0][0], Bycycle)
    assert isinstance(bg.df_features[0][0], pd.DataFrame)
    assert ismethod(bg.models[0][0].fit)
    assert ismethod(bg.models[0][0].plot)
    assert ismethod(bg[0][0].fit)
    assert ismethod(bg[0][0].plot)
    assert len(bg) == 2
    assert len(bg[0]) == 2
    assert bg.fs == fs
    assert bg.f_range == f_range
    assert (bg.sigs == sigs).all()