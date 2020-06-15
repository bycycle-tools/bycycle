"""Configuration file for pytest for bycycle."""

import pytest

import numpy as np

from neurodsp.utils.sim import set_random_seed
from neurodsp.sim import sim_oscillation
from bycycle.features import compute_features

###################################################################################################
###################################################################################################

def pytest_configure(config):

    set_random_seed(42)


@pytest.fixture(scope='module')
def sim_args():

    # Simulate oscillating time series
    n_seconds = 10
    fs = 500
    freq = 10
    f_range = (6, 14)

    sig = sim_oscillation(n_seconds, fs, freq)

    df = compute_features(sig, fs, f_range)


    yield {'df': df, 'sig': sig, 'fs': fs}
