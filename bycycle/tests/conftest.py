"""Configuration file for pytest for bycycle."""

import os
import shutil

import pytest

import numpy as np

from neurodsp.sim import sim_oscillation, sim_combined
from neurodsp.utils.sim import set_random_seed

from bycycle.features import compute_shape_features, compute_burst_features, compute_features
from bycycle.tests.settings import (N_SECONDS, FS, FREQ, F_RANGE,
                                    BASE_TEST_FILE_PATH, TEST_PLOTS_PATH)

###################################################################################################
###################################################################################################

def pytest_configure(config):

    set_random_seed(42)


@pytest.fixture(scope='module')
def sim_args():

    sig = sim_oscillation(N_SECONDS, FS, FREQ)

    df_shapes = compute_shape_features(sig, FS, F_RANGE)
    df_burst = compute_burst_features(df_shapes, sig)
    df_features = compute_features(sig, FS, F_RANGE)

    threshold_kwargs = {'amp_fraction_threshold': 0.,
                        'amp_consistency_threshold': .5,
                        'period_consistency_threshold': .5,
                        'monotonicity_threshold': .5,
                        'min_n_cycles': 3}

    yield {'sig': sig, 'fs': FS, 'f_range': F_RANGE, 'df_features': df_features,
           'df_shapes': df_shapes, 'df_burst': df_burst, 'threshold_kwargs': threshold_kwargs}


@pytest.fixture(scope='module')
def sim_args_comb():

    components = {'sim_bursty_oscillation': {'freq': FREQ}, 'sim_powerlaw': {'exp': 2}}

    sig = sim_combined(N_SECONDS, FS, components=components)

    df_shapes = compute_shape_features(sig, FS, F_RANGE)
    df_burst = compute_burst_features(df_shapes, sig)
    df_features = compute_features(sig, FS, F_RANGE)

    threshold_kwargs = {'amp_fraction_threshold': 0.,
                        'amp_consistency_threshold': .5,
                        'period_consistency_threshold': .5,
                        'monotonicity_threshold': .5,
                        'min_n_cycles': 3}

    yield {'sig': sig, 'fs': FS, 'f_range': F_RANGE, 'df_features': df_features,
           'df_shapes': df_shapes, 'df_burst': df_burst, 'threshold_kwargs': threshold_kwargs}


@pytest.fixture(scope='session', autouse=True)
def check_dir():
    """Once, prior to session, this will clear and re-initialize the test file directories."""

    # If the directories already exist, clear them
    if os.path.exists(BASE_TEST_FILE_PATH):
        shutil.rmtree(BASE_TEST_FILE_PATH)

    # Remake (empty) directories
    os.mkdir(BASE_TEST_FILE_PATH)
    os.mkdir(TEST_PLOTS_PATH)


@pytest.fixture(scope='module')
def sim_stationary():

    sig = sim_oscillation(N_SECONDS, FS, FREQ, phase=0.15, cycle="asine", rdsym=.3)

    yield sig


@pytest.fixture(scope='module')
def sim_spike():

    cycle_params = {'centers': (.25, .5), 'stds':(.25, .2),
                'alphas':(8, .2), 'heights': (15, 2.5)}

    spike = _sim_ap_cycle(1, 100, **cycle_params, max_extrema='trough')

    sig = np.zeros(1000)

    for start in np.arange(200, 1000, 200):
        sig[start:start+100] = spike

    yield sig


###################################################################################################
###################################################################################################


def _sim_gaussian_cycle(n_seconds, fs, std, center=.5, height=1.):

    xs = np.linspace(0, 1, int(np.ceil(n_seconds * fs)))
    cycle = height * np.exp(-(xs-center)**2 / (2*std**2))

    return cycle


def _sim_skewed_gaussian_cycle(n_seconds, fs, center, std, alpha, height=1):

    from scipy.stats import norm

    n_samples = int(np.ceil(n_seconds * fs))

    # Gaussian distribution
    cycle = _sim_gaussian_cycle(n_seconds, fs, std/2, center, height)

    # Skewed cumulative distribution function.
    #   Assumes time are centered around 0. Adjust to center around 0.5.
    times = np.linspace(-1, 1, n_samples)
    cdf = norm.cdf(alpha * ((times - ((center * 2) -1 )) / std))

    # Skew the gaussian
    cycle = cycle * cdf

    # Rescale height
    cycle = (cycle / np.max(cycle)) * height

    return cycle


def _sim_ap_cycle(n_seconds, fs, centers, stds, alphas, heights, max_extrema='peak'):

    polar = _sim_skewed_gaussian_cycle(n_seconds, fs, centers[0], stds[0],
                                      alphas[0], height=heights[0])

    repolar = _sim_skewed_gaussian_cycle(n_seconds, fs, centers[1], stds[1],
                                        alphas[1], height=heights[1])

    cycle = polar - repolar

    if max_extrema == 'trough':
        cycle = -cycle

    return cycle
