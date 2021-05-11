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

from bycycle import Spikes
from bycycle.spikes.features.gaussians import _sim_ap_cycle
from bycycle.spikes.cyclepoints import compute_spike_cyclepoints

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
def sim_spikes():

    # Simulate 5, 3-gaussian spikes
    cycle_params = {'centers': (.4, .5, .6), 'stds': (.1, .1, .1),
                    'alphas': (-1, 0, 1), 'heights': (10, -30, 20)}

    spike = _sim_ap_cycle(1, 100, **cycle_params)

    sig = np.zeros(1100)

    starts = np.arange(100, 1100, 200)
    ends = starts + 100

    for start in starts:
        sig[start:start+100] = spike

    # Pad edges
    pad = 500
    sig = np.pad(sig, pad)
    starts += pad

    # Simulate overlapping spikes
    cycle_params = {'centers': (.25, .5, .75), 'stds': (.05, .05, .05),
                    'alphas': (0, 0, 0), 'heights': (-14, -30, -14)}

    spike_overlap = _sim_ap_cycle(1, 100, **cycle_params)

    sig_overlap = np.zeros_like(sig)

    for start in starts:
        sig_overlap[start:start+100] = spike_overlap

    # Simulate prunable spikes
    cycle_params = {'centers': (.25, .5, .75), 'stds': (.05, .05, .05),
                    'alphas': (0, 0, 0), 'heights': (-30, -20, -30)}

    spike_prune = _sim_ap_cycle(1, 100, **cycle_params)

    sig_prune = np.zeros_like(sig)

    for start in starts:
        sig_prune[start:start+100] = spike_prune

    # Simulate Na current
    spike_na = _sim_ap_cycle(1, 100, .5, .1, 0, -20)

    sig_na = np.zeros_like(sig)

    for start in starts:
        sig_na[start:start+100] = spike_na

    # Simulate Na+K current
    cycle_params = {'centers': (.4, .6), 'stds': (.1, .1),
                    'alphas': (0, .2), 'heights': (-30, 15)}

    spike_na_k = _sim_ap_cycle(1, 100, **cycle_params)

    sig_na_k = np.zeros_like(sig)

    for start in starts:
        sig_na_k[start:start+100] = spike_na_k

    # Simulate Na+Conductive current
    cycle_params = {'centers': (.4, .5), 'stds': (.1, .1),
                    'alphas': (0, 0), 'heights': (15, -30)}

    spike_na_cond = _sim_ap_cycle(1, 100, **cycle_params)

    sig_na_cond = np.zeros_like(sig)

    for start in starts:
        sig_na_cond [start:start+100] = spike_na_cond

    yield {'sig': sig, 'sig_overlap': sig_overlap, 'sig_prune': sig_prune, 'sig_na': sig_na,
           'sig_na_k': sig_na_k, 'sig_na_cond': sig_na_cond, 'fs': 20000, 'f_range': (500, 3000),
           'spike':spike, 'locs':(starts, ends)}


@pytest.fixture(scope='module')
def sim_spikes_df(sim_spikes):

    sig = sim_spikes['sig']
    fs = sim_spikes['fs']
    f_range = sim_spikes['f_range']

    df_samples = compute_spike_cyclepoints(sig, fs, f_range, std=2)

    yield {'df_samples': df_samples}


@pytest.fixture(scope='module')
def sim_spikes_fit(sim_spikes):

    sig = sim_spikes['sig']
    fs = sim_spikes['fs']
    f_range = sim_spikes['f_range']

    spikes = Spikes()

    spikes.fit(sig, fs, f_range, n_gaussians=3, tol=1e-3)

    return {'spikes': spikes}
