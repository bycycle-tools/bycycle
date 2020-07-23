"""Configuration file for pytest for bycycle."""

import os
import shutil

import pytest

from neurodsp.sim import sim_oscillation
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

    df_shapes, df_samples = compute_shape_features(sig, FS, F_RANGE, return_samples=True)
    df_burst = compute_burst_features(df_shapes, df_samples, sig)
    df_features, df_samples = compute_features(sig, FS, F_RANGE, return_samples=True)

    threshold_kwargs = {'amp_fraction_threshold': 0.,
                        'amp_consistency_threshold': .5,
                        'period_consistency_threshold': .5,
                        'monotonicity_threshold': .5,
                        'n_cycles_min': 3}

    yield {'sig': sig, 'fs': FS, 'f_range': F_RANGE, 'df_features': df_features,
           'df_shapes': df_shapes, 'df_burst': df_burst, 'df_samples': df_samples,
           'threshold_kwargs': threshold_kwargs}


@pytest.fixture(scope='session', autouse=True)
def check_dir():
    """Once, prior to session, this will clear and re-initialize the test file directories."""

    # If the directories already exist, clear them
    if os.path.exists(BASE_TEST_FILE_PATH):
        shutil.rmtree(BASE_TEST_FILE_PATH)

    # Remake (empty) directories
    os.mkdir(BASE_TEST_FILE_PATH)
    os.mkdir(TEST_PLOTS_PATH)
