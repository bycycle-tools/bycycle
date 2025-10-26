"""Tests for bycycle.utils.download."""

import os
import shutil

import numpy as np

from bycycle.utils.download import *

###################################################################################################
###################################################################################################

TEST_FOLDER = 'test_data'

def clean_up_downloads():

    shutil.rmtree(TEST_FOLDER)

###################################################################################################
###################################################################################################


def test_fetch_bycycle_data():

    filename = 'ca1.npy'

    fetch_bycycle_data(filename, folder=TEST_FOLDER)
    assert os.path.isfile(os.path.join(TEST_FOLDER, filename))


def test_load_bycycle_data():

    filename = 'ca1.npy'

    data = load_bycycle_data(filename, folder=TEST_FOLDER)
    assert isinstance(data, np.ndarray)

    clean_up_downloads()
