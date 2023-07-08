"""Settings for bycycle tests."""

import os
from pathlib import Path

###################################################################################################
###################################################################################################

# Settings for test simulations
N_SECONDS = 10
FS = 1000
FREQ = 10
F_RANGE = (6, 14)

# Path Settings
TESTS_PATH = Path(os.path.abspath(os.path.dirname(__file__)))
BASE_TEST_FILE_PATH = TESTS_PATH / 'test_files'
TEST_PLOTS_PATH = os.path.join(BASE_TEST_FILE_PATH, 'plots')
