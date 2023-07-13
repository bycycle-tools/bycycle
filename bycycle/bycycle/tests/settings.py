"""Settings for bycycle tests."""

import os
import pkg_resources as pkg

###################################################################################################
###################################################################################################

# Settings for test simulations
N_SECONDS = 10
FS = 1000
FREQ = 10
F_RANGE = (6, 14)

# Path Settings
BASE_TEST_FILE_PATH = pkg.resource_filename(__name__, 'test_files')
TEST_PLOTS_PATH = os.path.join(BASE_TEST_FILE_PATH, 'plots')
