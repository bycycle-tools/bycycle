"""Test utilities for burst detection."""

import numpy as np

from bycycle.burst.utils import check_min_burst_cycles

###################################################################################################
###################################################################################################

def test_check_min_burst_cycles():

    is_burst = np.array([True, True, True, False])
    is_burst_check = check_min_burst_cycles(is_burst, min_n_cycles=3)

    assert (is_burst == is_burst_check).all()

    is_burst = np.array([True, False, True, False])
    is_burst_check = check_min_burst_cycles(is_burst, min_n_cycles=3)

    assert not any(is_burst_check)
