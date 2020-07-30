
"""Test interpolating phase from cyclepoints."""

import os
import numpy as np

from bycycle.cyclepoints.extrema import find_extrema, find_zerox
from bycycle.cyclepoints.phase import extrema_interpolated_phase

# Set data path
DATA_PATH = os.getcwd() + '/tutorials/data/'

###################################################################################################
###################################################################################################

def test_extrema_interpolated_phase():
    """Test waveform phase estimate."""

    # Load signal
    sig = np.load(DATA_PATH + 'sim_stationary.npy')

    fs = 1000
    f_range = (6, 14)

    # Find peaks and troughs
    peaks, troughs = find_extrema(sig, fs, f_range, boundary=1, first_extrema='peak')

    # Find zerocrossings
    rises, decays = find_zerox(sig, peaks, troughs)

    # Compute phase
    pha = extrema_interpolated_phase(sig, peaks, troughs, rises=rises, decays=decays)

    assert len(pha) == len(sig)
    assert np.all(np.isclose(pha[peaks], 0))
    assert np.all(np.isclose(pha[troughs], -np.pi))
    assert np.all(np.isclose(pha[rises], -np.pi/2))
    assert np.all(np.isclose(pha[decays], np.pi/2))
