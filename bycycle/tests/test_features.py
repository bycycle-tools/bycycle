"""Tests the main cycle-by-cycle feature computation function

NOTES
-----
The tests here are not strong tests for accuracy.
    They serve rather as 'smoke tests', for if anything fails completely.

Code to generate data from bycycle v.0.1.0 before sim module was removed:

from bycycle import sim
import numpy as np

# Stationary oscillator
np.random.seed(0)
cf = 10 # Oscillation center frequency
T = 10 # Recording duration (seconds)
Fs = 1000 # Sampling rate

rdsym = .3
signal = sim.sim_oscillator(T, Fs, cf, rdsym=rdsym)
np.save('sim_stationary.npy', signal)

# Bursting oscillator
np.random.seed(0)
cf = 10 # Oscillation center frequency
T = 10 # Recording duration (seconds)
Fs = 1000 # Sampling rate

signal = sim.sim_noisy_bursty_oscillator(T, Fs, cf, prob_enter_burst=.1,
                                         prob_leave_burst=.1, SNR=5)
np.save('sim_bursting.npy', signal)
"""

import numpy as np
from bycycle import features


def test_compute_features():
    """Test cycle-by-cycle feature computation"""

    # Load signal
    signal = np.load('data/sim_stationary.npy')
    Fs = 1000  # Sampling rate
    f_range = (6, 14)  # Frequency range

    # Compute cycle features
    df = features.compute_features(signal, Fs, f_range)

    # Check inverted signal gives appropriately opposite data
    df_opp = features.compute_features(-signal, Fs, f_range,
                                       center_extrema='T')

    np.testing.assert_allclose(df['sample_peak'], df_opp['sample_trough'])
    np.testing.assert_allclose(df['sample_last_trough'], df_opp['sample_last_peak'])
    np.testing.assert_allclose(df['time_peak'], df_opp['time_trough'])
    np.testing.assert_allclose(df['time_rise'], df_opp['time_decay'])
    np.testing.assert_allclose(df['volt_rise'], df_opp['volt_decay'])
    np.testing.assert_allclose(df['volt_amp'], df_opp['volt_amp'])
    np.testing.assert_allclose(df['period'], df_opp['period'])
    np.testing.assert_allclose(df['time_rdsym'], 1 - df_opp['time_rdsym'])
    np.testing.assert_allclose(df['time_ptsym'], 1 - df_opp['time_ptsym'])
