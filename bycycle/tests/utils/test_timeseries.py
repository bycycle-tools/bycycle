"""Tests for utils.timeseries."""

import numpy as np

from bycycle.utils.timeseries import *

###################################################################################################
###################################################################################################

def test_limit_signal(sim_args):

    sig = sim_args['sig']
    fs = sim_args['fs']

    times = np.arange(0, len(sig) / fs, 1 / fs)
    xlim = (1, 2)

    sig_short, times_short = limit_signal(times, sig, start=xlim[0], stop=xlim[1])

    assert np.array_equal(times_short, times[fs*xlim[0]:fs*xlim[1]])
    assert np.array_equal(sig_short, sig[fs*xlim[0]:fs*xlim[1]])
