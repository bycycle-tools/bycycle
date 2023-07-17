
# code and ideas in this document are adopted from https://neurodsp-tools.github.io/neurodsp/
# Thank you to all those who came before me, for your contributions.
# -Kenton Guarian

# general imports
from bycycle.tests.utils import *
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from neurodsp.utils import create_times
from neurodsp.utils import create_freqs

# imports for simulating periodic component
from neurodsp.sim import sim_bursty_oscillation
from neurodsp.sim import sim_variable_oscillation


# for simulating aperiodic component.
from neurodsp.sim import sim_powerlaw

# for plotting
from neurodsp.plts import plot_time_series
from neurodsp.plts import plot_power_spectra
from neurodsp.spectral import compute_spectrum
import matplotlib.pyplot as plt

# for bycycle analysis
import numpy as np
import sys
# from scipy.interpolate import make_interp_spline
from scipy.signal import resample

def test_report(self):
    for y_vals in combined_sigs:
        freqs,powers = compute_spectrum(y_vals, fs)
        bm.fit(y_vals,fs,(2,40))
        bm.report()

        # uncomment if you want the code to actually plot
        plot_time_series(times=times, sigs=y_vals)
        plot_power_spectra(freqs=freqs, powers=powers, fs=fs)
        bm.plot()
        plt.show()
