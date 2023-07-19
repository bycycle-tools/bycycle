# test reference functions
import matplotlib.pyplot as plt
from bycycle.burst import create_window_indices_from_signal, get_cycle_bounds, select_bursting_cycles, plot_bounded_windows, get_bursts_windows_dualthresh, get_signal_windows, create_signals, create_signals_burst_table, detect_bursts_amp
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import bycycle
from bycycle.tests.utils import *
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.signal import resample

from neurodsp.sim.periodic import make_is_osc_durations

FS = 500



def test_clustering_neurodsp_amp_function():
    # combined_sigs = create_signals(3,8)
    combined_sigs, ground_truth = create_signals_burst_table(nb=5,na=5,fs=FS,freq=10, n_seconds=10)

    for i in range(len(combined_sigs)):
        
        curr_sig = combined_sigs[i]

        bm = bycycle.Bycycle(
            burst_method='amp',
            thresholds={
                "burst_fraction_threshold": 0.5,
                "min_n_cycles": 1
            },
            burst_kwargs={
                "amp_threshes": (1, 1)
            }
        )
        bm.fit(sig=curr_sig, fs=FS, f_range=(8, 12))

        cycle_bounds_all = get_cycle_bounds(bm)

        bursting_cycle_idxs = bm.df_features.index[bm.df_features['is_burst'] == True]
        bursting_cycle_bounds = [cycle_bounds_all[i] for i in bursting_cycle_idxs]
        
        is_burst_pred = np.zeros(len(curr_sig), dtype=bool)

        for j in range(len(bursting_cycle_bounds)):
            is_burst_pred[bursting_cycle_bounds[j][0]:bursting_cycle_bounds[j][1]] = True

        score = (ground_truth == is_burst_pred).mean()

        # plot_bounded_windows(
        #     curr_sig, bursting_cycle_idxs, cycle_bounds_all)
    # plt.show()