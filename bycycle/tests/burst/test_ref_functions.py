# test reference functions
import matplotlib.pyplot as plt
from bycycle.burst import create_window_indices_from_signal, get_cycle_bounds, select_bursting_cycles, plot_bounded_windows, get_bursts_windows_dualthresh
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import bycycle


FS = 500


class TestRefFunctions(TestCase):
    

    # if this function plots full cycles from trough to trough, then we call it successful
    def test_create_window_indices_sinusoid_1(self):
        sig_length = 20
        fs = 500
        x = np.linspace(0, sig_length, sig_length*fs)
        f = np.sin
        y = f(x)
        BM = bycycle.Bycycle()
        windows = create_window_indices_from_signal(BM, f(x), fs, 1)
        for i in range(len(windows)):
            # Plot the scatter plot with windows as colors
            # plt.plot(x[windows[i][0]:windows[i][1]], y[windows[i][0]:windows[i][1]])
            print("noop")
        # plt.show()
        # unsuccessful. Function plots first cycle from peak to peak and all others from rising zero-crossing to rising zero-crossing.
        print("hello")

    def test_map_cycles_to_windows(self):
        combined_sigs = self.create_signals(1)
        # for i in range(len(combined_sigs)):
        #     plt.plot(combined_sigs[i])
        self.assertFalse(combined_sigs == None)
        BM = bycycle.Bycycle()
        for i in range(len(combined_sigs)):
            curr_sig = combined_sigs[i]
            BM.fit(sig=curr_sig, fs=FS, f_range=(8, 12))
            burst_bounds, complement_bounds = get_bursts_windows_dualthresh(
                curr_sig, FS, (8, 12))
            cycle_bounds_all = get_cycle_bounds(BM)
            bursting_cycle_idxs = select_bursting_cycles(
                cycle_bounds_all, burst_bounds)
            nonbursting_cycle_idxs = select_bursting_cycles(
                cycle_bounds_all, complement_bounds)
            # plot_bounded_windows(curr_sig, bursting_cycle_idxs, cycle_bounds_all)
            # plot_bounded_windows(curr_sig, nonbursting_cycle_idxs, cycle_bounds_all)
        # plt.show()

    def test_clustering_kmeans(self):
        self.assertTrue(False)

    def test_clustering_neurodsp_amp_function(self):
        combined_sigs = self.create_signals()
        for i in range(len(combined_sigs)):
            curr_sig = combined_sigs[i]
            # plt.plot(combined_sigs[i])
            self.assertFalse(combined_sigs == None)
            BM = bycycle.Bycycle(burst_method='amp')
            BM.fit(sig=curr_sig, fs=FS, f_range=(8, 12))
            burst_bounds, complement_bounds = get_bursts_windows_dualthresh(
                curr_sig, FS, (8, 12))
            cycle_bounds_all = get_cycle_bounds(BM)
            new_features = detect_bursts_amp(BM.df_features)
            new_bursting_cycle_idxs = new_features.index[new_features['is_burst'] == True].tolist(
            )
            plot_bounded_windows(
                curr_sig, new_bursting_cycle_idxs, cycle_bounds_all)
        plt.show()
        print("hi")
