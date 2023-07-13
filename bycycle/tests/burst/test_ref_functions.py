# test reference functions
import matplotlib.pyplot as plt
from bycycle.burst import create_window_indices_from_signal, get_cycle_bounds, select_bursting_cycles, plot_bounded_windows, get_bursts_windows_dualthresh
from neurodsp.sim import sim_powerlaw
from neurodsp.sim import sim_bursty_oscillation
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt


FS = 500


class TestRefFunctions(TestCase):
    def create_signals(self):
        n_seconds = 10
        # bursts and signals taken from tutorial pages.
        burst0 = sim_bursty_oscillation(n_seconds=n_seconds, fs=FS, freq=10, burst_def='durations', burst_params={
            'n_cycles_burst': 3, 'n_cycles_off': 3})
        burst1 = sig = sim_bursty_oscillation(n_seconds=n_seconds, fs=FS, freq=10, burst_params={
            'enter_burst': 0.2, 'leave_burst': 0.8})
        burst2 = sim_bursty_oscillation(n_seconds=n_seconds, fs=FS, freq=10, burst_def='durations',
                                        burst_params={'n_cycles_burst': 3, 'n_cycles_off': 5})
        burst3 = sim_bursty_oscillation(n_seconds=n_seconds, fs=FS, freq=10, burst_def='durations', burst_params={
            'n_cycles_burst': 8, 'n_cycles_off': 20})
        burst4 = sig = sim_bursty_oscillation(n_seconds=n_seconds, fs=FS, freq=10, burst_params={
            'enter_burst': 0.2, 'leave_burst': 0.8})
        burst5 = sim_bursty_oscillation(n_seconds=n_seconds, fs=FS, freq=10, burst_def='durations',
                                        burst_params={'n_cycles_burst': 3, 'n_cycles_off': 3})

        sig0 = sim_powerlaw(n_seconds=n_seconds, fs=FS, exponent=-2.0)

        sig1 = sim_powerlaw(n_seconds=n_seconds, fs=FS,
                            exponent=-1.5, f_range=(2, None))
        sig2 = sim_powerlaw(n_seconds=n_seconds, fs=FS, exponent=-0.5)

        sig3 = sim_powerlaw(n_seconds=n_seconds, fs=FS,
                            exponent=-3, f_range=(2, None))

        bursts = [burst1, burst2, burst3, burst4, burst5]
        # bursts = [burst0]
        sigs = [sig0, sig1, sig2, sig3]
        # sigs = [sig0]
        # ratios = [10, 1, 0.5, 0.1, 0.0]
        ratios = [10, 1, 0.5]
        # ratios=[20,10]
        combined_sigs = [None]*(len(bursts) * len(sigs)*len(ratios))

        # for y_vals in all_to_plot:
        #     plot_time_series(times=times, sigs=y_vals)
        # tested, looks good.

        for i in range(len(bursts)):
            for j in range(len(sigs)):
                for k in range(len(ratios)):
                    combined_sigs[len(sigs)*len(ratios)*i + len(ratios)
                                  * j+k] = bursts[i]+10*ratios[len(ratios)-k-1]*sigs[j]

        return combined_sigs

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
        self.assertFalse()
        print("hello")

    def test_map_cycles_to_windows(self):
        combined_sigs = self.create_signals()
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
