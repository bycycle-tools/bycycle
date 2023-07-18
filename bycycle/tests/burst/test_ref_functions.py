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
        # combined_sigs = create_signals(4,6)
        combined_sigs = create_signals(3,8)
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
            plot_bounded_windows(curr_sig, bursting_cycle_idxs, cycle_bounds_all)
            plot_bounded_windows(curr_sig, nonbursting_cycle_idxs, cycle_bounds_all)
        plt.show()

    def test_clustering_kmeans(self):
        bm=bycycle.Bycycle()
        combined_sigs, ground_truth = create_signals_burst_table(nb=4,na=6,fs=FS,freq=8, n_seconds=10)
        our_findings = np.full(len(ground_truth), False)
        sig = combined_sigs[1]
        idxs = create_window_indices_from_signal(bm=bm, sig=sig, fs=FS,window_length=1)
        sig_windows = get_signal_windows(sig, idxs)
        sig_windows_homogeneous_shape = [None]*len(sig_windows)

        longest_signal = 0
        for sig in sig_windows:
            length = len(sig)
            if length > longest_signal:
                longest_signal = length

        new_x = np.linspace(0,longest_signal,longest_signal)
        for i in range(len(sig_windows)):
            # spline = make_interp_spline(np.linspace(0,len(sig_windows[i]),len(sig_windows[i])), sig_windows[i])
            # sig_windows_homogeneous_shape[i]=spline(new_x)
            resampled_sig = resample(sig_windows[i],longest_signal)
            resampled_sig-=np.mean(resampled_sig)
            print(len(resampled_sig))
            sig_windows_homogeneous_shape[i]=resampled_sig
            
        scaler = preprocessing.StandardScaler().fit(sig_windows_homogeneous_shape)
        norm_homog_coll = scaler.transform(sig_windows_homogeneous_shape)
        # print(sig_windows.)
        # print(type(sig_windows))

        group0 = []
        group1 = []
        kmeans_model = KMeans(n_clusters=2, random_state=0, n_init="auto")
        kmeans = kmeans_model.fit_predict(norm_homog_coll)
        for i in range(len(kmeans)):
            if kmeans[i]==0:
                group0.append(idxs[i])
            else:
                group1.append(idxs[i])
        for i in range(len(group0)):
            for j in range(group0[i][0], group0[i][1]):
                our_findings[j]=True
        score = 0
        for i in range(len(ground_truth)):
            if ground_truth[i]==our_findings[i]:
                score+=1.0/float(len(ground_truth))
        print("wait")
        # TODO: if you ever uncomment this code, use subfigures.
        # plt.figure()
        # for i in range(len(group0)):
        #     plt.plot(np.linspace(0,len(group0[i]),len(group0[i])), group0[i], alpha=0.5)
        # plt.figure()
        # for i in range(len(group1)):
        #     plt.plot(np.linspace(0,len(group1[i]),len(group1[i])), group1[i], alpha=0.5)
        # plt.show()

        # print(len(group0))
        # print(len(group1))

    print("done")


    def test_clustering_neurodsp_amp_function(self):
        # combined_sigs = create_signals(3,8)
        combined_sigs, ground_truth = create_signals_burst_table(nb=15,na=0,fs=FS,freq=8, n_seconds=10)
        our_findings = np.full(len(ground_truth), False)
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
            new_bursting_cycle_bounds = [cycle_bounds_all[i] for i in new_bursting_cycle_idxs]
            # print("hi")
            for j in range(len(new_bursting_cycle_bounds)):
                for k in range(new_bursting_cycle_bounds[j][0], new_bursting_cycle_bounds[j][1]):
                    our_findings[k]=True
            score = (ground_truth == our_findings).mean()
            print(score)
            # plot_bounded_windows(
            #     curr_sig, new_bursting_cycle_idxs, cycle_bounds_all)
        # plt.show()

    def test_evaluate_clustering(self):
        print("")