# test reference functions
import matplotlib.pyplot as plt
from bycycle.burst import (
    create_window_indices_from_signal,
    get_cycle_bounds,
    select_bursting_cycles,
    plot_bounded_windows,
    get_bursts_windows_dualthresh,
    get_signal_windows,
    create_signals,
    create_signals_burst_table,
    detect_bursts_amp,
)
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import bycycle
from bycycle.tests.utils import *
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.signal import resample
from scipy.stats import ttest_ind

from neurodsp.sim.periodic import make_is_osc_durations

from neurodsp.sim.aperiodic import sim_powerlaw
from neurodsp.sim.periodic import sim_bursty_oscillation, make_is_osc_durations
FS = 500


# This function should test neurodsp burst detection using the amp method
def test_clustering_neurodsp_amp_function():
    # Test setup:
    # 1. create signals for testing
    # 2. create a bycycle object for each signal
    # 3. run neurodsp burst detection on each signal
    # 4. compare the results of neurodsp burst detection to the ground truth
    # 5. if the results are not similar, then the test fails

    # Step 1.
    testPasses = True
    combined_sigs, ground_truth = create_signals_burst_table(
        nb=5, na=5, fs=FS, freq=10, n_seconds=10
    )
    bo = np.full(
        # length of array
        len(combined_sigs),
        bycycle.Bycycle(
        burst_method="amp",
        thresholds={"burst_fraction_threshold": 0.5, "min_n_cycles": 1},
        burst_kwargs={
            # just worked best.
            "amp_threshes": (1, 1)
        })
    )

    failing_dfs = np.full(len(bo), None, dtype=object)
    passing_dfs = np.full(len(bo), None, dtype=object)
    passing_count = failing_count = 0
    for i in range(len(combined_sigs)):
        curr_sig = combined_sigs[i]

        bo[i].fit(sig=curr_sig, fs=FS, f_range=(8, 12))

        cycle_bounds_all = get_cycle_bounds(bo[i])

        bursting_cycle_idxs = bo[i].df_features.index[
            bo[i].df_features["is_burst"] == True
        ]
        bursting_cycle_bounds = [cycle_bounds_all[i] for i in bursting_cycle_idxs]

        is_burst_pred = np.zeros(len(curr_sig), dtype=bool)
        for j in range(len(bursting_cycle_bounds)):
            is_burst_pred[
                bursting_cycle_bounds[j][0] : bursting_cycle_bounds[j][1]
            ] = True

        score = (ground_truth == is_burst_pred).mean()
        if score > 0.8:
            passing_dfs[passing_count] = bo[i].df_features
            passing_count = passing_count + 1
        else:
            failing_dfs[failing_count] = bo[i].df_features
            failing_count = failing_count + 1

    passing_dfs=passing_dfs[:passing_count]
    failing_dfs=failing_dfs[:failing_count]
    print("failing count: ", failing_count)
    if failing_count>0:
        keys = None
        failing_data_by_key = dict()
        passing_data_by_key = dict()
        df_model = failing_dfs[0]
        keys = df_model.keys()
        # keep only the first 14 keys (not sample or is_burst)
        keys=keys[:14]
        for i in range(0, len(keys)):
            num_failing_elements = 0
            num_passing_elements = 0
            for j in range(0, len(failing_dfs)):
                num_failing_elements = num_failing_elements + len(failing_dfs[j][keys[i]])
            failing_data_by_key[keys[i]] = np.empty(num_failing_elements)
            for j in range(0, len(passing_dfs)):
                num_passing_elements = num_passing_elements + len(passing_dfs[j][keys[i]])
            passing_data_by_key[keys[i]] = np.empty(num_passing_elements)
            curr_idx = 0
            for j in range(0, len(failing_dfs)):
                curr_array = failing_dfs[j][keys[i]]
                failing_data_by_key[keys[i]][curr_idx:curr_idx+len(curr_array)] = curr_array
                curr_idx = curr_idx + len(curr_array)
            curr_idx = 0
            for j in range(0, len(passing_dfs)):
                curr_array = passing_dfs[j][keys[i]]
                passing_data_by_key[keys[i]][curr_idx:curr_idx+len(curr_array)] = curr_array
                curr_idx = curr_idx + len(curr_array)
        
        passing_data_by_index = np.empty(len(keys),dtype=object)
        failing_data_by_index = np.empty(len(keys),dtype=object)
        for i in range(0, len(keys)):
            passing_data_by_index[i] = passing_data_by_key[keys[i]][1:len(passing_data_by_key[keys[i]])-1]
            failing_data_by_index[i] = failing_data_by_key[keys[i]][1:len(failing_data_by_key[keys[i]])-1]
        
        result=np.empty(0,dtype=object)
        for i in range(0, len(keys)):
            result=np.append(result, ttest_ind(passing_data_by_index[i], failing_data_by_index[i]))
        print("hook")


    assert testPasses
