
import sys
from neurodsp.burst import detect_bursts_dual_threshold as dualthresh
from statsmodels.tsa.stattools import acf as autocorrelate
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from neurodsp.sim.aperiodic import sim_powerlaw
from neurodsp.sim.periodic import sim_bursty_oscillation

# NOTE: `create_*` functions that take a bycycle model as input DO call bm.fit.
# These methods assume the model has already been fit.

def create_signals(nb, na, fs, freq, n_seconds):
        n_seconds = 10
        # bursts and signals taken from tutorial pages.
        burst0 = sim_bursty_oscillation(n_seconds=n_seconds, fs=fs, freq=10, burst_def='durations', burst_params={
            'n_cycles_burst': nb, 'n_cycles_off': na})

        sig0 = sim_powerlaw(n_seconds=n_seconds, fs=fs, exponent=-2.0)

        sig1 = sim_powerlaw(n_seconds=n_seconds, fs=fs,
                            exponent=-1.5, f_range=(2, None))
        sig2 = sim_powerlaw(n_seconds=n_seconds, fs=fs, exponent=-0.5)

        sig3 = sim_powerlaw(n_seconds=n_seconds, fs=fs,
                            exponent=-3, f_range=(2, None))

        bursts = [burst0]
        sigs = [sig0, sig1, sig2, sig3]
        # sigs = [sig0]
        # ratios = [10, 1, 0.5, 0.1, 0.0]
        ratios = [10, 1, 0.5]
        # ratios=[20,10]
        combined_sigs = [None]*(len(bursts) * len(sigs)*len(ratios))

        # for y_vals in all_to_plot:
        #     plot_time_series(times=times, sigs=y_vals)
        # tested, looks good.

        # keeping in case we add more bursts down the line.
        for i in range(len(bursts)):
            for j in range(len(sigs)):
                for k in range(len(ratios)):
                    combined_sigs[len(sigs)*len(ratios)*i + len(ratios)
                                  * j+k] = bursts[i]+ratios[len(ratios)-k-1]*sigs[j]

        return combined_sigs

def create_signals_burst_table(nb, na, fs, freq, n_seconds):
    sigs = create_signals(nb, na, fs, freq, n_seconds)
    truth_table = np.full(fs*n_seconds,False)
    for i in range(n_seconds*fs):
        if i%(nb+na) < nb:
            truth_table[i]=True
    return sigs, truth_table

# Complexity: O(bm.fit + len(bm.df_features **after bm.fit**))
def create_window_indices_from_signal(bm=None, sig=None, fs=500, window_length=3):
    bm.fit(sig=sig, fs=fs, f_range=(8, 12))
    last_troughs = bm.df_features["sample_last_trough"]
    next_troughs = bm.df_features["sample_next_trough"]
    # print(last_troughs[len(last_troughs)-1])
    # print(next_troughs[len(last_troughs)-2])
    window_bound_collection = [None]*(len(last_troughs)-window_length)
    for i in range(window_length, len(last_troughs)):
        # make cycles disjoint
        window = (last_troughs[i-window_length], next_troughs[i-1]-1)
        window_bound_collection[i-window_length] = window
    # in disjoint cycle adjustment, last cycle needs to be full-length
    window_bound_collection[-1][1]=window_bound_collection[-1][1]+1
    return window_bound_collection

# NOTE: `get_*` functions that take a bycycle model as input do not call bm.fit.
# These methods assume the model has already been fit.

## Complexity: O(len(last_trough)
def get_cycle_bounds(bm=None):
    last_troughs = bm.df_features["sample_last_trough"]
    next_troughs = bm.df_features["sample_next_trough"]

    lt_len = len(last_troughs)

    cycle_bounds = [None]*lt_len
    for i in range(lt_len):
        cycle_bounds[i] = (last_troughs[i], next_troughs[i])
    return cycle_bounds

#  Complexity: O(len(window_idx_collection))
def get_signal_windows(sig, window_idx_collection):
    collection = [None]*len(window_idx_collection)
    for i in range(len(window_idx_collection)):
        collection[i] = sig[window_idx_collection[i]
                            [0]:window_idx_collection[i][1]]

    # for i in collection:
    #     plt.plot(np.linspace(0, len(i), len(i)), i)

    # plt.show()
    return collection

# maps cycles to windows which contain them.

# iteratively checks for each window, for each cycle, whether the
# infimum of the window is lt/eq the lower bound of the cycle
# AND the supremum of the window is gt/eq the upper bound of the cycle
# Complexity: O(num_cycle*num_windows)
def map_cycles_to_windows(cycle_bounds=None, window_indices=None):
    num_cycles = len(cycle_bounds)
    num_windows = len(window_indices)
    window_map_bools = [[False]*num_windows]*num_cycles
    retVal = [None]*num_cycles
    for i in range(num_cycles):
        for j in range(num_windows):
            current_cycle_bound = cycle_bounds[i]
            current_window_bound = window_indices[j]
            if (current_cycle_bound[0] >= current_window_bound[0]) and (current_cycle_bound[1] <= current_window_bound[1]):
                window_map_bools[i][j] = True
        print(num_cycles)
        print(num_windows)
        print(np.asarray(window_map_bools).shape)
        # fixed line
        truth_count = sum(window_map_bools[i])
        if truth_count>0:
            membership_set = np.zeros(truth_count)
            k=0
            print(window_map_bools[i])
            for j in range(num_windows):
                # BUG: this was reaching an oob error before I made `truth_count = sum(window_map_bools[i])``
                if window_map_bools[i][j]:
                    membership_set[k] = j
                    k+=1
            retVal[i] = membership_set
    return retVal

# INPUT: (OUTPUT FROM: get_cycle_bounds), (OUTPUT FROM: create_window_indices_from_signal)
# OUTPUT: vector from Z^(len(cycle_bounds))
# Complexity: O(num_cycle*num_windows)
# Can probably be optimized to linear by converting to binary and using binary operators.
# Save that for refactoring
def select_bursting_cycles(cycle_bounds=None, window_indices=None):
    map = map_cycles_to_windows(cycle_bounds, window_indices)
    boolVec = [False]*len(cycle_bounds)
    truths = 0
    for i in range(len(boolVec)):
        if map[i] is not None:
            boolVec[i]=True
            truths+=1
    retVec = np.zeros(truths)
    j=0
    for i in range(len(boolVec)):
        if map[i] is not None:
            retVec[j]=i
            j+=1
    return retVec

def plot_bounded_windows(sig, window_truth_array, window_bounds):
    plt.figure()
    longest_burst = 0
    window_bounds=[window_bounds[int(i)] for i in window_truth_array]
    for i in range(len(window_bounds)):
        curr_sig = sig[window_bounds[i][0]:window_bounds[i][1]]
        length = len(curr_sig)
        if length > longest_burst:
            longest_burst = length

    # new_x = np.linspace(0,longest_burst,longest_burst)
    for i in range(len(window_bounds)):
        resampled_sig = resample(x=sig[window_bounds[i][0]:window_bounds[i][1]], num=longest_burst)
        mean = np.mean(resampled_sig)
        resampled_sig-=mean
        plt.plot(resampled_sig)


def extract_cleaned_hyperparameters_from_signal(bm=None, sig=None, fs=500, window_length=3):
    bm.fit(sig=sig, fs=fs, f_range=(8, 12))
    features = bm.df_features
    keys = features.keys()
    np_features = features.to_numpy()
    subindex_length = len(np_features[0])
    # Not sure the following comment is true:
    # # we actually won't be using this
    for i in range(len(np_features)):
        for j in range(subindex_length):
            if not (np.isfinite(np_features[i][j])):
                np_features[i][j] = -1

    np_features = np_features[1:len(np_features-2)]
    return keys, np_features


# def autocorrelate_signal_wrong(sig):
#     slen = len(sig)
#     shifts = [sig]*slen
#     for i in range(slen):
#         for j in range(slen):
#             shifts[i][j] = sig[(j+i) % slen]
#     df = pd.DataFrame(shifts)
#     corr = df.corr(method='pearson')
#     return corr


def autocorrelate_signal(sig):
    return autocorrelate(sig)


def autocorrelate_all_windowed_signals(sig_window_collection):
    result_length = len(sig_window_collection)
    result = [None]*result_length
    for i in range(result_length):
        result[i] = autocorrelate_signal(sig_window_collection[i])
    return result


def get_bursts_windows_dualthresh(current_signal, fs, f_range=(8,12), min_burst_duration=3):
    longest = 0
    shortest = 0
    last_mode = False
    last_true = 0
    last_false = 0
    bursts = [None]*len(current_signal)
    complements = [None]*len(current_signal)
    bursts_idx = 0
    complements_idx = 0
    dt_burst = dualthresh(sig=current_signal, fs=fs, dual_thresh=(
        1, 2), f_range=f_range, min_n_cycles=1, min_burst_duration=min_burst_duration)
    for i in range(len(current_signal)):
        if i >= 1:
            # print(dt_burst[i])
            new_mode = dt_burst[i]
            if new_mode != last_mode:
                if new_mode:
                    interval = i-last_true
                    longest = np.max((interval, longest))
                    shortest = np.min((interval, shortest))
                    complements[complements_idx] = (last_true, i)
                    complements_idx += 1
                else:
                    interval = i-last_false
                    longest = np.max((interval, longest))
                    shortest = np.min((interval, shortest))
                    bursts[bursts_idx] = (last_false, i)
                    bursts_idx += 1
        last_mode = dt_burst[i]
        if last_mode:
            last_true = i
        else:
            last_false = i
    num_burst_cycles = 0
    num_complement_cycles = 0
    check_burst_list, check_complement_list = True, True
    for i in range(len(current_signal)):
        if not check_burst_list and not check_complement_list:
            break
        if bursts[i] == None:
            check_burst_list = False
        if complements[i] == None:
            check_complement_list = False
        if check_burst_list:
            num_burst_cycles += 1
        if check_complement_list:
            num_complement_cycles += 1

    bursts = bursts[:num_burst_cycles]
    complements = complements[:num_complement_cycles]
    return bursts, complements
