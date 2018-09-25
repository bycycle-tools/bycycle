# Code to generate data from bycycle v.0.1.0 before sim module was removed

from bycycle import sim
import numpy as np

# Stationary oscillator
np.random.seed(0)
cf = 10 # Oscillation center frequency
T = 10 # Recording duration (seconds)
Fs = 1000 # Sampling rate

rdsym = .3
signal = sim.sim_oscillator(T, Fs, cf, rdsym=rdsym)
np.save('data/sim_stationary.npy', signal)

# Bursting oscillator
np.random.seed(0)
cf = 10 # Oscillation center frequency
T = 10 # Recording duration (seconds)
Fs = 1000 # Sampling rate

signal = sim.sim_noisy_bursty_oscillator(T, Fs, cf, prob_enter_burst=.1,
                                         prob_leave_burst=.1, SNR=5)
np.save('data/sim_bursting.npy', signal)

# Bursting oscillator with worse SNR
cf = 10
T = 10
Fs = 1000
np.random.seed(0)

signal = sim.sim_noisy_bursty_oscillator(T, Fs, cf, prob_enter_burst=.1,
                                         prob_leave_burst=.1, SNR=2,
                                         cycle_features={'rdsym_mean': .3})
np.save('data/sim_bursting_more_noise.npy', signal)

# Experiment with 20 people
np.random.seed(0)
cf = 10 # Oscillation center frequency
T = 10 # Recording duration (seconds)
Fs = 1000 # Sampling rate

N = 20
rdsyms = (.5, .35)
signals = np.zeros((N, int(Fs * T)))
for i in range(N):
    if i >= int(N / 2):
        rdsym = rdsyms[1]
    else:
        rdsym = rdsyms[0]

    signal = sim.sim_noisy_bursty_oscillator(T, Fs, cf, prob_enter_burst=.1,
                                             prob_leave_burst=.2, SNR=2,
                                             cycle_features={'rdsym_mean': rdsym})
    signals[i] = signal
np.save('data/sim_experiment.npy', signals)
