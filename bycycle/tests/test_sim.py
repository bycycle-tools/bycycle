"""Tests simulating oscillations and 1/f noise."""

from bycycle import sim
import pytest

###################################################################################################
###################################################################################################

@pytest.mark.parametrize("filter_f_range", [None, (6, 14), (6, None)])
@pytest.mark.parametrize("N", [2, 3])
def test_sim_filtered_brown_noise(filter_f_range, N):
    """Test simulating brown noise."""
    Fs = 1000
    T = 25
    brownNf = sim.sim_filtered_brown_noise(T, Fs, filter_f_range, N)
    assert len(brownNf) == Fs * T


def test_sim_oscillator():
    """Test simulating a stationary oscillation."""
    T = 25
    Fs = 1000
    freq = 5
    osc = sim.sim_oscillator(T, Fs, freq, rdsym=.5)
    assert len(osc) == T * Fs


def test_sim_noisy_oscillator():
    """Test simulating an oscillation with brown noise."""
    T = 25
    Fs = 1000
    freq = 5
    osc = sim.sim_noisy_oscillator(T, Fs, freq, rdsym=.5)
    assert len(osc) == T * Fs


@pytest.mark.parametrize("return_cyc_df", [True, False])
def test_sim_bursty_oscillator(return_cyc_df):
    """Test bursty oscillation."""
    T = 25
    Fs = 1000
    freq = 4
    cyc_features = {'amp_mean': 1, 'amp_burst_std': .1, 'amp_std': .2}

    if return_cyc_df:
        osc, df = sim.sim_bursty_oscillator(T, Fs, freq,
                                            cycle_features=cyc_features,
                                            return_cycle_df=return_cyc_df)
    else:
        osc = sim.sim_bursty_oscillator(T, Fs, freq,
                                        cycle_features=cyc_features,
                                        return_cycle_df=return_cyc_df)

    if return_cyc_df:
        putative_cycles = (T * 1000)/(Fs / freq)
        actual_cycles = df.shape[0]
        assert round(actual_cycles / putative_cycles) * putative_cycles \
            == putative_cycles

    assert len(osc) == T * Fs


@pytest.mark.parametrize("f_hipass_brown", [2, 3])
@pytest.mark.parametrize("return_cycle_df", [True, False])
@pytest.mark.parametrize("return_comps", [True, False])
def test_sim_noisy_bursty_oscillator(f_hipass_brown, return_cycle_df,
                                     return_comps):
    """Test bursty oscillation with noise."""
    T = 25
    Fs = 1000
    freq = 4

    if return_comps and return_cycle_df:
        signal, oscillator, brown, df = \
            sim.sim_noisy_bursty_oscillator(T, Fs, freq, f_hipass_brown,
                                            return_components=return_comps,
                                            return_cycle_df=return_cycle_df)
    elif return_comps:
        signal, oscillator, brown = \
            sim.sim_noisy_bursty_oscillator(T, Fs, freq, f_hipass_brown,
                                            return_components=return_comps,
                                            return_cycle_df=return_cycle_df)
    elif return_cycle_df:
        signal, df = \
            sim.sim_noisy_bursty_oscillator(T, Fs, freq, f_hipass_brown,
                                            return_components=return_comps,
                                            return_cycle_df=return_cycle_df)
    else:
        signal = \
            sim.sim_noisy_bursty_oscillator(T, Fs, freq, f_hipass_brown,
                                            return_components=return_comps,
                                            return_cycle_df=return_cycle_df)

    if return_comps:
        assert len(oscillator) == len(brown) == T * Fs

    if return_cycle_df:
        putative_cycles = (T * 1000)/(Fs / freq)
        actual_cycles = df.shape[0]
        assert round(actual_cycles / putative_cycles) * putative_cycles \
            == putative_cycles

    assert len(signal) == T * Fs
