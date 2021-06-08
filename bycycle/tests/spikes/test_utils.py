"""Tests for utility functions."""

import pandas as pd

from bycycle.spikes.utils import create_cyclepoints_df, split_signal, rename_df

###################################################################################################
###################################################################################################

def test_create_cyclepoints_df(sim_spikes):

    sig = sim_spikes['sig']

    starts = [0, 10, 20]
    ends = [start+9 for start in starts]
    decays = [start+2 for start in starts]
    troughs = [start+4 for start in starts]
    rises = [start+6 for start in starts]

    last_peaks = starts
    next_peaks = ends
    next_decays = ends

    df_samples = create_cyclepoints_df(sig, starts, decays, troughs, rises,
                                       last_peaks, next_peaks, ends)

    keys = [
        'sample_start', 'sample_decay', 'sample_trough', 'sample_rise',
        'sample_last_peak', 'sample_next_peak', 'sample_end'
    ]

    assert isinstance(df_samples, pd.DataFrame)

    for key in keys:
        assert key in df_samples.columns


def test_split_signal(sim_spikes, sim_spikes_fit):

    sig = sim_spikes['sig']

    spikes = sim_spikes_fit['spikes']

    df_samples = spikes.df_features.copy()

    sig_spikes = split_signal(df_samples, sig)

    assert (sig_spikes == spikes.spikes).all()


def test_rename_df(sim_spikes_fit):

    df_features = sim_spikes_fit['spikes'].df_features.copy()

    df_features_rename = rename_df(df_features)

    assert (df_features.columns != df_features.rename).any()
    assert 'sample_peak' in df_features_rename.columns
    assert 'sample_last_trough' in df_features_rename.columns
    assert 'sample_next_trough' in df_features_rename.columns
