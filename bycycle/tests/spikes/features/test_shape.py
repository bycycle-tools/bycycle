"""Tests for shape features."""

import pytest
import numpy as np

from bycycle.spikes.features.shape import (
    compute_shape_features, compute_symmetry, compute_durations, compute_voltages
)

###################################################################################################
###################################################################################################

def test_compute_shape_features(sim_spikes, sim_spikes_df):


    sig = sim_spikes['sig']
    df_samples = sim_spikes_df['df_samples']

    df_shape = compute_shape_features(df_samples, sig)

    columns = [
        'period', 'time_trough', 'volt_trough', 'volt_last_peak', 'volt_next_peak',
        'volt_decay', 'volt_rise', 'time_decay', 'time_rise', 'time_decay_sym', 'time_rise_sym',
    ]

    for column in columns:
        assert column in df_shape.columns


def test_compute_symmetry(sim_spikes_df):

    df_samples = sim_spikes_df['df_samples']

    sym_features = compute_symmetry(df_samples)

    columns = ['time_decay', 'time_rise', 'time_decay_sym', 'time_rise_sym']

    for column in columns:

        assert column in sym_features.keys()

        if 'sym' in column:
            assert sym_features[column].dtype == 'float64'
        else:
            assert sym_features[column].dtype == 'int'


def test_compute_voltages(sim_spikes, sim_spikes_df):

    sig = sim_spikes['sig']
    df_samples = sim_spikes_df['df_samples']

    volts = compute_voltages(df_samples, sig)

    volt_trough, volt_last_peak, volt_next_peak, volt_decay, volt_rise = volts

    non_trough_volts = [volt_last_peak, volt_next_peak, volt_decay, volt_rise]

    for volt in volts:
        assert isinstance(volt, np.ndarray)

    for volt in non_trough_volts:
        assert (volt_trough < volt).all()


def test_compute_durations(sim_spikes_df):

    df_samples = sim_spikes_df['df_samples']

    period, time_trough = compute_durations(df_samples)

    assert period.dtype == 'int64'
    assert time_trough.dtype == 'int64'

    assert (period > time_trough).all()
