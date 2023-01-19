"""Tests for features.shape."""

import pytest

import numpy as np

from bycycle.features.shape import *

###################################################################################################
###################################################################################################

@pytest.mark.parametrize("find_extrema_kwargs",
    [
        None,
        pytest.param({'first_extrema': 'peak'}, marks=pytest.mark.xfail)
    ]
)
@pytest.mark.parametrize("center_extrema",
    [
        'peak',
        'trough',
        pytest.param(None, marks=pytest.mark.xfail)
    ]
)
def test_compute_shape_features(sim_args, find_extrema_kwargs, center_extrema):

    sig = sim_args['sig']
    fs = sim_args['fs']
    f_range = sim_args['f_range']

    df_shapes = compute_shape_features(sig, fs, f_range, center_extrema=center_extrema,
                                       find_extrema_kwargs=find_extrema_kwargs)

    # Ensure sample columns are returned
    sample_cols = [col for col in list(df_shapes.columns) if "sample_" in col]
    assert len(sample_cols) == 6

    # Assert that np.nan isn't in dataframe(s), with the exception of the first and last row
    for idx, row in enumerate(df_shapes.to_dict('records')):

        assert not np.isnan(row[1:-1]).any()

    # Check inverted signal gives appropriately opposite data
    extrema_opp = 'trough' if center_extrema == 'peak' else 'peak'

    df_opp = compute_shape_features(-sig, fs, f_range,
                                    center_extrema=extrema_opp,
                                    find_extrema_kwargs=find_extrema_kwargs)

    cols_peak = ['time_peak', 'time_rise', 'volt_rise',
                 'volt_amp', 'period', 'time_rdsym', 'time_ptsym']
    cols_trough = ['time_trough', 'time_decay', 'volt_decay',
                   'volt_amp', 'period', 'time_rdsym', 'time_ptsym']

    df_opp['time_rdsym'] = 1 - df_opp['time_rdsym']
    df_opp['time_ptsym'] = 1 - df_opp['time_ptsym']

    for idx, col in enumerate(cols_peak):

        if center_extrema == 'peak':
            np.testing.assert_allclose(df_shapes.loc[:, col], df_opp.loc[:, cols_trough[idx]])
        else:
            np.testing.assert_allclose(df_opp.loc[:, col], df_shapes.loc[:, cols_trough[idx]])


def test_compute_durations(sim_args):

    df_shapes = sim_args['df_shapes']
    period, time_peak, time_trough = compute_durations(df_shapes)

    assert ((time_trough + time_peak) == period).all()


def test_compute_extrema_voltage(sim_args):

    df_shapes = sim_args['df_shapes']
    sig = sim_args['sig']

    volt_peaks, volt_troughs = compute_extrema_voltage(df_shapes, sig)

    for idx, volt_peak in enumerate(volt_peaks):
        assert volt_peak > volt_troughs[idx]


@pytest.mark.parametrize("recompute", [True, False])
def test_compute_symmetry(sim_args, recompute):

    df_shapes = sim_args['df_shapes']
    sig = sim_args['sig']

    if recompute:
        sym_features = compute_symmetry(df_shapes, sig)
    else:
        period, time_peak, time_trough = compute_durations(df_shapes)
        sym_features = compute_symmetry(df_shapes, sig, period=period,
                                        time_peak=time_peak, time_trough=time_trough)

    # This is the case for only simulated sine waves
    assert (sym_features['time_decay'] == sym_features['time_rise']).all()
    assert (sym_features['volt_decay'] == sym_features['volt_rise']).all()

    assert (sym_features['volt_amp'] == \
        (sym_features['volt_rise'] + sym_features['volt_decay'])/2).all()

    np.testing.assert_almost_equal(sym_features['time_rdsym'].values,
                                   sym_features['time_ptsym'].values, decimal=1)


def test_compute_band_amp(sim_args):

    df_shapes = sim_args['df_shapes']
    sig = sim_args['sig']
    fs = sim_args['fs']
    f_range = sim_args['f_range']

    band_amp = compute_band_amp(df_shapes, sig, fs, f_range)

    for amp in band_amp[:-1]:
        np.testing.assert_allclose(amp, band_amp[0], rtol=1e-1)
