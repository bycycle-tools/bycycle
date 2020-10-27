"""Tests for utils.dataframe."""

from copy import deepcopy
import numpy as np
import pandas as pd

from bycycle.utils.dataframes import *

###################################################################################################
###################################################################################################

def test_limit_df(sim_args):

    df_features = sim_args['df_features']
    fs = sim_args['fs']

    xlim = (1, 2)

    df_short = limit_df(df_features, fs, start=xlim[0], stop=xlim[1])

    assert df_short['sample_next_trough'].min() >= 0
    assert df_short['sample_last_trough'].max() <= fs * (xlim[1] - xlim[0])


def test_get_extrema_df(sim_args):

    df_features = sim_args['df_features']
    center_e, side_e = get_extrema_df(df_features)

    # The fixture will return peak centered cycles
    assert center_e == 'peak'
    assert side_e == 'trough'

    df_features = pd.DataFrame({'sample_trough': []})
    center_e, side_e = get_extrema_df(df_features)

    assert center_e == 'trough'
    assert side_e == 'peak'


def test_rename_extrema_df(sim_args):

    df_features = sim_args['df_features']

    # Remove column names that will not change based on extrema center
    same_cols = ['amp_fraction', 'amp_consistency', 'period_consistency',
                 'monotonicity', 'band_amp', 'period', 'volt_amp',
                 'is_burst']

    df_features = df_features.drop(same_cols, axis=1)

    # Rename columns
    df_features_renamed = rename_extrema_df('trough', deepcopy(df_features), return_samples=True)

    # These columns names don't change, but their values do
    diff_cols = ['time_rdsym', 'time_ptsym']

    for diff_col in diff_cols:

        assert (df_features_renamed.pop(diff_col).values == \
            df_features.pop(diff_col).values).all()

    assert (df_features.columns.values != df_features_renamed.columns.values).all()


def test_split_samples_df(sim_args):

    df_features = sim_args['df_features']

    df_features, df_samples = split_samples_df(df_features)

    # Ensure sample columns are isolated to df_samples
    for col in df_features.columns:
        assert "sample_" not in col

    for col in df_samples.columns:
        assert "sample_" in col
