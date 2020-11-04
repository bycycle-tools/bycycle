"""Test functions to compute features across epoched data."""

from itertools import product
import numpy as np
import pandas as pd

from pytest import mark, param

from bycycle.group.features import compute_features_2d, compute_features_3d

###################################################################################################
###################################################################################################

@mark.parametrize("kwargs_dtype", ['dict', 'list_cycles', 'list_amp', None])
@mark.parametrize("axis", [0, None, param(1, marks=mark.xfail)])
def test_compute_features_2d(sim_args, kwargs_dtype, axis):

    n_sigs = 5
    sigs = np.array([sim_args['sig']] * n_sigs)
    fs  = sim_args['fs']
    f_range = sim_args['f_range']

    # Return_samples is disregarded when used in compute_features_kwargs
    compute_features_kwargs = {'center_extrema': 'peak', 'return_samples': False}

    # Update kwargs based on data type
    if kwargs_dtype == 'list_cycles' or kwargs_dtype == 'list_amp':

        compute_features_kwargs = [compute_features_kwargs] * n_sigs

        # Raise center extrema mistmatch warning
        compute_features_kwargs[1] = {'center_extrema': 'trough', 'return_samples': False}

        # Update kwargs to include burst_method
        burst_method = 'amp' if kwargs_dtype == 'list_amp' else 'cycles'

        compute_features_kwargs = [dict(kwargs, burst_method=burst_method) for kwargs in \
                compute_features_kwargs]

    elif kwargs_dtype == None:
         compute_features_kwargs = None

    # Sequential processing
    features_seq = compute_features_2d(sigs, fs, f_range, n_jobs=1, return_samples=True,
                                       compute_features_kwargs=compute_features_kwargs,
                                       axis=axis)

    # Parallel processing (assuming >1 job is available)
    if axis == 0:

        features_par = compute_features_2d(sigs, fs, f_range, n_jobs=-1, return_samples=True,
                                           compute_features_kwargs=compute_features_kwargs)

        # Compare sequential and parallel processing dfs
        for idx, df_par in enumerate(features_par):
            assert df_par.equals(features_seq[idx])

    if axis == None:

        # Asserts sequential dfs are equal (except first and last)
        for df_features in features_seq[2:-1]:

            pd.testing.assert_frame_equal(features_seq[1], df_features)

        # Edge artifacts will cause the first/last dataframes to differ when flattened
        assert not features_seq[0].equals(features_seq[-1])
        assert not features_seq[0].equals(features_seq[1])
        assert not features_seq[-1].equals(features_seq[1])

    else:

        # All dfs will be equal when computed independently
        for df_seq in features_seq[1:]:
            features_seq[0].equals(df_seq)


@mark.parametrize("return_samples", [True, False])
@mark.parametrize("axis", [0, 1, (0, 1), None, param(2, marks=mark.xfail)])
def test_compute_features_3d(sim_args, return_samples, axis):

    dim1 = 3
    dim2 = 2

    sigs_2d = np.array([sim_args['sig']] * dim2)
    sigs_3d = np.array([sigs_2d] * dim1)

    fs = sim_args['fs']
    f_range = sim_args['f_range']

    compute_features_kwargs = {'center_extrema': 'peak'}

    df_features = \
        compute_features_3d(sigs_3d, fs, f_range, compute_features_kwargs=compute_features_kwargs,
                            return_samples=return_samples, n_jobs=-1, progress=None, axis=axis)

    # Check lengths
    assert len(df_features) == dim1
    assert len(df_features[0]) == dim2

    # Check equal values, skipping the first and last dfs
    for row_idx, col_idx in list(product(range(0, dim1), range(0, dim2)))[2:-1]:
        pd.testing.assert_frame_equal(df_features[row_idx][col_idx], df_features[0][1])

