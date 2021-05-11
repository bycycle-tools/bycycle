"""Tests for gaussian features."""

import pytest
import numpy as np

from bycycle.spikes.features.gaussians import (
    compute_gaussian_features, _compute_gaussian_features, estimate_params, _estimate_bounds,
    _fit_gaussians
)
from bycycle.spikes.cyclepoints import compute_spike_cyclepoints

###################################################################################################
###################################################################################################


def test_compute_gaussian_features(sim_spikes, sim_spikes_df):

    sig = sim_spikes['sig']
    fs = sim_spikes['fs']
    df_samples = sim_spikes_df['df_samples']

    params = compute_gaussian_features(df_samples, sig, fs, n_gaussians=3,
                                       maxfev=2000, tol=1e-3, n_jobs=-1, progress=None)

    assert len(params) == len(df_samples)

    for param in params:
        assert len(param) == 15


@pytest.mark.parametrize("sig_key", ['sig', 'sig_na', 'sig_na_k', 'sig_na_cond'])
@pytest.mark.parametrize("n_gaussians", [2, 3])
def test__compute_gaussian_features(sim_spikes, sig_key, n_gaussians):

    sig = sim_spikes[sig_key]

    fs = 20000
    f_range = (500, 3000)

    df_samples = compute_spike_cyclepoints(sig, fs, f_range, std=2)

    params = _compute_gaussian_features(0, df_samples, sig, fs, tol=1e-3, n_gaussians=n_gaussians)

    assert len(params) == (n_gaussians * 4) + 3


def test_fit_gaussians(sim_spikes, sim_spikes_df):

    sig = sim_spikes['sig']
    fs = sim_spikes['fs']
    df_samples = sim_spikes_df['df_samples']

    start = df_samples.iloc[0]['sample_start'].astype(int)
    end = df_samples.iloc[0]['sample_end'].astype(int)

    ys = sig[start:end+1]
    xs = np.arange(0, len(ys)/fs, 1/fs)

    guess = estimate_params(df_samples, sig, fs, 0)
    bounds = _estimate_bounds(ys, *guess[:-3].reshape(4, -1)[[0, 1, 3]])

    params = _fit_gaussians(xs, ys, guess, bounds, 1e-3, 2000, 0)

    # Failed fit  (bounds error)
    params = _fit_gaussians(xs, ys, guess, (bounds[1], bounds[0]), 1e-3, 2000, 0)



