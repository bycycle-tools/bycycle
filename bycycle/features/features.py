"""Quantify the shape of oscillatory waveforms on a cycle-by-cycle basis."""

import pandas as pd

from bycycle.features.shape import compute_shape_features
from bycycle.features.burst import compute_burst_features

###################################################################################################
###################################################################################################


def compute_features(sig, fs, f_range, center_extrema='peak', find_extrema_kwargs=None,
                     hilbert_increase_n=False, return_samples=True, dual_threshold_kwargs=None):
    """Compute shape and burst features for each cycle.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest (Hz).
    center_extrema : {'peak', 'trough'}
        The center extrema in the cycle.

        - 'peak' : cycles are defined trough-to-trough
        - 'trough' : cycles are defined peak-to-peak

    find_extrema_kwargs : dict, optional, default: None
        Keyword arguments for function to find peaks an troughs (:func:`~.find_extrema`)
        to change filter Parameters or boundary. By default, it sets the filter length to three
        cycles of the low cutoff frequency (``f_range[0]``).
    hilbert_increase_n : bool, optional, default: False
        Corresponding kwarg for :func:`~neurodsp.timefrequency.hilbert.amp_by_time` for determining
        ``band_amp``. If true, this zero-pads the signal when computing the Fourier transform, which
        can be necessary for computing it in a reasonable amount of time.
    return_samples : bool, optional, default: True
        Returns samples indices of cyclepoints used for determining features if True.
    dual_threshold_kwargs : dict, optional, default: None
        Additional arguments in :func:`~.compute_burst_fraction`.

    Returns
    -------
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for each cycle. Columns:

        - ``period`` : period of the cycle
        - ``time_decay`` : time between peak and next trough
        - ``time_rise`` : time between peak and previous trough
        - ``time_peak`` : time between rise and decay zero-crosses
        - ``time_trough`` : duration of previous trough estimated by zero-crossings
        - ``volt_decay`` : voltage change between peak and next trough
        - ``volt_rise`` : voltage change between peak and previous trough
        - ``volt_amp`` : average of rise and decay voltage
        - ``volt_peak`` : voltage at the peak
        - ``volt_trough`` : voltage at the last trough
        - ``time_rdsym`` : fraction of cycle in the rise period
        - ``time_ptsym`` : fraction of cycle in the peak period
        - ``band_amp`` : average analytic amplitude of the oscillation computed using narrowband
          filtering and the Hilbert transform. Filter length is 3 cycles of the low cutoff
          frequency. Average taken across all time points in the cycle.

        When consistency burst detection is used (i.e. dual_threshold_kwargs is None):

        - ``amplitude_fraction`` : normalized amplitude
        - ``amplitude_consistency`` : difference in the rise and decay voltage within a cycle
        - ``period_consistency`` : difference between a cycle’s period and the period of the
          adjacent cycles
        - ``monotonicity`` : fraction of instantaneous voltage changes between consecutive
          samples that are positive during the rise phase and negative during the decay phase

        When dual threshold burst detection is used (i.e. dual_threshold_kwargs is not None):

        - ``burst_fraction`` : fraction of a cycle that is bursting

    df_samples : pandas.DataFrame, optional, default: True
        An optionally returned dataframe containing cyclepoints for each cycle.
        Columns (listed for peak-centered cycles):

        - ``sample_peak`` : sample of 'sig' at which the peak occurs
        - ``sample_zerox_decay`` : sample of the decaying zero-crossing
        - ``sample_zerox_rise`` : sample of the rising zero-crossing
        - ``sample_last_trough`` : sample of the last trough
        - ``sample_next_trough`` : sample of the next trough

    Notes
    -----
    If ``dual_threshold_kwargs`` is *not* None, dual amplitude threshold burst detection will be
    used, rather than cycle feature consistency.
    """

    # Compute shape features for each cycle.
    df_shape_features, df_samples = \
        compute_shape_features(sig, fs, f_range, center_extrema=center_extrema,
                               find_extrema_kwargs=find_extrema_kwargs,
                               hilbert_increase_n=hilbert_increase_n)

    # Compute burst features for each cycle.
    df_burst_features = compute_burst_features(df_shape_features, df_samples, sig,
                                               dual_threshold_kwargs=dual_threshold_kwargs)

    # Concatenate shape and burst features
    df_features = pd.concat((df_shape_features, df_burst_features), axis=1)

    if return_samples:

        return df_features, df_samples

    return df_features
