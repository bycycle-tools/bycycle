"""Functions to determine the shape features, and the locations of peaks, troughs, and
zero-crossings (rise and decay) for individual cycles.
"""

import numpy as np
import pandas as pd

from neurodsp.timefrequency import amp_by_time

from bycycle.cyclepoints import find_extrema, find_zerox

###################################################################################################
###################################################################################################


def compute_shape_features(sig, fs, f_range, center_extrema='peak', find_extrema_kwargs=None,
                           hilbert_increase_n=False, return_samples=True):
    """Compute shapes parameters of each cycle, used for determining burst features.

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
        Corresponding kwarg for :func:`~neurodsp.timefrequency.hilbert.amp_by_time`.
        If true, this zero-pads the signal when computing the Fourier transform, which can be
        necessary for computing it in a reasonable amount of time.
    return_samples : bool, optional, default: True
        Returns samples indices of cyclepoints used for determining features if True.

    Returns
    -------
    df_shapes : pandas.DataFrame
        Dataframe containing cycle shape features. Each row is one cycle. Columns:

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

    df_samples : pandas.DataFrame, optional, default: True
        Dataframe containing sample indices of cyclepoints.
        Columns (listed for peak-centered cycles):

        - ``sample_peak`` : sample of 'sig' at which the peak occurs
        - ``sample_zerox_decay`` : sample of the decaying zero-crossing
        - ``sample_zerox_rise`` : sample of the rising zero-crossing
        - ``sample_last_trough`` : sample of the last trough
        - ``sample_next_trough`` : sample of the next trough

    Notes
    -----
    Peak vs trough centering

        - By default, the first extrema analyzed will be a peak, and the final one a trough.
        - In order to switch the preference, the signal is simply inverted and columns are renamed.
        - Columns are slightly different depending on if ``center_extrema`` is set to 'peak' or
          'trough'.

    """

    # Set defaults if user input is None
    if find_extrema_kwargs is None:
        find_extrema_kwargs = {'filter_kwargs': {'n_cycles': 3}}

    elif 'first_extrema' in find_extrema_kwargs.keys():
        raise ValueError('''This function has been designed to assume that the first extrema
            identified will be a peak. This cannot be overwritten at this time.''')

    # Negate signal if to analyze trough-centered cycles
    if center_extrema == 'peak':
        pass
    elif center_extrema == 'trough':
        sig = -sig
    else:
        raise ValueError('Parameter "center_extrema" must be either "P" or "T"')

    # Find extrema and zero-crossings locations in the signal
    ps, ts = find_extrema(sig, fs, f_range, **find_extrema_kwargs)
    rises, decays = find_zerox(sig, ps, ts)

    # For each cycle, identify the sample of each extrema and zero-crossing
    df_samples = compute_samples(ps, ts, decays, rises)

    # Compute duration of period
    shape_features = {}
    shape_features['period'] = df_samples['sample_next_trough'] - df_samples['sample_last_trough']

    # Compute duration of peak
    shape_features['time_peak'] = df_samples['sample_zerox_decay'] - df_samples['sample_zerox_rise']

    # Compute duration of last trough
    shape_features['time_trough'] = rises - decays[:-1]

    # Determine extrema voltage
    shape_features['volt_peak'] = sig[ps[1:]]
    shape_features['volt_trough'] = sig[ts[:-1]]

    # Determine rise and decay characteristics
    shape_features['time_decay'] = (ts[1:] - ps[1:])
    shape_features['time_rise'] = (ps[1:] - ts[:-1])

    shape_features['volt_decay'] = sig[ps[1:]] - sig[ts[1:]]
    shape_features['volt_rise'] = sig[ps[1:]] - sig[ts[:-1]]
    shape_features['volt_amp'] = (shape_features['volt_decay'] + shape_features['volt_rise']) / 2

    # Compute rise-decay symmetry features
    shape_features['time_rdsym'] = shape_features['time_rise'] / shape_features['period']

    # Compute peak-trough symmetry features
    shape_features['time_ptsym'] = shape_features['time_peak'] / \
            (shape_features['time_peak'] + shape_features['time_trough'])

    # Compute average oscillatory amplitude estimate during cycle
    amp = amp_by_time(sig, fs, f_range, hilbert_increase_n=hilbert_increase_n, n_cycles=3)

    shape_features['band_amp'] = [np.mean(amp[ts[sig_idx]:ts[sig_idx + 1]]) for sig_idx in
                                  range(len(df_samples['sample_peak']))]

    # Convert feature dictionary into a DataFrame
    df_features= pd.DataFrame.from_dict(shape_features)

    # Rename columns if they are actually trough-centered
    if center_extrema == 'trough':

        samples_rename_dict = {'sample_peak': 'sample_trough',
                               'sample_zerox_decay': 'sample_zerox_rise',
                               'sample_zerox_rise': 'sample_zerox_decay',
                               'sample_last_trough': 'sample_last_peak',
                               'sample_next_trough': 'sample_next_peak'}

        features_rename_dict = {'time_peak': 'time_trough',
                                'time_trough': 'time_peak',
                                'volt_peak': 'volt_trough',
                                'volt_trough': 'volt_peak',
                                'time_rise': 'time_decay',
                                'time_decay': 'time_rise',
                                'volt_rise': 'volt_decay',
                                'volt_decay': 'volt_rise'}

        df_samples.rename(columns=samples_rename_dict, inplace=True)
        df_features.rename(columns=features_rename_dict, inplace=True)

        # Need to reverse symmetry measures
        df_features['volt_peak'] = -df_features['volt_peak']
        df_features['volt_trough'] = -df_features['volt_trough']
        df_features['time_rdsym'] = 1 - df_features['time_rdsym']
        df_features['time_ptsym'] = 1 - df_features['time_ptsym']

    if return_samples:

        return df_features, df_samples

    return df_features


def compute_samples(ps, ts, decays, rises):
    """Compute sample indices for cyclepoints.

    Parameters
    ----------
    ps : 1d array
        Signal indices of oscillatory peaks.
    ts : 1d array
        Signal indices of oscillatory troughs.
    rises : 1d array, optional
        Signal indices of oscillatory rising zero-crossings.
    decays : 1d array, optional
        Signal indices of oscillatory decaying zero-crossings.

    Returns
    -------
    df_samples : pandas.DataFrame, optional, default: False
        Dataframe containing sample indices of cyclepoints.
        Columns (listed for peak-centered cycles):

        - ``sample_peak`` : sample of 'sig' at which the peak occurs
        - ``sample_zerox_decay`` : sample of the decaying zero-crossing
        - ``sample_zerox_rise`` : sample of the rising zero-crossing
        - ``sample_last_trough`` : sample of the last trough
        - ``sample_next_trough`` : sample of the next trough

    """

    # For each cycle, identify the sample of each extrema and zero-crossing
    samples = {}
    samples['sample_peak'] = ps[1:]
    samples['sample_zerox_decay'] = decays[1:]
    samples['sample_zerox_rise'] = rises
    samples['sample_last_trough'] = ts[:-1]
    samples['sample_next_trough'] = ts[1:]

    df_samples = pd.DataFrame.from_dict(samples)

    return df_samples
