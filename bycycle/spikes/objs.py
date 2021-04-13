""""""

import numpy as np

from neurodsp.plts.utils import savefig

from bycycle.features import compute_features, compute_shape_features
from bycycle.features.burst import compute_amp_fraction, compute_monotonicity
from bycycle.plts import plot_burst_detect_summary, plot_cyclepoints_df

###################################################################################################
###################################################################################################

class Spike:
    """Compute bycycle features from a spike waveforms.

    Attributes
    ----------
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for the spike waveform.
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

    pad_factor : int
        Factor to scale the sampling rate (fs) by to pad on either side of the waveform.
    """

    def __init__(self, center_extrema='peak', pad_factor=2):
        """Initialize object settings."""

        self.center_extrema = center_extrema
        self.pad_factor = pad_factor

        # Attributes set in the fit method
        self.sig = None
        self.fs = None
        self.f_range = None
        self.df_features = None

    def fit(self, sig, fs, f_range):
        """Fit bycycle to a single spike.

        Parameters
        ----------
        sig : 1d array
            Voltage time series.
        fs : float
            Sampling rate, in Hz.
        f_range : tuple of (float, float)
            Frequency range for narrowband signal of interest (Hz).
        """

        # Set arguments as attributes
        self.sig = sig
        self.fs = fs
        self.f_range = f_range

        # Pad the signal
        sig_pad = np.pad(self.sig, int(self.fs * self.pad_factor))

        # Compute cyclepoints and shape features
        df_features = compute_shape_features(sig_pad, self.fs, self.f_range,
                                             center_extrema=self.center_extrema)

        # Trim and shift dataframe back to original signal length
        start = np.where(df_features['sample_' + self.center_extrema].values \
            >= (self.fs * self.pad_factor))[0][0]

        end = np.where(df_features['sample_' + self.center_extrema].values \
            < len(sig_pad) - (self.fs * self.pad_factor))[0][-1] + 1

        df_features = df_features.iloc[start:end]
        df_features = df_features.reset_index(drop=True)

        for key in df_features.keys().tolist():
            if key.startswith('sample_'):
                df_features[key] = df_features[key] - int(self.pad_factor * self.fs)

        # Additional burst features
        df_features['amp_fraction'] = compute_amp_fraction(df_features.copy())
        df_features['monotonicity'] = compute_monotonicity(df_features.copy(), self.sig)

        self.df_features = df_features

    @savefig
    def plot(self, xlim=None, figsize=(15, 3), plot_only_results=True, interp=True):
        """Plot cyclepoints.

        Parameters
        ----------
        xlim : tuple of (float, float), optional, default: None
            Start and stop times for plot.
        figsize : tuple of (float, float), optional, default: (15, 3)
            Size of each plot.
        plot_only_result : bool, optional, default: True
            Plot only the signal and bursts, excluding burst parameter plots.
        interp : bool, optional, default: True
            If True, interpolates between given values. Otherwise, plots in a step-wise fashion.
        """

        if self.df_features is None or self.sig is None or self.fs is None:
            raise ValueError('The fit method must be successfully called prior to plotting.')

        plot_cyclepoints_df(self.df_features, self.sig, self.fs, plot_sig=True, plot_extrema=True,
                            plot_zerox=True, xlim=None, ax=None, figsize=figsize)
