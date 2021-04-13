"""Class objects to compute features for spiking data."""

import numpy as np

from neurodsp.plts.utils import savefig

from bycycle import Bycycle
from bycycle.spikes.dataframes import slice_spikes
from bycycle.spikes.plts import plot_spike

from bycycle.features import compute_features, compute_shape_features
from bycycle.features.burst import compute_amp_fraction, compute_monotonicity
from bycycle.plts import plot_burst_detect_summary, plot_cyclepoints_df

###################################################################################################
###################################################################################################

class Spikes:
    """Compute spikes features.

    Attributes
    ----------
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for each spike.
    spikes : 2d array
        The signal associated with each spike (row in the ``df_features``).
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
        to change filter parameters or boundary. By default, the filter length is set to three
        cycles of the low cutoff frequency (``f_range[0]``).
    normalize : bool, optional, default: True
        Mean centers and variance normalizes when True.
    """

    def __init__(self, center_extrema='trough', find_extrema_kwargs=None, normalize=True):
        """Initialize object."""

        self.center_extrema = center_extrema

        if find_extrema_kwargs is None:
            self.find_extrema_kwargs = {'filter_kwargs': {'n_cycles': 3}}
        else:
            self.find_extrema_kwargs = find_extrema_kwargs

        self.normalize = True


    def fit(self, sig, fs, f_range):
        """Compute features for each spike.

        Parameters
        ----------
        sig : 1d array
            Voltage time series.
        fs : float
            Sampling rate, in Hz.
        f_range : tuple of (float, float)
            Frequency range for narrowband signal of interest (Hz).
        """

        # Set attibutes
        self.sig = sig
        self.fs = fs
        self.f_range = f_range

        # Initial fit
        bm = Bycycle(center_extrema=self.center_extrema,
                     find_extrema_kwargs=self.find_extrema_kwargs)
        bm.fit(self.sig, self.fs, self.f_range)

        # Isolate spikes
        df_features, spikes = slice_spikes(bm, std=2)

        self.df_features = df_features
        self.spikes = spikes

        # Mean and varaince normalize
        if self.normalize:
            self.normalize_spikes()


    def normalize_spikes(self):
        """Mean and variance normalize spikes."""

        if self.df_features is None or self.sig is None or self.fs is None:
            raise ValueError('The fit method must be successfully called prior to plotting.')

        # Mean and variance normalize
        mean_ = np.nanmean(self.spikes, axis=1)
        scale_ = np.nanstd(self.spikes, axis=1)
        scale_[np.where(scale_ == 0)[0]] = 1

        sig_roll = np.rollaxis(self.spikes, axis=1)
        sig_roll -= mean_
        sig_roll /= scale_


    @savefig
    def plot(self, index=None, ax=None):
        """Plot spike results.

        Parameters
        ----------
        index : int, optional, default: None
            The index in ``spikes`` and ``df_features`` to plot. If none, plot all spikes.
        ax : matplotlib.Axes, optional, default: None
            Figure axes upon which to plot.
        """

        if self.df_features is None or self.sig is None or self.fs is None:
            raise ValueError('The fit method must be successfully called prior to plotting.')

        # Plot an individual spike or a spike summary
        plot_spike(self.fs, self.spikes, self.df_features, index=index, ax=ax)
