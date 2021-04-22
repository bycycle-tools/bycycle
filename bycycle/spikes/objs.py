"""Class objects to compute features for spiking data."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from neurodsp.plts.utils import savefig

from bycycle import Bycycle

from bycycle.spikes.features import compute_shape_features, compute_gaussian_features
from bycycle.spikes.cyclepoints import compute_spike_cyclepoints
from bycycle.spikes.plts import plot_spikes
from bycycle.spikes.utils import split_signal, rename_df

###################################################################################################
###################################################################################################

class Spikes:
    """Compute spikes features.

    Attributes
    ----------
    df_features : pandas.DataFrame
        Dataframe containing shape and burst features for each spike.
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
    std : float or int, optional, default: 1.5
            The standard deviation used to identify spikes.
    """

    def __init__(self, center_extrema='trough', find_extrema_kwargs=None):
        """Initialize object."""

        # Settings
        self.center_extrema = center_extrema

        if find_extrema_kwargs is None:
            self.find_extrema_kwargs = {'filter_kwargs': {'n_cycles': 3}}
        else:
            self.find_extrema_kwargs = find_extrema_kwargs

        # Fit params
        self.sig = None
        self.fs = None
        self.f_range = None
        self.std = None
        self.prune = None

        # Results
        self.df_features = None
        self.spikes = []


    def __len__(self):
        """Define the length of the object."""

        return len(self.spikes)


    def __iter__(self):
        """Allow for iterating across the object."""

        for spike in self.spikes:
            yield spike[~np.isnan(spike)]


    def __getitem__(self, index):
        """Allow for indexing into the object."""

        return self.spikes[index][~np.isnan(self.spikes[index])]


    def fit(self, sig, fs, f_range, std=1.5, prune=False,
            gaussians=True, maxfev=2000, n_jobs=-1, progress=None):
        """Compute features for each spike.

        Parameters
        ----------
        sig : 1d array
            Voltage time series.
        fs : float
            Sampling rate, in Hz.
        f_range : tuple of (float, float)
            Frequency range for narrowband signal of interest (Hz).
        std : float or int, optional, default: 1.5
            The standard deviation used to identify spikes.
        prune : bool, optional, default: False
            Remove spikes with high variablility in non-trough peaks.
        gaussians : bool, optional, default: True
            Fit a double gaussian model to each spike when True.
        maxfev : int, optional, default: 2000
            The maximum number of calls in curve_fit.
            Only used when gaussians is True.
        n_jobs : int, optional, default: -1
            The number of jobs to compute features in parallel.
            Only used when gaussians is True.
        progress : {None, 'tqdm', 'tqdm.notebook'}
            Specify whether to display a progress bar. Uses 'tqdm', if installed.
            Only used when gaussians is True.
        """

        # Set attibutes
        self.sig = sig
        self.fs = fs
        self.f_range = f_range
        self.std = std
        self.prune = prune

        # Cyclepoints
        if self.center_extrema == 'trough':
            df_features = compute_spike_cyclepoints(self.sig, self.fs, self.f_range,
                                                    self.std, self.prune)
        else:
            df_features = compute_spike_cyclepoints(-self.sig, self.fs, self.f_range,
                                                    self.std, self.prune)


        # Isolate spikes
        spikes = split_signal(df_features, self.sig)

        self.spikes = spikes

        # Compute shape features
        df_shape_features = compute_shape_features(df_features, self.sig)

        df_features = pd.concat((df_features, df_shape_features), axis=1)

        # Compute gaussian features
        if self.center_extrema == 'trough':
            params, r_squared = compute_gaussian_features(df_features, self.sig, self.fs,
                                                          maxfev=maxfev, n_jobs=-1, progress=None)
        else:
            params, r_squared = compute_gaussian_features(df_features, -self.sig, self.fs,
                                                          maxfev=maxfev, n_jobs=-1, progress=None)

        df_gaussian_features = pd.DataFrame.from_dict(params)
        df_gaussian_features['r_squared'] = r_squared

        # Merge dataframes
        self.df_features = pd.concat((df_features, df_gaussian_features), axis=1)

        # Rename dataframe
        if self.center_extrema == 'peak':

            # Rename columns
            self.df_features = rename_df(self.df_features)


    def normalize(self, inplace=True):
        """Mean and variance normalize spikes.

        Parameters
        ----------
        inplace : bool, optional, default: True
            Modifies the sig attibute in place if True. If false, return a normalized signal.
        """

        if self.spikes is None:
            raise ValueError('The fit method must be successfully called prior to normalization.')

        # Mean and variance normalize
        mean_ = np.nanmean(self.spikes, axis=1)
        scale_ = np.nanstd(self.spikes, axis=1)
        scale_[np.where(scale_ == 0)[0]] = 1

        sig_roll = np.rollaxis(self.spikes.copy(), axis=1)

        sig_roll -= mean_
        sig_roll /= scale_

        if inplace:
            self.spikes = sig_roll.transpose()
        else:
            return sig_roll.transpose()


    @savefig
    def plot(self, stack=True, index=None, normalize=False, xlim=None, ax=None):
        """Plot spike results.

        Parameters
        ----------
        stack : bool, optional, default: True
            Plots spikes as 2d arrays ontop of one another. Ignored if index is not None.
        index : int, optional, default: None
            The index in ``df_features`` to plot. If None, plot all spikes.
        normalize : bool, optional, default: True
            Mean centers and variance normalizes spikes attribute when True.
        xlim : tuple
            Upper and lower time limits. Ignored if ``stack`` is True or ``index`` is passed.
        ax : matplotlib.Axes, optional, default: None
            Figure axes upon which to plot.
        """

        if self.df_features is None or self.sig is None or self.fs is None:
            raise ValueError('The fit method must be successfully called prior to plotting.')

        # Plot an individual spike or a spike summary
        if normalize:
            spikes = self.normalize(inplace=False)
        else:
            spikes = self.spikes.copy()

        if stack:
            plot_spikes(self.df_features, self.sig, self.fs, spikes, index, xlim, ax)
        else:
            plot_spikes(self.df_features, self.sig, self.fs, None, index, xlim, ax)


    @savefig
    def plot_gaussian_params(self):
        """Plot gaussian parameters distributions."""

        if self.df_features is None or 'center0' not in self.df_features:
            raise ValueError('The fit method must be successfully called, '
                             'with gaussian=True, prior to plotting.')

        # Setup figure and axes
        fig = plt.figure(figsize=(10, 18))

        gs = GridSpec(6, 2, figure=fig)

        axes_idxs = [(row, col) for row in range(5) for col in range(2)]

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])
        ax6 = fig.add_subplot(gs[3, 0])
        ax7 = fig.add_subplot(gs[3, 1])
        ax8 = fig.add_subplot(gs[4, 0])
        ax9 = fig.add_subplot(gs[4, 1])
        ax10 = fig.add_subplot(gs[5, :])

        # Labels
        titles = ['center0', 'center1', 'std0', 'std1', 'alpha0',
                'alpha1', 'height0', 'height1', 'shift0', 'shift1']

        xlabels = ['mv (normalized)', 'mv (normalized)',
                'Skew Coefficient', 'mV', 'mV']

        xlabels = [l for lab in xlabels for l in [lab, lab]]

        # Plot
        axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
        for idx in range(10):

            axes[idx].hist(self.df_features[titles[idx]].values)
            axes[idx].set_title(titles[idx])
            axes[idx].set_xlabel(xlabels[idx])

        ax10.hist(self.df_features['r_squared'].values)
        ax10.set_title('R-Squared')

        # Increase spacing
        fig.subplots_adjust(hspace=.5)

        plt.show()
