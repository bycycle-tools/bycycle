"""Bycycle class objects."""

import warnings
import numpy as np

from neurodsp.plts.utils import savefig
import pandas as pd

from bycycle.features import compute_features
from bycycle.group import compute_features_2d, compute_features_3d
from bycycle.plts import plot_burst_detect_summary
from bycycle.burst.utils import recompute_edges as rc_edges

###################################################################################################
###################################################################################################

class BycycleBase:
    """Shared base sub-class."""

    def __init__(self, center_extrema='peak', burst_method='cycles', burst_kwargs=None,
                 thresholds=None, find_extrema_kwargs=None, return_samples=True):

        # Compute features settings
        self.center_extrema = center_extrema

        self.burst_method = burst_method
        self.burst_kwargs = {} if burst_kwargs is None else burst_kwargs

        # Thresholds
        if not isinstance(thresholds, dict):
            warnings.warn("""
                No burst detection thresholds are provided. This is not recommended. Please
                inspect your data and choose appropriate parameters for 'thresholds'.
                Default burst detection parameters are likely not well suited for your
                desired application.
                """)

        if thresholds is None and burst_method == 'cycles':
            self.thresholds = {
                'amp_fraction_threshold': 0.,
                'amp_consistency_threshold': .5,
                'period_consistency_threshold': .5,
                'monotonicity_threshold': .8,
                'min_n_cycles': 3
            }
        elif thresholds is None and burst_method == 'amp':
            self.thresholds = {
                'burst_fraction_threshold': 1,
                'min_n_cycles': 3
            }
        else:
            self.thresholds = thresholds

        # Allow shorthand (e.g. monotonicicy instead of monotonicity_threshold)
        if isinstance(self.thresholds, dict):
            for k in list(self.thresholds.keys()):
                if not k.endswith('_threshold') and k != 'min_n_cycles':
                    self.thresholds[k + '_threshold'] = self.thresholds.pop(k)

        if find_extrema_kwargs is None:
            self.find_extrema_kwargs = {'filter_kwargs': {'n_cycles': 3}}
        else:
            self.find_extrema_kwargs = find_extrema_kwargs

        self.return_samples = return_samples

        # Compute features args
        self.sig = None
        self.fs = None
        self.f_range = None

        # Results
        self.df_features = None

    def reduce_thresholds(self, reduction):
        """Adjust thresholds by a given amount.

        Parameters
        ----------
        reduction : float, optional, default: None
            Reduces all float thresholds by given amount.

        Returns
        -------
        reduced_thresholds : dict
            Copy of thresholds with reduction applied.
        """
        reduction = 0 if reduction is None else reduction
        reduced_thresholds = {}

        for k, v in self.thresholds.items():
            if k.endswith('threshold'):
                reduced_thresholds[k] = v - reduction
            else:
                reduced_thresholds[k] = v

        return reduced_thresholds


class Bycycle(BycycleBase):
    """Compute bycycle features from a signal.

    Attributes
    ----------
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for each cycle.
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

    burst_method : {'cycles', 'amp'}
        Method for detecting bursts.

        - 'cycles': detect bursts based on the consistency of consecutive periods & amplitudes
        - 'amp': detect bursts using an amplitude threshold

    burst_kwargs : dict, optional, default: None
        Additional keyword arguments defined in :func:`~.compute_burst_fraction` for dual
        amplitude threshold burst detection (i.e. when burst_method='amp').
    threshold_kwargs : dict, optional, default: None
        Feature thresholds for cycles to be considered bursts, matching keyword arguments for:

        - :func:`~.detect_bursts_cycles` for consistency burst detection
          (i.e. when burst_method='cycles')
        - :func:`~.detect_bursts_amp` for  amplitude threshold burst detection
          (i.e. when burst_method='amp').

    find_extrema_kwargs : dict, optional, default: None
        Keyword arguments for function to find peaks an troughs (:func:`~.find_extrema`)
        to change filter parameters or boundary. By default, the filter length is set to three
        cycles of the low cutoff frequency (``f_range[0]``).
    return_samples : bool, optional, default: True
        Returns samples indices of cyclepoints used for determining features if True.
    """

    def __init__(self, center_extrema='peak', burst_method='cycles', burst_kwargs=None,
                 thresholds=None, find_extrema_kwargs=None, return_samples=True):
        """Initialize object settings."""

        super().__init__(center_extrema, burst_method, burst_kwargs, thresholds,
                         find_extrema_kwargs, return_samples)


    def __getattr__(self, key):
        """Access df_features columns as class attributes.

        Parameters
        ----------
        key : str
            Column name.

        Returns
        -------
        1d-array
            Column values.
        """

        if key in {'__getstate__', '__setstate__'}:
            return object.__getattr__(self, key)
        elif (self.df_features is not None and key in self.df_features.keys()):
            return self.df_features[key].values
        else:
            raise AttributeError(f'\'{self.__class__.__name__}\' object has no attribute \'{key}\'')


    def fit(self, sig, fs, f_range):
        """Run the bycycle algorithm on a signal.

        Parameters
        ----------
        sig : 1d array
            Time series.
        fs : float
            Sampling rate, in Hz.
        f_range : tuple of (float, float)
            Frequency range for narrowband signal of interest (Hz).
        """

        if sig.ndim != 1:
            raise ValueError('Signal must be 1-dimensional.')

        # Add settings as attributes
        self.sig = sig
        self.fs = fs
        self.f_range = f_range
        self.df_features = compute_features(
            self.sig, self.fs, self.f_range, self.center_extrema,
            self.burst_method, self.burst_kwargs, self.thresholds,
            self.find_extrema_kwargs, self.return_samples
        )


    def recompute_edges(self, reduction=None):
        """Recomputes features for cycles on the edge of bursts.

        Parameters
        ----------
        reduction : float, optional, default: None
            Reduces all float thresholds by given amount.
        """
        reduced_thresholds = self.reduce_thresholds(reduction)
        self.df_features = rc_edges(self.df_features, reduced_thresholds)


    @savefig
    def plot(self, xlim=None, figsize=(15, 3), plot_only_results=False, interp=True):
        """Plot burst detection results.

        Parameters
        ----------
        xlim : tuple of (float, float), optional, default: None
            Start and stop times for plot.
        figsize : tuple of (float, float), optional, default: (15, 3)
            Size of each plot.
        plot_only_result : bool, optional, default: False
            Plot only the signal and bursts, excluding burst parameter plots.
        interp : bool, optional, default: True
            If True, interpolates between given values. Otherwise, plots in a step-wise fashion.
        """

        if self.df_features is None or self.sig is None or self.fs is None:
            raise ValueError('The fit method must be successfully called prior to plotting.')

        plot_burst_detect_summary(self.df_features, self.sig, self.fs, self.thresholds,
                                  xlim, figsize, plot_only_results, interp)


    def load(self, df_features, sig, fs, f_range):
        """Load external results."""

        self.sig = sig
        self.fs = fs
        self.f_range = f_range
        self.df_features = df_features


class BycycleGroup(BycycleBase):
    """Compute bycycle features for a 2d or 3d signal.

    Attributes
    ----------
    models : list of Bycycle
        Fit Bycycle objects.
    sigs : 2d or 3d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest (Hz).
    center_extrema : {'peak', 'trough'}
        The center extrema in the cycle.

        - 'peak' : cycles are defined trough-to-trough
        - 'trough' : cycles are defined peak-to-peak

    burst_method : {'cycles', 'amp'}
        Method for detecting bursts.

        - 'cycles': detect bursts based on the consistency of consecutive periods & amplitudes
        - 'amp': detect bursts using an amplitude threshold

    burst_kwargs : dict, optional, default: None
        Additional keyword arguments defined in :func:`~.compute_burst_fraction` for dual
        amplitude threshold burst detection (i.e. when burst_method='amp').
    threshold_kwargs : dict, optional, default: None
        Feature thresholds for cycles to be considered bursts, matching keyword arguments for:

        - :func:`~.detect_bursts_cycles` for consistency burst detection
          (i.e. when burst_method='cycles')
        - :func:`~.detect_bursts_amp` for  amplitude threshold burst detection
          (i.e. when burst_method='amp').

    find_extrema_kwargs : dict, optional, default: None
        Keyword arguments for function to find peaks an troughs (:func:`~.find_extrema`)
        to change filter parameters or boundary. By default, the filter length is set to three
        cycles of the low cutoff frequency (``f_range[0]``).
    axis : {0, 1, (0, 1), None}
        For 2d arrays:

        - ``axis=0`` : Iterates over each row/signal in an array independently (i.e. for each
        channel in (n_channels, n_timepoints)).
        - ``axis=None`` : Flattens rows/signals prior to computing features (i.e. across flatten
        epochs in (n_epochs, n_timepoints)).

        For 3d arrays:

        - ``axis=0`` : Iterates over 2D slices along the zeroth dimension, (i.e. for each
        channel in (n_channels, n_epochs, n_timepoints)).
        - ``axis=1`` : Iterates over 2D slices along the first dimension (i.e. across flatten
        epochs in (n_epochs, n_channels, n_timepoints)).
        - ``axis=(0, 1)`` : Iterates over 1D slices along the zeroth and first dimensions (i.e
        across each signal independently in (n_participants, n_channels, n_timepoints)).

    return_samples : bool, optional, default: True
        Returns samples indices of cyclepoints used for determining features if True.

    """
    def __init__(self, center_extrema='peak', burst_method='cycles', burst_kwargs=None,
                 thresholds=None, find_extrema_kwargs=None, return_samples=True):
        """Initialize object settings."""
        super().__init__(center_extrema, burst_method, burst_kwargs, thresholds,
                         find_extrema_kwargs, return_samples)

        # 2d settings
        self.axis = None
        self.n_jobs = None
        self.n_dims = None

        # Results
        self.models = []


    def __len__(self):
        """Define the length of the object."""

        return len(self.models)


    def __iter__(self):
        """Allow for iterating across the object."""

        for result in self.models:
            yield result


    def __getitem__(self, index):
        """Allow for indexing into the object."""

        return self.models[index]


    def fit(self, sigs, fs, f_range, axis=0, n_jobs=-1, progress=None):
        """Run the bycycle algorithm on a 2D or 3D array of signals.

        Parameters
        ----------
        sigs : 3d array
            Voltage time series, with 2d or 3d shape.
        fs : float
            Sampling rate, in Hz.
        f_range : tuple of (float, float)
            Frequency range for narrowband signal of interest, in Hz.
        recompute_edges : bool, optional, default: False
            Recomputes features for cycles on the edge of bursts.
        axis : {0, 1, (0, 1), None}
            For 2d arrays:


            - ``axis=0`` : Iterates over each row/signal in an array independently (i.e. for each
            channel in (n_channels, n_timepoints)).
            - ``axis=None`` : Flattens rows/signals prior to computing features (i.e. across flatten
            epochs in (n_epochs, n_timepoints)).

            For 3d arrays:

            - ``axis=0`` : Iterates over 2D slices along the zeroth dimension, (i.e. for each
            channel in (n_channels, n_epochs, n_timepoints)).
            - ``axis=1`` : Iterates over 2D slices along the first dimension (i.e. across flatten
            epochs in (n_epochs, n_channels, n_timepoints)).
            - ``axis=(0, 1)`` : Iterates over 1D slices along the zeroth and first dimensions (i.e
            across each signal independently in (n_participants, n_channels, n_timepoints)).

        n_jobs : int, optional, default: -1
            The number of jobs to compute features in parallel.
        progress : {None, 'tqdm', 'tqdm.notebook'}
            Specify whether to display a progress bar. Uses 'tqdm', if installed.
        """

        if sigs.ndim not in (2, 3):
            raise ValueError('Signal must be 2 or 3-dimensional.')

        self.sigs = sigs
        self.fs = fs
        self.f_range = f_range
        self.axis = axis
        self.n_jobs = n_jobs

        compute_features_kwargs = {
            'center_extrema': self.center_extrema,
            'burst_method': self.burst_method,
            'burst_kwargs': self.burst_kwargs,
            'threshold_kwargs': self.thresholds,
            'find_extrema_kwargs': self.find_extrema_kwargs
        }

        compute_func = compute_features_2d if self.sigs.ndim == 2 else compute_features_3d

        self.df_features = compute_func(
            self.sigs, self.fs, self.f_range, compute_features_kwargs,
            self.axis, self.return_samples, self.n_jobs, progress
        )

        # Initialize lists
        if  self.sigs.ndim == 3:
            self.models = np.zeros((len(self.df_features), len(self.df_features[0]))).tolist()
        else:
            self.models = np.zeros(len(self.df_features)).tolist()

        # Convert dataframes to Bycycle objects
        self.n_dims = self.sigs.ndim

        for dim0, sig in enumerate(self.sigs):

            if self.n_dims == 3:

                for dim1, sig_ in enumerate(sig):

                    # Intialize
                    bm = Bycycle(self.center_extrema, self.burst_method, self.burst_kwargs,
                                 self.thresholds, self.find_extrema_kwargs, self.return_samples)
                    # Load
                    bm.load(self.df_features[dim0][dim1], sig_, self.fs, self.f_range)

                    # Set
                    self.models[dim0][dim1] = bm

            else:

                # Intialize
                bm = Bycycle(self.center_extrema, self.burst_method, self.burst_kwargs,
                             self.thresholds, self.find_extrema_kwargs, self.return_samples)
                # Load
                bm.load(self.df_features[dim0], sig, self.fs, self.f_range)

                # Set
                self.models[dim0] = bm


    def recompute_edges(self, reduction=None):
        """Recomputes features for cycles on the edge of bursts.

        Parameters
        ----------
        reduction : float, optional, default: None
            Reduces all float thresholds by given amount.
        """

        for dim0, sig in enumerate(self.sigs):
            if self.n_dims == 3:
                for dim1 in range(len(sig)):
                    self.models[dim0][dim1].recompute_edges(reduction)
            else:
                 self.models[dim0].recompute_edges(reduction)
