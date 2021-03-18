"""Bycycle class object."""

from bycycle.features import compute_features
from bycycle.plts import plot_burst_detect_summary

from neurodsp.plts.utils import savefig

###################################################################################################
###################################################################################################

class Bycycle:
    """Compute bycycle features from a signal.

    Attributes
    ----------
    sig : 1d array
        Time series.
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
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for each cycle.
    """

    def __init__(self):
        """Initialize object settings"""

        self.center_extrema = 'peak'

        self.burst_method = 'cycles'
        self.burst_kwargs = None

        self.thresholds = {
            'amp_fraction_threshold': 0.,
            'amp_consistency_threshold': .5,
            'period_consistency_threshold': .5,
            'monotonicity_threshold': .8,
            'min_n_cycles': 3
        }

        self.find_extrema_kwargs = None
        self.return_samples = True

        self.df_features = None
        self.sig = None
        self.fs = None
        self.f_range = None


    def fit(self, sig, fs, f_range, center_extrema=None, burst_method=None,
            burst_kwargs=None, thresholds=None, find_extrema_kwargs=None, return_samples=None):
        """Run the bycycle algorithm on a signal.

        Parameters
        ----------
        sig : 1d array
            Time series.
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

        # Add settings as attributes
        self.sig = sig
        self.fs = fs
        self.f_range = f_range

        self.center_extrema = center_extrema if center_extrema is not None else self.center_extrema

        self.burst_method = burst_method if burst_method is not None else self.burst_method

        self.burst_kwargs = {} if burst_kwargs is None else self.burst_kwargs

        self.thresholds = thresholds if thresholds is not None else self.thresholds

        self.find_extrema_kwargs = {'filter_kwargs': {'n_cycles': 3}} if find_extrema_kwargs \
            is None else self.find_extrema_kwargs

        self.return_samples = return_samples

        df_features = compute_features(self.sig, self.fs, self.f_range, self.center_extrema,
                                       self.burst_method, self.burst_kwargs, self.thresholds,
                                       self.find_extrema_kwargs, self.return_samples)

        self.df_features = df_features


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
