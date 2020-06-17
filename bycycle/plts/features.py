"""Plot cycle features."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################


def plot_feature_hist(df, burst_param, ax=None, bins='auto', **kwargs):
    """ Plot a histogram for a cycle feature.

    Parameters
    ----------
    df : pandas.DataFrame or 1d array
        Dataframe output from :func:`~.compute_features` or a 1d array.
    burst_param : str
        Column name of the parameter of interest in ``df``.
    ax : matplotlib.Axes, optional, default: None
        Figure axes upon which to plot.
    bins : int or string, optional, default: 'auto'
        The number of bins or binning strategy string,
        as specified in matplotlib.pyplot.hist.
    **kwargs
        Keyword arguments to pass into matplotlib methods.

    Notes
    -----
    Default keyword arguments include:

    - ``xlabel``: str, default: ``burst_param``
    - ``figsize``: tuple of (float, float), default: (10, 10)
    - ``color``: str, default: 'k'
    - ``xlim``: tuple of (float, float), default: None
    - ``fontsize``: float, default: 15

    Examples
    --------
    See the `feature distribution example <http://bycycle-tools.github.io/bycycle/auto_examples/plot_theta_feature_distributions.html#plot-feature-distributions>`_.
    """

    # Limit dataframe to bursts
    if isinstance(df, pd.core.frame.DataFrame):
        df = df[df['is_burst']]

    # Optional keyword args
    figsize = kwargs.pop('figsize', (5, 5))
    color = kwargs.pop('color', 'k')
    xlabel = kwargs.pop('xlabel', burst_param)
    xlim = kwargs.pop('xlim', None)
    fontsize = kwargs.pop('fontsize', 15)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Axis style
    ax.set_xlabel(xlabel, size=fontsize)
    ax.set_ylabel('# cycles', size=fontsize)

    if xlim:
        ax.set_xlim(xlim)

    feature = df if not isinstance(df, pd.core.frame.DataFrame) else df[burst_param]

    ax.hist(feature, bins=bins, color=color, alpha=.5)







