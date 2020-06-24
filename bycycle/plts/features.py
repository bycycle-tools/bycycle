"""Plot cycle features."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neurodsp.plts.utils import savefig

###################################################################################################
###################################################################################################

@savefig
def plot_feature_hist(feature, burst_param, bins='auto', ax=None, **kwargs):
    """Plot a histogram for a cycle feature.

    Parameters
    ----------
    feature : pandas.DataFrame or 1d array
        Dataframe output from :func:`~.compute_features` or a 1d array.
    burst_param : str
        Column name of the parameter of interest in ``df``.
    ax : matplotlib.Axes, optional, default: None
        Figure axes upon which to plot.
    bins : int or string, optional, default: 'auto'
        The number of bins or binning strategy string,
        as specified in matplotlib.pyplot.hist.
    **kwargs
        Keyword arguments to pass into `matplotlib.pyplot.hist`.

    Notes
    -----
    Default keyword arguments include:

    - ``xlabel``: str, default: ``burst_param``
    - ``figsize``: tuple of (float, float), default: (10, 10)
    - ``color``: str, default: 'k'
    - ``xlim``: tuple of (float, float), default: None
    - ``fontsize``: float, default: 15
    - ``alpha``: float, deafult: .5

    """

    # Limit dataframe to bursts
    if isinstance(feature, pd.core.frame.DataFrame):
        feature = feature[feature['is_burst']][burst_param]

    # Optional keyword args
    figsize = kwargs.pop('figsize', (5, 5))
    color = kwargs.pop('color', 'k')
    xlabel = kwargs.pop('xlabel', burst_param)
    xlim = kwargs.pop('xlim', None)
    fontsize = kwargs.pop('fontsize', 15)
    alpha = kwargs.pop('alpha', .5)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Axis style
    ax.set_xlabel(xlabel, size=fontsize)
    ax.set_ylabel('# cycles', size=fontsize)

    if xlim:
        ax.set_xlim(xlim)

    ax.hist(feature, bins=bins, color=color, alpha=alpha, **kwargs)


@savefig
def plot_feature_catplot(df, burst_param, group_by=None, ax=None, **kwargs):
    """ Plot a catplot for a cycle feature.

    Parameters
    ----------
    df : pandas.DataFrame.
        Dataframe output from :func:`~.compute_features`.
    burst_param : str
        Column name of the parameter of interest in ``df``.
    group_by : str, optional, default: None
        Dataframe column name of a grouping variable to split plotting by.
    ax : matplotlib.Axes, optional, default: None
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments to pass into `matplotlib.pyplot.scatter`.

    Notes
    -----
    Default keyword arguments include:

    - ``xlabel``: list, default: [1, 2, 3...]
    - ``ylabel``: str, default: ``burst_param``
    - ``figsize``: tuple of (float, float), default: (10, 10)
    - ``fontsize``: float, default: 20

    """

    # Split features by group if specified
    features = [df[burst_param]]

    if group_by is not None:

        features = []

        for group in np.unique(df[group_by].values):
            features.append(df[df[group_by] == group][burst_param])

    # Add random variance along x-axis
    x_values = [np.random.normal(idx+1, 0.05, len(feature)) for idx, feature in enumerate(features)]

    # Default keyword args
    figsize = kwargs.pop('figsize', (10, 10))
    ylabel = kwargs.pop('ylabel', burst_param)
    xlabel = kwargs.pop('xlabel', [idx+1 for idx in range(len(features))])
    fontsize = kwargs.pop('fontsize', 20)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for idx, feature in enumerate(features):
        ax.scatter(x_values[idx], feature, **kwargs)

    # Plot styling
    ax.set_xlim(.5, len(features)+.5)
    ax.set_xticks([idx+1 for idx in range(len(features))])
    ax.set_xticklabels(xlabel)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
