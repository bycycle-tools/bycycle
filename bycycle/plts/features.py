"""Plot cycle features."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neurodsp.plts.utils import savefig

###################################################################################################
###################################################################################################

@savefig
def plot_feature_hist(feature, param_label, only_bursts=True, bins='auto', ax=None, **kwargs):
    """Plot a histogram for a cycle feature.

    Parameters
    ----------
    feature : pandas.DataFrame or 1d array
        Dataframe output from :func:`~.compute_features` or a 1d array.
    param_label : str
        Column name of the parameter of interest in ``df_features``.
    only_burst : bool, optional, default: True
        Whether to limit cycles to only those that are bursting.
    bins : int or string, optional, default: 'auto'
        The number of bins or binning strategy, as specified in `matplotlib.pyplot.hist`.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments to pass into `matplotlib.pyplot.hist`.

    Notes
    -----
    Default keyword arguments include:

    - ``xlabel``: str, default: ``param_label``
    - ``figsize``: tuple of (float, float), default: (10, 10)
    - ``color``: str, default: 'k'
    - ``xlim``: tuple of (float, float), default: None
    - ``fontsize``: float, default: 15
    - ``alpha``: float, default: .5

    Examples
    --------
    Plot a histogram of each cycle's mean band amplitude:

    >>> from bycycle.features import compute_features
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> df_features = compute_features(sig, fs, f_range=(8, 12), return_samples=False)
    >>> plot_feature_hist(df_features, 'band_amp', only_bursts=False)
    """

    # Limit dataframe to bursts
    if isinstance(feature, pd.core.frame.DataFrame) and only_bursts is True:
        feature = feature[feature['is_burst']][param_label]
    elif isinstance(feature, pd.core.frame.DataFrame) and only_bursts is False:
        feature = feature[param_label]

    # Default keyword args
    figsize = kwargs.pop('figsize', (5, 5))
    xlabel = kwargs.pop('xlabel', param_label)
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

    ax.hist(feature, bins=bins, alpha=alpha, **kwargs)

    if 'label' in kwargs:
        ax.legend(fontsize=fontsize)


@savefig
def plot_feature_categorical(df_features, param_label, group_by=None, ax=None, **kwargs):
    """Plot a cycle feature by one or more categories.

    Parameters
    ----------
    df_features : pandas.DataFrame
        Dataframe output from :func:`~.compute_features`.
    param_label : str
        Column name of the parameter of interest in ``df_features``.
    group_by : str, optional
        Dataframe column name of a grouping variable to split plotting by.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments to pass into `matplotlib.pyplot.scatter`.

    Notes
    -----
    Default keyword arguments include:

    - ``xlabel``: list, default: [1, 2, 3...]
    - ``ylabel``: str, default: ``param_label``
    - ``figsize``: tuple of (float, float), default: (10, 10)
    - ``fontsize``: float, default: 20

    Examples
    --------
    Plot and compare the rise-decay times of two asine signals:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from neurodsp.sim import sim_oscillation
    >>> from bycycle.group import compute_features_2d
    >>> fs = 500
    >>> sigs = np.array([sim_oscillation(5, fs, 10, cycle='asine', rdsym=0.2),
    ...                  sim_oscillation(5, fs, 10, cycle='asine', rdsym=0.8)])
    >>> features = compute_features_2d(sigs, fs, f_range=(8, 12), return_samples=False, n_jobs=2)
    >>> features[0]['group'], features[1]['group'] = 'low', 'high'
    >>> df_features = pd.concat([features[0], features[1]])
    >>> plot_feature_categorical(df_features, 'time_rdsym', group_by='group')
    """

    # Split features by group if specified
    features = [df_features[param_label]]

    if group_by is not None:

        features = []

        for group in np.unique(df_features[group_by].values):
            features.append(df_features[df_features[group_by] == group][param_label])

    # Add random variance along x-axis
    x_values = [np.random.normal(idx+1, 0.05, len(feature)) for idx, feature in enumerate(features)]

    # Default keyword args
    figsize = kwargs.pop('figsize', (10, 10))
    ylabel = kwargs.pop('ylabel', param_label)
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
