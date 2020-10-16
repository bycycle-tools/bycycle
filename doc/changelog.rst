Changelog
=========

1.0.0
-----

Warning: the 1.0.0 is a breaking release from the beta 0.X.X series.

As compared to the prior series (0.X.X), some names and module organizations have changed. This
means existing code that uses Bycycle may no longer work as currently written with the new version,
and may need updating. You should update to the new version when you are ready to update your code
to reflect the new changes.

Note that the main changes are in code organization, some names, and the addition of many new
features. The algorithm itself has not changed, and results from the new version should be roughly
equivalent to those with older versions, with a few minor exceptions.

Code Oraganization
~~~~~~~~~~~~~~~~~~

- Increased modularity.

  - burst: ``burst.amp``, ``burst.cycle``, ``burst.dualthresh``, ``burst.utils``
  - cyclepoints: ``cyclepoints.extrema``, ``cyclepoints.zerox``, ``cyclepoints.phase``
  - features: ``features.features``, ``features.burst``, ``features.shape``, ``features.cyclepoints``
  - The dataframe output returned from ``compute_features`` now returns:

    1. ``df_features``: burst/shape cycle features.
    2. ``df_samples``: cyclepoint locations as signal indices.

- Neurodsp dependency.

  - `filt.py` and `sim.py` have been replaced with equivalent neurodsp functions.

Naming Updates
~~~~~~~~~~~~~~

- Consistency with neurodsp and PEP8 compliance:

  - x -> sig
  - N_cycles_min -> min_n_cycles
  - Fs -> fs

- PEP8

  - Ps -> ps
  - Ts -> ts
  - zeroriseN -> rise_xs
  - zerofallN -> decay_xs

- Consistency between burst detection kwargs and dataframe column names:

  - amplitude_fraction_threshold -> amp_fraction_threshold
  - amplitude_consistency_threshold -> amp_consistency_threshold

Code Updates
~~~~~~~~~~~~

The 1.X.X series adds a large number of code updates & additions, including:

- Parallel processing for 2D and 3D numpy arrays (``bycycle.group``):

  - ``compute_features_2d``
  - ``compute_features_3d``

- New plottings functions (``bycycle.plts``):

  - burst: ``plot_burst_detect_summary``, ``plot_burst_detect_param``
  - cyclepoints: ``plot_cyclepoints_df``, ``plot_cyclepoints_array``
  - features: ``plot_feature_hist``, ``plot_feature_categorical``

- Bug fixes:

  - Missing cyclepoints at the beginning/end of a signal.
  - Monotonicity computation didn't include all points in a cycle.
  - Zero-crossing off-by-one error when a signal crossed at exactly zero.

Documentation
~~~~~~~~~~~~~

The 1.X.X series comes with an updated documentation site.

As well as updating the tutorials, API list, and other existing documentation, there are
also (upcoming) new materials.
