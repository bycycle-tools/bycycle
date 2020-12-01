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

- The module has been re-organized to increase modularity. Sub-folders and files are now organized
  as:

  - bycycle/burst/

    - amp.py : detects burst using a dual amplitude threshold method.
    - cycle.py : detects burst using cycle-by-cycle.
    - dualthresh.py : alias to the neurodsp function.
    - utils.py : enforces miniumum consecutive cycles for burst detection and allows cycles on the
      edges of burst to be recomputed with new thresholds.

  - bycycle/cyclepoints/

    - extrema.py : identify peak and trough locations in a time series.
    - phase.py : esimates the instantaneous phase of a time series.
    - zerox.py : identify the rise and decay zero-crossing in a time series.

  - bycycle/features/

    - burst.py : compute features directly use in burst detection.
    - cyclepoints.py : organizes the locations of extrema and zero-crossings for each cycle into a
      dataframe.
    - features.py : compute burst, shape, and cyclepoint features.
    - shape.py : compute shape features required to compute burst features.

  - bycycle/group/

    - features.py : compute features for 2D and 3D array of time series.
    - utils.py : create a progress bar for parallel computations.

  - bycycle/plts/

    - burst.py : plot burst detection parameters
    - cyclepoints.py : plot extrema and zero-crossings.
    - features.py : plot cycle features.

  - bycycle/utils/

    - checks.py : alias to neurodsp paramter check function.
    - dataframes.py : dataframe manipulation helper functions.
    - timeseries.py : timeseries manipulation helper functions.

- Add neurodsp dependency.

  - `filt.py` and `sim.py` have been replaced with equivalent neurodsp functions.

Naming Updates
~~~~~~~~~~~~~~

- The following argument and variables names have been changed throughout the module:

  - x -> sig
  - N_cycles_min -> min_n_cycles
  - Fs -> fs
  - Ps -> ps
  - Ts -> ts
  - zeroriseN -> rise_xs
  - zerofallN -> decay_xs
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

This includes new tutorials, examples, API list, and updates to pre-existing documentation.
