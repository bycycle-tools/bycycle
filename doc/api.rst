.. _api_documentation:

=================
API Documentation
=================

API reference for the bycyle module.

.. contents::
   :local:
   :depth: 2

.. currentmodule:: bycycle

Cycle-by-cycle
===============

Feature Functions
~~~~~~~~~~~~~~~~~

.. currentmodule:: bycycle.features

.. autosummary::
   :toctree: generated/

   compute_features
   compute_shape_features
   compute_burst_features
   compute_cyclepoints

Group Feature Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: bycycle.group.features

.. autosummary::
   :toctree: generated/

   compute_features_2d
   compute_features_3d

Segmentation Functions
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: bycycle.cyclepoints

.. autosummary::
   :toctree: generated/

   find_extrema
   find_zerox

Waveform Phase Estimation Functions
===================================

.. currentmodule:: bycycle.cyclepoints

.. autosummary::
    :toctree: generated/

    extrema_interpolated_phase

Burst Detection Functions
=========================

.. currentmodule:: bycycle.burst

.. autosummary::
    :toctree: generated/

    detect_bursts_cycles
    detect_bursts_amp
    detect_bursts_dual_threshold
    recompute_edges


Plotting Functions
==================

.. currentmodule:: bycycle.plts

.. autosummary::
    :toctree: generated/

    plot_burst_detect_summary
    plot_burst_detect_param
    plot_cyclepoints_df
    plot_cyclepoints_array
    plot_feature_hist
    plot_feature_categorical

Utility Functions
=================

.. currentmodule:: bycycle.utils

.. autosummary::
    :toctree: generated/

    limit_df
    limit_signal
    get_extrema_df
    rename_extrema_df
