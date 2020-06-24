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

Features Functions
~~~~~~~~~~~~~~~~~~

.. currentmodule:: bycycle.features

.. autosummary::
   :toctree: generated/

   compute_features

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
    plot_feature_catplot

