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
    detect_bursts_df_amp

Plotting Function
=================

.. currentmodule:: bycycle.burst

.. autosummary::
    :toctree: generated/

    plot_burst_detect_params
