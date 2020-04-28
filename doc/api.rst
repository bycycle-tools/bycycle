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
    twothresh_amp

Filter Functions
================

.. currentmodule:: bycycle.filt

.. autosummary::
    :toctree: generated/

    bandpass_filter
    lowpass_filter
    amp_by_time
    phase_by_time

Plotting Function
=================

.. currentmodule:: bycycle.burst

.. autosummary::
    :toctree: generated/

    plot_burst_detect_params

Simulation Functions
====================

Noise
~~~~~

.. currentmodule:: bycycle.sim

.. autosummary::
    :toctree: generated/

    sim_filtered_brown_noise
    sim_brown_noise

Oscillators
~~~~~~~~~~~

.. currentmodule:: bycycle.sim

.. autosummary::
    :toctree: generated/

    sim_noisy_oscillator
    sim_bursty_oscillator
    sim_noisy_bursty_oscillator
