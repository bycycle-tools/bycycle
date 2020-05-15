========================================================
bycycle - cycle-by-cycle analysis of neural oscillations
========================================================

|ProjectStatus|_ |Version|_ |BuildStatus|_ |codecov|_ |License|_ |PythonVersions|_ |Publication|_

.. |ProjectStatus| image:: https://www.repostatus.org/badges/latest/active.svg
.. _ProjectStatus: https://www.repostatus.org/#active

.. |Version| image:: https://img.shields.io/pypi/v/bycycle.svg
.. _Version: https://pypi.python.org/pypi/bycycle/

.. |BuildStatus| image:: https://travis-ci.org/bycycle-tools/bycycle.svg
.. _BuildStatus: https://travis-ci.org/bycycle-tools/bycycle

.. |codecov| image:: https://codecov.io/gh/bycycle-tools/bycycle/branch/master/graph/badge.svg
.. _codecov : https://codecov.io/gh/bycycle-tools/bycycle

.. |License| image:: https://img.shields.io/pypi/l/bycycle.svg
.. _License: https://opensource.org/licenses/Apache-2.0

.. |PythonVersions| image:: https://img.shields.io/pypi/pyversions/bycycle.svg
.. _PythonVersions: https://pypi.python.org/pypi/bycycle/

.. |Publication| image:: https://img.shields.io/badge/publication-10.1152%2Fjn.00273.2019-blue
.. _Publication: https://journals.physiology.org/doi/abs/10.1152/jn.00273.2019


Overview
--------

``bycycle`` is a python implementation of a cycle-by-cycle approach to analyzing neural oscillations
(`Cole & Voytek, J Neurophysiol 2019 <https://journals.physiology.org/doi/abs/10.1152/jn.00273.2019>`_).
This approach quantifies features of neural oscillations in the time domain as opposed to the
frequency domain. Rather than applying narrowband filters and other methods that utilize a
sinusoidal basis, this characterization segments a recording into individual cycles and directly
measures each of their properties including amplitude, period, and symmetry. This is most
advantageous for analyzing the waveform shape properties of neural oscillations, but it may also
provide advantages for studying traditional amplitude and frequency effects, as well. It also
implements burst detection, which has gained traction recently (
`Jones, 2016 <https://www.sciencedirect.com/science/article/pii/S0959438816300769?via%3Dihub>`_).
Therefore, we only analyze oscillatory properties when there is indeed an oscillation.

A full description of the method and approach is available in the paper below.


Reference
---------

If you use this code in your project, please cite:
::

    Cole SR & Voytek B (2019) Cycle-by-cycle analysis of neural oscillations. J Neurophysiol
    122:2, 849-861. doi: https://doi.org/10.1152/jn.00273.2019

Direct Link: https://journals.physiology.org/doi/abs/10.1152/jn.00273.2019


Documentation
-------------

Documentation for ``bycycle`` is available on the
`documentation site <https://bycycle-tools.github.io/bycycle/index.html>`_.

This documentation includes:

- `Tutorials <https://bycycle-tools.github.io/bycycle/auto_tutorials/index.html>`_:
  with a step-by-step guide through the approach and how to apply it
- `Examples <https://bycycle-tools.github.io/bycycle/auto_examples/index.html>`_:
  demonstrating an example analysis and use case
- `API list <https://bycycle-tools.github.io/bycycle/api.html>`_:
  which lists and describes all the code and functionality available in the module


Dependencies
------------

``bycycle`` is written in Python, and is tested to work with Python 3.5.

It has the following dependencies:

- `numpy <https://github.com/numpy/numpy>`_
- `scipy <https://github.com/scipy/scipy>`_ >= 0.19
- `pandas <https://github.com/pandas-dev/pandas>`_
- `matplotlib <https://github.com/matplotlib/matplotlib>`_
- `pytest <https://github.com/pytest-dev/pytest>`_ (optional)


Install
-------

**Stable Version**

To install the latest stable release of ``bycycle``, you can use pip:

.. code-block:: shell

    $ pip install bycycle

**Development Version**

To get the lastest, development version, you can get the code using git:

.. code-block:: shell

    $ git clone https://github.com/bycycle-tools/bycycle

To install this cloned copy, move into the directory you just cloned, and run:

.. code-block:: shell

    $ pip install .

**Editable Version**

To install an editable, development version, move into the directory you cloned and install with:

.. code-block:: shell

    $ pip install -e .


Usage
-----

The main function in ``bycycle`` is ``compute_features``, which takes a time series and some
parameters as inputs and returns a table of features for each cycle. Consider having a 1-dimensional
numpy array, ``signal``, which is a neural signal time series sampled at 1000 Hz (``Fs``) that
contains an alpha (8-12 Hz, ``f_range``) oscillation. We can compute the table of cycle features
with the following:

.. code-block:: python

    from bycycle.filt import lowpass_filter
    from bycycle.features import compute_features

    signal = lowpass_filter(signal, Fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)

    Fs = 1000
    f_range = (8, 12)
    df = compute_features(signal, Fs, f_range)


Note that a lowpass filter is applied in order to remove high-frequency power that may interfere
with extrema localization. (see section 0 of the
`algorithm tutorial <https://bycycle-tools.github.io/bycycle/auto_tutorials/plot_2_bycycle_algorithm.html#sphx-glr-auto-tutorials-plot-2-bycycle-algorithm-py>`_
for more details).

It's necessary to note that the above ``compute_features`` command used default parameters to
localize extrema and detect bursts of oscillations. However, it is important to knowledgeably select
these parameters, as described in the
`algorithm tutorial <https://bycycle-tools.github.io/bycycle/auto_tutorials/plot_2_bycycle_algorithm.html#sphx-glr-auto-tutorials-plot-2-bycycle-algorithm-py>`_.
The following example and text go over the different potential parameter changes:

.. code-block:: python

    burst_kwargs = {'amplitude_fraction_threshold': .2,
                    'amplitude_consistency_threshold': .5,
                    'period_consistency_threshold': .5,
                    'monotonicity_threshold': .8,
                    'N_cycles_min': 3}

    narrowband_kwargs = {'N_seconds': .5}

    df = compute_features(signal, Fs, f_range,
                          center_extrema='T',
                          burst_detection_method='cycles',
                          burst_detection_kwargs=burst_kwargs,
                          find_extrema_kwargs={'filter_kwargs': narrowband_kwargs},
                          hilbert_increase_N=True)


- **center_extrema** determines how the cycles are segmented. 'T' indicates the center extrema is \
  a trough, so cycles are segmented peak-to-peak.
- **burst_detection_method** selects which method for burst detection is used. The 'cycles' option \
  uses features of adjacent cycles in order to detect bursts (e.g. period consistency, see next \
  item). The 'amp' option uses an amplitude threshold to determine the cycles that are part of an \
  oscillatory burst.
- **burst_detection_kwargs** set the keyword arguments for the burst detection function. For the \
  ``cycles`` method, there are 5 keyword arguments (see the end of the \
  `algorithm tutorial <https://bycycle-tools.github.io/bycycle/auto_tutorials/plot_2_bycycle_algorithm.html#sphx-glr-auto-tutorials-plot-2-bycycle-algorithm-py>`_ \
  for advice on choosing these parameters).
- **find_extrema_kwargs** set the keyword arguments for the function used to localize peaks and \
  troughs. Most notably, you can change the duration of the bandpass filter (``N_seconds``) used \
  during extrema localization (see section 1 of the \
  `algorithm tutorial <https://bycycle-tools.github.io/bycycle/auto_tutorials/plot_2_bycycle_algorithm.html#sphx-glr-auto-tutorials-plot-2-bycycle-algorithm-py>`_).
- **hilbert_increase_N** is a boolean indicator of whether or not to zeropad the signal to bypass \
  complications that ``scipy.signal.hilbert()`` has with some long signal durations. Try setting \
  this parameter to `True` if this function is taking a long time to run. Note the Hilbert \
  Transform is used to compute the `band_amp` feature of each cycle, which is the average analytic \
  amplitude of the frequency of interest in that cycle. This is complementary to the `volt_amp` \
  measure, and may be desired for some burst detection applications.

Output
------

The output of ``bycycle`` is a ``pandas.DataFrame``, a table like the one shown below (with many
columns, so it is split into two images).

Each row of this table corresponds to an individuals segment of the signal, or a putative cycle of
the rhythm of interest.

.. image:: https://github.com/bycycle-tools/bycycle/raw/master/doc/img/cycledf_1.png

|

.. image:: https://github.com/bycycle-tools/bycycle/raw/master/doc/img/cycledf_2.png

Some of the columns include:

- **sample_peak**: the sample of the signal at which the peak of this cycle occurs
- **period**: period of the cycle
- **time_peak**: duration of the peak period
- **volt_amp**: amplitude of this cycle, average of the rise and decay voltage
- **time_rdsym**: rise-decay symmetry, the fraction of the cycle in the rise period (0.5 is symmetric)
- **time_ptsym**: peak-trough symmetry, the fraction of the cycle in the peak period (0.5 is symmetric)
- **period_consistency**: consistency between the periods of the adjacent cycles, used in burst detection
- **is_burst**: indicator if the cycle is part of an oscillatory burst

The features in this table can then go on to be analyzed, as demonstrated in the
`resting-state data tutorial <https://bycycle-tools.github.io/bycycle/auto_tutorials/plot_2_bycycle_algorithm.html#sphx-glr-auto-tutorials-plot-2-bycycle-algorithm-py>`_
and the `data example <https://bycycle-tools.github.io/bycycle/auto_examples/plot_theta_feature_distributions.html#sphx-glr-auto-examples-plot-theta-feature-distributions-py>`_.
For example, we may be interested in the distribution of rise-decay symmetry values in a resting state recording, shown below.

Rdsym Distribution:
~~~~~~~~~~~~~~~~~~~

.. image:: https://github.com/bycycle-tools/bycycle/raw/master/doc/img/rdsym_distribution.png

|

The plot below indicates in red the cycles of the signal that were identified as part of an
oscillatory burst.

Burst Detection Results
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://github.com/bycycle-tools/bycycle/raw/master/doc/img/bursts_detected.png

|
