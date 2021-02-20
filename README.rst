========================================================
bycycle - cycle-by-cycle analysis of neural oscillations
========================================================

|ProjectStatus|_ |Version|_ |BuildStatus|_ |Coverage|_ |License|_ |PythonVersions|_ |Publication|_

.. |ProjectStatus| image:: https://www.repostatus.org/badges/latest/active.svg
.. _ProjectStatus: https://www.repostatus.org/#active

.. |Version| image:: https://img.shields.io/pypi/v/bycycle.svg
.. _Version: https://pypi.python.org/pypi/bycycle/

.. |BuildStatus| image:: https://github.com/bycycle-tools/bycycle/actions/workflows/build.yml/badge.svg
.. _BuildStatus: https://github.com/bycycle-tools/bycycle/actions/workflows/build.yml

.. |Coverage| image:: https://codecov.io/gh/bycycle-tools/bycycle/branch/main/graph/badge.svg
.. _Coverage: https://codecov.io/gh/bycycle-tools/bycycle

.. |License| image:: https://img.shields.io/pypi/l/bycycle.svg
.. _License: https://opensource.org/licenses/Apache-2.0

.. |PythonVersions| image:: https://img.shields.io/pypi/pyversions/bycycle.svg
.. _PythonVersions: https://pypi.python.org/pypi/bycycle/

.. |Publication| image:: https://img.shields.io/badge/publication-10.1152%2Fjn.00273.2019-blue
.. _Publication: https://journals.physiology.org/doi/abs/10.1152/jn.00273.2019

ByCycle is a module for analyzing neural oscillations in a cycle-by-cycle approach.

Overview
--------

``bycycle`` is a tool for quantifying features of neural oscillations in the time domain, as opposed to the
frequency domain, using a cycle-by-cycle approach. Rather than applying narrowband filters and other methods
that use a sinusoidal basis, this approach segments a recording into individual cycles and directly measures
each of their properties including amplitude, period, and symmetry.

This is most advantageous for analyzing the waveform shape properties of neural oscillations.
It may also provide advantages for studying traditional amplitude and frequency effects, as well.
Using cycle properties can also be used for burst detection.

A full description of the method and approach is available in the paper below.

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
- `Glossary <https://bycycle-tools.github.io/bycycle/glossary.html>`_:
  which defines key terms used in the module

Dependencies
------------

``bycycle`` is written in Python, and requires >= Python 3.5 to run.

It has the following dependencies:

- `neurodsp <https://github.com/neurodsp-tools/neurodsp>`_ >= 2.1.0
- `numpy <https://github.com/numpy/numpy>`_ >= 1.18.5
- `scipy <https://github.com/scipy/scipy>`_ >=  1.4.1
- `pandas <https://github.com/pandas-dev/pandas>`_ >= 0.25.3
- `matplotlib <https://github.com/matplotlib/matplotlib>`_ >= 3.0.3
- `pytest <https://github.com/pytest-dev/pytest>`_ (optional)

Install
-------

The current major release is the 1.X.X series, which is a breaking change from the prior 0.X.X series.

Check the `changelog <https://bycycle-tools.github.io/bycycle/v1.0.0/changelog.html>`_ for notes on updating to the new version.

**Stable Version**

To install the latest stable release, you can use pip:

.. code-block:: shell

    $ pip install bycycle

ByCycle can also be installed with conda, from the conda-forge channel:

.. code-block:: shell

    $ conda install -c conda-forge bycycle

**Development Version**

To get the latest, development version, you can get the code using git:

.. code-block:: shell

    $ git clone https://github.com/bycycle-tools/bycycle

To install this cloned copy, move into the directory you just cloned, and run:

.. code-block:: shell

    $ pip install .

**Editable Version**

To install an editable, development version, move into the directory you cloned and install with:

.. code-block:: shell

    $ pip install -e .

Reference
---------

If you use this code in your project, please cite:
::

    Cole SR & Voytek B (2019) Cycle-by-cycle analysis of neural oscillations. Journal of neurophysiology
    122(2), 849-861. DOI: 10.1152/jn.00273.2019

Direct Link: https://doi.org/10.1152/jn.00273.2019

Contribute
----------

This project welcomes and encourages contributions from the community!

To file bug reports and/or ask questions about this project, please use the
`Github issue tracker <https://github.com/bycycle-tools/bycycle/issues>`_.

To see and get involved in discussions about the module, check out:

- the `issues board <https://github.com/bycycle-tools/bycycle/issues>`_ for topics relating to code updates, bugs, and fixes
- the `development page <https://github.com/bycycle-tools/Development>`_ for discussion of potential major updates to the module

When interacting with this project, please use the
`contribution guidelines <https://github.com/bycycle-tools/bycycle/blob/main/CONTRIBUTING.md>`_
and follow the
`code of conduct <https://github.com/bycycle-tools/bycycle/blob/main/CODE_OF_CONDUCT.md>`_.

Quickstart
----------

The main function in ``bycycle`` is ``compute_features``, which takes a time series and some
parameters as inputs, and returns a table of features for each cycle.

For example, consider having a 1-dimensional numpy array, ``sig``, which is a neural signal time series
sampled at 1000 Hz (``fs``) with an alpha (8-12 Hz, ``f_range``) oscillation. We can compute the table
of cycle features with the following:

.. code-block:: python

    from neurodsp.sim import sim_bursty_oscillation
    from bycycle.features import compute_features

    fs = 1000
    f_range = (8, 12)

    sig = sim_bursty_oscillation(10, fs, freq=10)
    df_features = compute_features(sig, fs, f_range)


Note that the above ``compute_features`` command used default parameters to localize extrema and detect
bursts of oscillations. However, it is important to knowledgeably select these parameters, as described in the
`algorithm tutorial <https://bycycle-tools.github.io/bycycle/auto_tutorials/plot_2_bycycle_algorithm.html>`_.

The following example introduces some potential parameter changes:

.. code-block:: python

    threshold_kwargs = {'amp_fraction_threshold': .2,
                        'amp_consistency_threshold': .5,
                        'period_consistency_threshold': .5,
                        'monotonicity_threshold': .8,
                        'min_n_cycles': 3}

    narrowband_kwargs = {'n_seconds': .5}

    df = compute_features(sig, fs, f_range, center_extrema='trough',
                          burst_method='cycles', threshold_kwargs=threshold_kwargs,
                          find_extrema_kwargs={'filter_kwargs': narrowband_kwargs})


- **center_extrema** determines how the cycles are segmented. 'T' indicates the center extrema is \
  a trough, so cycles are segmented peak-to-peak.
- **burst_method** selects which method to use for burst detection. The 'cycles' option \
  uses features of adjacent cycles in order to detect bursts (e.g. period consistency, see next \
  item). The 'amp' option uses an amplitude threshold to determine the cycles that are part of an \
  oscillatory burst.
- **threshold_kwargs** sets the keyword arguments for the burst detection functions. For the \
  ``cycles`` method, there are 5 keyword arguments (see the end of the \
  `algorithm tutorial <https://bycycle-tools.github.io/bycycle/auto_tutorials/plot_2_bycycle_algorithm.html>`_ \
  for advice on choosing these parameters).
- **find_extrema_kwargs** sets the keyword arguments for the function used to localize peaks and \
  troughs. Most notably, you can change the duration of the bandpass filter (``n_seconds``) used \
  during extrema localization (see section 1 of the \
  `algorithm tutorial <https://bycycle-tools.github.io/bycycle/auto_tutorials/plot_2_bycycle_algorithm.html>`_)

DataFrame Output
~~~~~~~~~~~~~~~~

The output of ``bycycle`` is a ``pandas.DataFrame``, which is a table, as shown below.
There are many columns, so the table is split into two images here.

Each row of this table corresponds to an individual segment of the signal, or a putative cycle of
the rhythm of interest.

.. image:: https://github.com/bycycle-tools/bycycle/raw/main/doc/img/cycledf_1.png

|

.. image:: https://github.com/bycycle-tools/bycycle/raw/main/doc/img/cycledf_2.png

Columns include:

- **sample_peak**: the sample of the signal at which the peak of this cycle occurs
- **period**: period of the cycle
- **time_peak**: duration of the peak period
- **volt_amp**: amplitude of this cycle, average of the rise and decay voltage
- **time_rdsym**: rise-decay symmetry, the fraction of the cycle in the rise period (0.5 is symmetric)
- **time_ptsym**: peak-trough symmetry, the fraction of the cycle in the peak period (0.5 is symmetric)
- **period_consistency**: consistency between the periods of the adjacent cycles, used in burst detection
- **is_burst**: indicator if the cycle is part of an oscillatory burst

The features in this table can be further analyzed, as demonstrated in the
`resting state data tutorial <https://bycycle-tools.github.io/bycycle/auto_tutorials/plot_2_bycycle_algorithm.html>`_
and the `data example <https://bycycle-tools.github.io/bycycle/auto_examples/plot_1_theta_feature_distributions.html>`_.
For example, we may be interested in the distribution of rise-decay symmetry values in a resting state recording, shown below.

Burst Detection Results
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://github.com/bycycle-tools/bycycle/raw/main/doc/img/bursts_detected.png

Funding
-------

Supported by NIH award R01 GM134363 from the
`NIGMS <https://www.nigms.nih.gov/>`_.

.. image:: https://www.nih.gov/sites/all/themes/nih/images/nih-logo-color.png
  :width: 400

|
