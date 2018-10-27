.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_tutorials_example_copy.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_tutorials_example_copy.py:


Theta oscillation cycle feature distributions
=============================================


Cycle feature distributions
---------------------------

This tutorial computes the distributions of cycle features for two recordings



.. code-block:: python


    import numpy as np
    import pandas as pd

    import neurodsp
    from bycycle.features import compute_features
    from scipy import stats

    import matplotlib.pyplot as plt



.. code-block:: python


    # Load data
    ca1_raw = np.load('data/ca1.npy')
    ec3_raw = np.load('data/ec3.npy')
    Fs = 1250
    f_theta = (4, 10)



.. code-block:: python


    # Apply lowpass filter
    fc = 25
    filter_seconds = .5
    ca1 = neurodsp.filter(ca1, Fs, 'lowpass', fc,
                          N_seconds=filter_seconds,
                          remove_edge_artifacts=False)
    ec3 = neurodsp.filter(ec3, Fs, 'lowpass', fc,
                          N_seconds=filter_seconds,
                          remove_edge_artifacts=False)



.. code-block:: python


    # Set parameters for defining oscillatory bursts
    osc_kwargs = {'amplitude_fraction_threshold':0,
                  'amplitude_consistency_threshold':.6,
                  'period_consistency_threshold':.75,
                  'monotonicity_threshold':.8,
                  'N_cycles_min':3}

    # Cycle-by-cycle analysis
    df_ca1 = compute_features(ca1, Fs, f_theta, center_extrema='T',
                                 burst_detection_kwargs=osc_kwargs)

    df_ec3 = compute_features(ec3, Fs, f_theta, center_extrema='T',
                                 burst_detection_kwargs=osc_kwargs)

    # Limit analysis only to oscillatory bursts
    df_ca1_cycles = df_ca1[df_ca1['is_burst']]
    df_ec3_cycles = df_ec3[df_ec3['is_burst']]


Plot time series



.. code-block:: python


    samplims = (10000, 12000)
    ca1_plt = ca1_raw[samplims[0]:samplims[1]]/1000
    ec3_plt = ec3_raw[samplims[0]:samplims[1]]/1000
    t = np.arange(0, len(ca1_plt)/Fs, 1/Fs)

    plt.figure(figsize=(12,3))
    plt.plot(t, ca1_plt, 'k')
    plt.xlim((0,1.6))
    plt.ylim((-2.4,2.4))
    plt.xlabel('Time (s)', size=15)
    plt.ylabel('CA1 Voltage (mV)', size=15)
    plt.xticks(size=12)
    plt.yticks(size=12)

    plt.figure(figsize=(12,3))
    plt.plot(t, ec3_plt, 'r')
    plt.xlim((0,1.6))
    plt.ylim((-2.4,2.4))
    plt.xlabel('Time (s)', size=15)
    plt.ylabel('EC3 Voltage (mV)', size=15)
    plt.xticks(size=12)
    plt.yticks(size=12)



Plot feature distributions



.. code-block:: python


    ylim = 3000

    plt.figure(figsize=(5,5))
    plt.hist(df_ca1_cycles['volt_amp']/1000, bins=np.arange(0, 8, .1), color='k', alpha=.5, label='CA1')
    plt.hist(df_ec3_cycles['volt_amp']/1000, bins=np.arange(0, 8, .1), color='r', alpha=.5, label='EC3')
    plt.xticks(np.arange(5), size=12)
    plt.legend(fontsize=15)
    plt.yticks(size=12)
    plt.xlim((0,4.5))
    plt.ylim((0,ylim))
    plt.xlabel('Cycle amplitude (mV)', size=15)
    plt.ylabel('# cycles', size=15)

    plt.figure(figsize=(5,5))
    plt.hist(df_ca1_cycles['period'] / Fs * 1000, bins=np.arange(0, 250, 5), color='k', alpha=.5)
    plt.hist(df_ec3_cycles['period'] / Fs * 1000, bins=np.arange(0, 250, 5), color='r', alpha=.5)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlim((0,250))
    plt.ylim((0,ylim))
    plt.xlabel('Cycle period (ms)', size=15)
    plt.ylabel('# cycles', size=15)

    plt.figure(figsize=(5,5))
    plt.hist(df_ca1_cycles['time_rdsym'], bins=np.arange(0, 1, .02), color='k', alpha=.5)
    plt.hist(df_ec3_cycles['time_rdsym'], bins=np.arange(0, 1, .02), color='r', alpha=.5)
    plt.plot([.5, .5], [0, ylim], 'k--')
    plt.ylim((0,ylim))
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlim((0,1))
    plt.xlabel('Rise-decay asymmetry\n(fraction of cycle in rise period)', size=15)
    plt.ylabel('# cycles', size=15)

    plt.figure(figsize=(5,5))
    plt.hist(df_ca1_cycles['time_ptsym'], bins=np.arange(0, 1, .02), color='k', alpha=.5)
    plt.hist(df_ec3_cycles['time_ptsym'], bins=np.arange(0, 1, .02), color='r', alpha=.5)
    plt.plot([.5, .5], [0, ylim], 'k--')
    plt.ylim((0,ylim))
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlim((0,1))
    plt.xlabel('Peak-trough asymmetry\n(fraction of cycle in peak period)', size=15)
    plt.ylabel('# cycles', size=15)


Try this other example notebook, go 


.. code-block:: python

    [here](04-FOOOFGroup.ipynb).
**Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_auto_tutorials_example_copy.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: example_copy.py <example_copy.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: example_copy.ipynb <example_copy.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
