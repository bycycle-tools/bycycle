Glossary
========

A glossary of terms used in the module, with a description of how they are used, as well as common abbreviations.
Note, these terms are defined using peak-centered cycles.

Shape
-----
Features that describe the shape of a cycle.

.. glossary::

    Period (``period``)
        A single cycle of a rhythm, defined as the time between two consecutive troughs (or peaks).

    Peak (``time_peak``)
        The time between the rise and decay zero-crossings.

    Trough (``time_trough``)
        The time between the previous decay and the current rise.

    Rise (``time_rise``)
        The time between the current peak and the next trough.

    Decay (``time_decay``)
        The time between the current peak and the last trough.

    Rise-decay symmetry (``time_rdsym``)
        The fraction of the period in the rise phase.

    Peak-trough symmetry (``time_ptsym``)
        The fraction of the period in the peak phase.

    Sinusoidality
        A symmetrical wave with 0.5 rise-decay and peak-trough symmetry.

    Voltage Peak (``volt_peak``)
        The voltage at the current peak.

    Voltage Trough (``volt_trough``)
        The voltage at the trough before the peak.

    Voltage Rise (``volt_rise``)
        The voltage change between the previous trough and the current peak.

    Voltage Decay (``volt_decay``)
        The voltage change between the current peak and the next trough.

    Voltage Amplitude (``volt_amp``)
        The average of the rise and decay voltage.

    Band Amplitude (``band_amp``)
        The average analytic amplitude of the period or oscillation.

Burst
-----
Additional shape features that aid in determining where a signal may be bursting.

.. glossary::

    Amplitude Fraction (``amp_frac``)
        The average amplitude, relative to all other cycles. A value of 1.0 represents the cycle with
        the maximum average amplitude. Values approaching 0.0 represents the minimum.

    Amplitude Consistency (``amp_consistency``)
        The amplitude consistency of a cycle is equal to the maximum relative difference between rises and
        decay amplitudes across all pairs of adjacent rises and decays that include one of the flanks in the
        cycle (3 pairs) (e.g. if a rise is 10mV and a decay is 7mV, then its amplitude consistency is 0.7).

    Period Consistency (``period_consistency``)
        Period consistency is equal to the maximum relative difference between all pairs of adjacent periods
        that include the cycle of interest (2 pairs: current + previous cycles and current + next cycles) (e.g. if the
        previous, current, and next cycles have periods 60ms, 100ms, and 120ms, respectively, then the period
        consistency is min(60/100, 100/120) = 0.6)).

    Monotonicity (``monotonicity``)
        The monotonicity is the fraction of samples that the instantaneous derivative (numpy.diff) is consistent with
        the direction of the flank. (e.g. if in the rise, the instantaneous derivative is 90% positive, and in the decay,
        the instantaneous derivative is 80% negative, then the monotonicity of the cycle would be 0.85 ((0.9+0.8)/2)).
        The rise and decay flanks of the cycle should be mostly monotonic.

    Burst Fraction (``burst_fraction``)
        The proportion of a cycle's samples that are bursting according to the dual amplitude threshold algorithm
        (e.g. if a cycle contains three samples and the corresponding section of ``is_burst`` is
        np.array([True, True, False]), the burst fraction is 0.66 for that cycle).


Cyclepoints
-----------
Signal indices the mark the locations of rise/decay and peak/trough points in a signal.
Note: These indices may be converted to time by multiplying by the sampling rate (fs).

.. glossary::

    Sample Peak (``sample_peak``)
        Sample indices at which peaks occur.

    Sample Last Trough (``sample_last_trough``)
        Sample indices at which troughs occur before the sample peak.

    Sample Next Trough (``sample_next_trough``)
        Sample indices at which troughs occur following the sample peak.

    Sample Rise (``sample_zerox_rise``)
        Sample indices at which rising zero-crossings occur.

    Sample Decay (`sample_zerox_decay``)
        Sample indices at which decaying zero-crossings occur.

