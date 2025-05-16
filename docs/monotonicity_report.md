# Monotonicity Calculation Report

This report describes how monotonicity is calculated for each recording using the `compute_monotonicity` function in `bycycle/features/burst.py`.

## Monotonicity Calculation

Monotonicity is calculated for each cycle by computing the fraction of time segments between samples that are going in the same direction. The calculation is performed separately for the rise and decay periods of each cycle.

- The rise period is the segment between the trough and the peak.
- The decay period is the segment between the peak and the next trough.

The `compute_monotonicity` function takes `df_samples` and `sig` as input parameters and returns a 1D array containing the monotonicity values for each cycle.

## Example Calculation

Here is an example of how the monotonicity is calculated:

```python
import numpy as np
import pandas as pd

def compute_monotonicity(df_samples, sig):
    cycles = len(df_samples)
    monotonicity = np.zeros(cycles)

    for idx, row in enumerate(df_samples.to_dict('records')):
        if 'sample_peak' in df_samples.columns:
            rise_period = sig[int(row['sample_last_trough']):int(row['sample_peak'])+1]
            decay_period = sig[int(row['sample_peak']):int(row['sample_next_trough'])+1]
        else:
            decay_period = sig[int(row['sample_last_peak']):int(row['sample_trough'])+1]
            rise_period = sig[int(row['sample_trough']):int(row['sample_next_peak'])+1]

        decay_mono = np.mean(np.diff(decay_period) < 0)
        rise_mono = np.mean(np.diff(rise_period) > 0)
        monotonicity[idx] = np.mean([decay_mono, rise_mono])

    return monotonicity

# Example usage
df_samples = pd.DataFrame({
    'sample_last_trough': [0, 100, 200],
    'sample_peak': [50, 150, 250],
    'sample_next_trough': [100, 200, 300]
})
sig = np.sin(np.linspace(0, 2 * np.pi, 300))

monotonicity_values = compute_monotonicity(df_samples, sig)
print(monotonicity_values)
```

In this example, the `compute_monotonicity` function calculates the monotonicity values for each cycle in the signal `sig` based on the sample indices provided in `df_samples`.
