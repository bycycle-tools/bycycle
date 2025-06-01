"""Default parameters and bounds."""

DEFAULT_OPT = {
    "sine": {
        "phase": {"p0": 0.75, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_b": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_d": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 0.0, "bounds": (-100, 100)},
    },
    "asine": {
        "rdsym": {"p0": 0.5, "bounds": (0.0, 1.0)},
        "phase": {"p0": 0.75, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_b": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_d": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 0.0, "bounds": (-100, 100)},
    },
    "asym_harmonic": {
        "phi": {"p0": 0.0, "bounds": (-10.0, 10.0)},
        "n_harmonics": {"p0": 1, "bounds": (1, 1)},
        "phase": {"p0": 0.5, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_b": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_d": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 0.0, "bounds": (-100, 100)},
    },
    "double_exp": {
        "tau_d": {"p0": 0.1, "bounds": (0.01, 1.0)},
        "tau_r": {"p0": 0.05, "bounds": (0.01, 1.0)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_b": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_d": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 0.0, "bounds": (-100, 100)},
    },
    "exp_cos": {
        "exp": {"p0": 1.0, "bounds": (0.1, 10)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_b": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_d": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 0.0, "bounds": (-100, 100)},
    },
    "gaussian": {
        "std": {"p0": 1.0, "bounds": (0.01, 1)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_b": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_d": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 0.0, "bounds": (-100, 100)},
    },
    "skewed_gaussian": {
        "center": {"p0": 0.5, "bounds": (0.2, 0.8)},
        "std": {"p0": 0.5, "bounds": (0.1, 0.25)},
        "alpha": {"p0": 0.0, "bounds": (0.0, 2.0)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_b": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_d": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 0.0, "bounds": (-100, 100)},
    },
    "sawtooth": {
        "width": {"p0": 0.5, "bounds": (0.0, 1.0)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_b": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_d": {"p0": 0.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 0.0, "bounds": (-100, 100)},
    },
}

BOUNDS = {}
P0 = {}
KEYS = {}

# For each cycle type
for k in DEFAULT_OPT:
    _bounds = []
    _p0 = []
    # For each parameter
    for p in DEFAULT_OPT[k]:
        _bounds.append(DEFAULT_OPT[k][p]["bounds"])
        _p0.append(DEFAULT_OPT[k][p]["p0"])

    # For affine transform
    BOUNDS[k] = _bounds
    P0[k] = _p0
    KEYS[k] = list(DEFAULT_OPT[k].keys())
