from bycycle.recon.recon_obj import BycycleRecon
import numpy as np
from neurodsp.sim.cycles import sim_normalized_cycle


def test_constructor():
    recon = BycycleRecon()


def create_new_cycles():
    phase = {"phase": {"default": 0.0, "bounds": (0.0, 1.0)}}

    mappings = {
        "sine": {"phase": 4, "asym": False, "cyc_kwargs": phase},
        "asine": {
            "phase": 4,
            "asym": True,
            "cyc_kwargs": {"rdsym": {"default": 0.5, "bounds": (0.0, 1.0)}, **phase},
        },
        "asym_harmonic": {
            "phase": 2,
            "asym": True,
            "cyc_kwargs": {
                "phi": {"default": 0.0, "bounds": (-10.0, 10.0)},
                "n_harmonics": {"default": 1, "bounds": (0, 5)},
                **phase,
            },
        },
        "2exp": {
            "phase": 4,
            "asym": True,
            "cyc_kwargs": {
                "tau_d": {"default": 0.1, "bounds": (0.01, 1.0)},
                "tau_r": {"default": 0.05, "bounds": (0.01, 1.0)},
                **phase,
            },
        },
        "exp_cos": {
            "phase": 4,
            "asym": False,
            "cyc_kwargs": {"exp": {"default": 1.0, "bounds": (0.1, 10)}, **phase},
        },
        "gaussian": {
            "phase": 1,
            "asym": False,
            "cyc_kwargs": {"std": {"default": 1.0, "bounds": (0.01, 1)}, **phase},
        },
        "skewed_gaussian": {
            "phase": 1,
            "asym": True,
            "cyc_kwargs": {
                "center": {"default": 0.5, "bounds": (0.2, 0.8)},
                "std": {"default": 0.5, "bounds": (0.1, 0.25)},
                "alpha": {"default": 0.0, "bounds": (0.0, 2.0)},
                **phase,
            },
        },
        "sawtooth": {
            "phase": 2,
            "asym": False,
            "cyc_kwargs": {"width": {"default": 0.5, "bounds": (0.0, 1.0)}, **phase},
        },
    }

    n_sims = 10

    cycle_targets = np.zeros((len(mappings), n_sims, 100))
    target_values = []

    for im, cyc_type in enumerate(mappings.keys()):
        # Get simulation kwargs or parameters
        cmap = mappings[cyc_type]["cyc_kwargs"]
        kwargs = {}

        if cmap is not None:
            for i in cmap.keys():
                pdist = np.random.uniform(
                    cmap[i]["bounds"][0], cmap[i]["bounds"][1], n_sims
                )
                if isinstance(cmap[i]["default"], int):
                    pdist = pdist.astype(int)
                kwargs[i] = pdist
        target_values.append(kwargs)

        _kwargs = kwargs.copy()
        del _kwargs["phase"]

        # For each cycle type, run 100 simulations of randomized parameters
        for i in range(n_sims):
            # Collect params
            sim_kwargs = {}
            for k in _kwargs.keys():
                sim_kwargs[k] = _kwargs[k][i]

            # Seed and simulate
            np.random.seed(i)

            cycle_targets[im, i] = sim_normalized_cycle(
                0.1, 1000, cyc_type, phase="min", **sim_kwargs
            )[:100]
    cycle_targets_dim0 = cycle_targets.shape[0]
    cycle_targets_dim1 = cycle_targets.shape[1]
    cycle_targets_newshape = cycle_targets_dim0 * cycle_targets_dim1
    new_cycles = np.full(cycle_targets_newshape, fill_value=None)
    ind_val = 0
    for _, i in enumerate(cycle_targets):
        for j in i:
            new_cycles[ind_val] = j
            ind_val += 1
    print(new_cycles.shape)
    return new_cycles


def test_fit():
    # assert False
    cycles = create_new_cycles()
    rc = BycycleRecon()
    rc.fit(cycles)
    models =  rc.models
    # assert n models,
    # assert models has optimized parameters (float)
    # check model.popt, model.loss, model.rsq for setting
    print(models)
