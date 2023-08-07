import time
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit, minimize

from skimage.transform import (
    warp,
    estimate_transform,
    AffineTransform,
    EuclideanTransform,
    SimilarityTransform,
    ProjectiveTransform,
    PolynomialTransform,
    PiecewiseAffineTransform,
)

from neurodsp.sim.cycles import (
    sim_sine_cycle,
    sim_asine_cycle,
    sim_asym_harmonic_cycle,
    sim_exp_cycle,
    sim_2exp_cycle,
    sim_exp_cos_cycle,
    sim_gaussian_cycle,
    sim_skewed_gaussian_cycle,
    sim_sawtooth_cycle,
    sim_normalized_cycle,
)


from multiprocessing import Pool, cpu_count

# bycycle/recon/defaults.py

DEFAULT_OPT = {
    "sine": {
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 3.1, "bounds": (-100, 100)},
        "affine_b": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 2.5, "bounds": (-100, 100)},
        "affine_d": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 1.0, "bounds": (-100, 100)},
    },
    "asine": {
        "rdsym": {"p0": 0.5, "bounds": (0.0, 1.0)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 3.1, "bounds": (-100, 100)},
        "affine_b": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 2.5, "bounds": (-100, 100)},
        "affine_d": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 1.0, "bounds": (-100, 100)},
    },
    "asym_harmonic": {
        "phi": {"p0": 0.0, "bounds": (-10.0, 10.0)},
        "n_harmonics": {"p0": 1, "bounds": (1, 1)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 3.1, "bounds": (-100, 100)},
        "affine_b": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 2.5, "bounds": (-100, 100)},
        "affine_d": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 1.0, "bounds": (-100, 100)},
    },
    "2exp": {
        "tau_d": {"p0": 0.1, "bounds": (0.01, 1.0)},
        "tau_r": {"p0": 0.05, "bounds": (0.01, 1.0)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 3.1, "bounds": (-100, 100)},
        "affine_b": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 2.5, "bounds": (-100, 100)},
        "affine_d": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 1.0, "bounds": (-100, 100)},
    },
    "exp_cos": {
        "exp": {"p0": 1.0, "bounds": (0.1, 10)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 3.1, "bounds": (-100, 100)},
        "affine_b": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 2.5, "bounds": (-100, 100)},
        "affine_d": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 1.0, "bounds": (-100, 100)},
    },
    "gaussian": {
        "std": {"p0": 1.0, "bounds": (0.01, 1)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 3.1, "bounds": (-100, 100)},
        "affine_b": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 2.5, "bounds": (-100, 100)},
        "affine_d": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 1.0, "bounds": (-100, 100)},
    },
    "skewed_gaussian": {
        "center": {"p0": 0.5, "bounds": (0.2, 0.8)},
        "std": {"p0": 0.5, "bounds": (0.1, 0.25)},
        "alpha": {"p0": 0.0, "bounds": (0.0, 2.0)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 3.1, "bounds": (-100, 100)},
        "affine_b": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 2.5, "bounds": (-100, 100)},
        "affine_d": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 1.0, "bounds": (-100, 100)},
    },
    "sawtooth": {
        "width": {"p0": 0.5, "bounds": (0.0, 1.0)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
        "affine_a": {"p0": 3.1, "bounds": (-100, 100)},
        "affine_b": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_c": {"p0": 2.5, "bounds": (-100, 100)},
        "affine_d": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_e": {"p0": 1.0, "bounds": (-100, 100)},
        "affine_f": {"p0": 1.0, "bounds": (-100, 100)},
    },
}

BOUNDS = {}
P0 = {}
KEYS = {}

# for each cycle type
for k in DEFAULT_OPT:
    _bounds = []
    _p0 = []
    # for each parameter
    for p in DEFAULT_OPT[k]:
        _bounds.append(DEFAULT_OPT[k][p]["bounds"])
        _p0.append(DEFAULT_OPT[k][p]["p0"])

    # for affine transform
    BOUNDS[k] = _bounds
    P0[k] = _p0
    KEYS[k] = list(DEFAULT_OPT[k].keys())

print("ok?")


# bycycle/recon/metrics.py
def sim_nc_wrapper(y_true, cycle_type, keys, dtypes, *params):
    # Ryan's original code.
    y_pred = sim_normalized_cycle(
        1,
        len(y_true),
        cycle_type,
        # adjustment for affine transform
        **{k: dt(v) for k, dt, v in zip(keys, dtypes, params)}
    )[: len(y_true)]

    # # My version:
    # kwargs={}
    # largest_len = max([len(x) for x in [keys, dtypes, params]])
    # for i in range(largest_len):
    #     kwargs[keys[i]] = dtypes[i](params[i])
    # y_pred = sim_normalized_cycle(1, len(y_true), cycle_type, **kwargs)[:len(y_true)]

    # return ((y_pred - y_true) ** 2).mean()
    return y_pred


"""
affine transform wrapper
y: input signal
affine_params: affine transform parameters (6-element array of float)
"""


def affine_transform_wrapper(y_true, cycle_type, keys, dtypes, *params):
    num_keys = len(keys) - 6
    y = sim_nc_wrapper(
        y_true, cycle_type, keys[:num_keys], dtypes[:num_keys], *(params[:][:num_keys])
    )
    if len(y) == 0:
        return y

    affine_params = params[:][num_keys:]
    if len(affine_params) != 6:
        raise ValueError("affine_params must be a 6-element array of float")

    Transform = AffineTransform(
        [
            [affine_params[0], affine_params[1], affine_params[2]],
            [affine_params[3], affine_params[4], affine_params[5]],
            [0, 0, 1]
        ]
    )
    x = np.arange(len(y))
    right_operand = np.array([x, y])
    # print(right_operand.shape)
    # right_operand=right_operand.T
    y=Transform(right_operand.T)
    y=(y.T)[1]
    return y[1]


def rsq(y_true, cycle_type, keys, dtypes, *params):
    # R-squared
    y_pred = sim_normalized_cycle(
        1,
        len(y_true),
        cycle_type,
        **{k: dt(v) for k, dt, v in zip(keys, dtypes, params)}
    )[: len(y_true)]

    return np.corrcoef(y_true, y_pred)[0][1] ** 2


# bycycle/recon/objs.py


class Models:
    def __init__(self, bases):
        for b in bases:
            setattr(self, b.cycle, b)


class Basis:
    def __init__(self, cycle, p0, bounds, param_names):
        self.cycle = cycle
        self.p0 = p0
        self.bounds = bounds
        self.param_names = param_names

        self.popt = None
        self.loss = None
        self.rsq = None


class BycycleRecon:
    def __init__(
        self, cycles=None, affine=True, p0=None, bounds=None, param_names=None
    ):
        self.cycles = list(DEFAULT_OPT.keys()) if cycles is None else cycles
        self.bounds = BOUNDS if bounds is None else bounds
        self.p0 = P0 if p0 is None else p0
        self.param_names = (
            {k: list(DEFAULT_OPT[k].keys()) for k in DEFAULT_OPT}
            if param_names is None
            else param_names
        )

        self.affine = affine

        self.popt = None
        self.loss = None
        self.rsq = None

        self._bases = []
        for cyc in self.cycles:
            # Todo: can't use name "2exp" (rename this to something else) in attribute
            #   (e.g. self.models.2exp won't work)
            self._bases.append(
                Basis(
                    cyc,
                    p0=self.p0[cyc],
                    bounds=self.bounds[cyc],
                    param_names=self.param_names[cyc],
                )
            )

        self.models = Models(self._bases)

    """
    Minimize wrapper for multiprocessing and code separation
    """

    def minimize_wrapper(self, x, cyc_sim, keys, dtypes, icyc, len_x, *params):
        # Initalize arrays
        model = getattr(self.models, cyc_sim)

        if model.popt is None:
            model.popt = np.zeros((len_x, len(model.p0)))
            model.loss = np.zeros(len_x)
            model.rsq = np.zeros(len_x)
            model.affine = np.full(
                shape=(len_x, 3, 3),
                fill_value=AffineTransform(matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                dtype=AffineTransform,
            )

        # Minimize loss
        dtypes = [type(i) for i in model.p0]
        lfunc = lambda params: affine_transform_wrapper(
            x, cyc_sim, model.param_names, dtypes, *params
        )

        # Todo: expose some of the import optimization options
        res = minimize(
            lfunc, self.p0[cyc_sim], bounds=self.bounds[cyc_sim], method="L-BFGS-B"
        )

        model.popt[icyc] = res.x
        model.loss[icyc] = res.fun

        # To-do: make rsq work
        # model.rsq[icyc] = rsq(x, cyc_sim, keys, dtypes, *params)

    def fit(self, X, n_jobs=1):
        self.popt = np.zeros((len(X), len(self.cycles)))
        self.loss = np.zeros((len(X), len(self.cycles)))
        self.rsq = np.zeros((len(X), len(self.cycles)))

        if n_jobs == 1:
            for icyc, x in enumerate(X):
                for cyc_sim in self.cycles:
                    self.minimize_wrapper(
                        x,
                        cyc_sim,
                        self.param_names[cyc_sim],
                        self.p0[cyc_sim],
                        icyc,
                        len(X),
                    )
        else:
            # make a thread pool
            p = Pool(n_jobs)
            # count tasks, since they might run out of order
            task_count = 0
            for icyc, x in enumerate(X):
                for cyc_sim in self.cycles:
                    p.apply_async(
                        self.minimize_wrapper,
                        args=(
                            x,
                            cyc_sim,
                            self.param_names[cyc_sim],
                            self.p0[cyc_sim],
                            icyc,
                            len(X),
                        ),
                    )
                    task_count += 1
            # no new tasks
            p.close()
            # wait for all tasks to finish
            p.join()


# # ## Simulate


# n_sims = 1000

# cycle_targets = np.zeros((len(DEFAULT_OPT), n_sims, 100))
# target_values = []

# for imap, cyc_type in enumerate(DEFAULT_OPT.keys()):

#     sim_kwargs = {}
#     _types = [type(i) for i in P0[cyc_type]]

#     for ikwarg, (k, b) in enumerate(zip(KEYS[cyc_type], BOUNDS[cyc_type])):
#         sim_kwargs[k] = np.random.uniform(b[0], b[1], n_sims).astype(_types[ikwarg])


#     # For each cycle type, run 1000 simulations of randomized parameters
#     for isim in range(n_sims):

#         kwargs = {k: v[isim] for k, v in sim_kwargs.items() if k != 'phase'}

#         # Seed and simulate
#         np.random.seed(isim)
#         cycle_targets[imap, isim] = sim_normalized_cycle(
#             1, 100, cyc_type, phase='min', **kwargs
#         )[:100]

# for ind, i in enumerate(cycle_targets):
#     plt.figure()
#     for j in i:
#         plt.plot(j, color='k', alpha=.1)
#     plt.title(list(DEFAULT_OPT.keys())[ind])


# # ## Fit


# br = BycycleRecon()
# br.fit(cycle_targets[1][:5])


# plt.plot(br.models.skewed_gaussian.loss)
# plt.plot(br.models.gaussian.loss)
