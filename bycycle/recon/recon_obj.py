# AUTHORS: Ryan Hammonds, Kenton Guarian
# DATE: August 8, 2023
# DESCRIPTION: BycycleRecon class for fitting & reconstructing bursting
# cycles. This class can fit a set of cycles to the best fit of the
# cycle types neurodsp.sim_normalized_cycle can generate, and an affine
# transform. Using that fit, it can approximately reconstruct the
# cycles from the fit parameters.

import numpy as np

from scipy.optimize import minimize

from skimage.transform import (
    AffineTransform,
)

from neurodsp.sim.cycles import (
    sim_normalized_cycle,
)

# to accelerate the fitting process
# TODO: write tests for this
from multiprocessing import Pool

# NOTE ABOUT (.)*PARAM(.)* objects: They are
# values for sim_normalized_cycle parameters when
# called by scipy.optimize.minimize. The goal of
# this package is to fit the best fit of an affine
# transform of sim_normalized_cycle to the data.

# The affine transform is defined by a 6-element
# array of float because the bottom row of the
# affine transform matrix is always [0, 0, 1].
DEFAULT_AFFINE_PARAMS = {
    "p0": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "bounds": [(-100, 100) for _ in range(6)],
}

# parameters for sim_normalized_cycle before
# affine transform
SIGNAL_PARAM_DICT = {
    "sine": {
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
    },
    "asine": {
        "rdsym": {"p0": 0.5, "bounds": (0.0, 1.0)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
    },
    "asym_harmonic": {
        "phi": {"p0": 0.0, "bounds": (-10.0, 10.0)},
        "n_harmonics": {"p0": 1, "bounds": (1, 1)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
    },
    "2exp": {
        "tau_d": {"p0": 0.1, "bounds": (0.01, 1.0)},
        "tau_r": {"p0": 0.05, "bounds": (0.01, 1.0)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
    },
    "exp_cos": {
        "exp": {"p0": 1.0, "bounds": (0.1, 10)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
    },
    "gaussian": {
        "std": {"p0": 1.0, "bounds": (0.01, 1)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
    },
    "skewed_gaussian": {
        "center": {"p0": 0.5, "bounds": (0.2, 0.8)},
        "std": {"p0": 0.5, "bounds": (0.1, 0.25)},
        "alpha": {"p0": 0.0, "bounds": (0.0, 2.0)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
    },
    "sawtooth": {
        "width": {"p0": 0.5, "bounds": (0.0, 1.0)},
        "phase": {"p0": 0.0, "bounds": (0.0, 1.0)},
    },
}

# for each cycle type
BOUNDS = {}
P0 = {}
KEYS = {}

# this loop constructs the BOUNDS, P0, and KEYS dicts
# it associates each cycle type with its own BOUNDS,
# P0, and KEYS, which are separated into their own dicts
# for use in the mininimize_wrapper call in fit.
for cycle_type in SIGNAL_PARAM_DICT.keys():
    _param_bounds = []
    _init_param_guess = []
    for param_name in SIGNAL_PARAM_DICT[cycle_type]:
        _param_bounds.append(SIGNAL_PARAM_DICT[cycle_type][param_name]["bounds"])
        _init_param_guess.append(SIGNAL_PARAM_DICT[cycle_type][param_name]["p0"])
    BOUNDS[cycle_type] = _param_bounds
    P0[cycle_type] = _init_param_guess
    KEYS[cycle_type] = list(SIGNAL_PARAM_DICT[cycle_type].keys())


def normalized_cycle(y_true, cycle_type, keys, dtypes, *params):
    """
    normalized_cycle is a wrapper for neurodsp.sim_normalized_cycle

    Parameters
    ----------
    y_true: 1d array
        Time series.
    cycle_type: str
        Type of cycle to simulate. Must be one of the following:
        'sine', 'asine', 'asym_harmonic', '2exp', 'exp_cos', 'gaussian',
        'skewed_gaussian', 'sawtooth'.
    keys: list of str
        Keys for the parameters of the cycle type.
    dtypes: list of type
        Types for the parameters of the cycle type.
    params: list of float
        Parameters for the cycle type.
    """
    y_pred = sim_normalized_cycle(
        1,
        len(y_true),
        cycle_type,
        **{k: dt(v) for k, dt, v in zip(keys, dtypes, params)}
    )[: len(y_true)]
    return y_pred


def affine_transform_wrapper(y_true, cycle_type, keys, dtypes, *params, rsq=False):
    """
    affine transform wrapper. This function

    Parameters:
    -----------
    y_true: 1d array
        Time series.
    cycle_type: str
        Type of cycle to simulate. Must be one of the following:
        'sine', 'asine', 'asym_harmonic', '2exp', 'exp_cos', 'gaussian',
        'skewed_gaussian', 'sawtooth'.
    keys: list of str
        Keys for the parameters of the cycle type.
    dtypes: list of type
        Types for the parameters of the cycle type.
    params: list of float
        Parameters for the cycle type.
    """

    # num_keys is the number of keys that are not affine parameters. The affine
    # parameters are the last 6 elements of params.
    y = normalized_cycle(y_true, cycle_type, keys[:-6], dtypes[:-6], *(params[:][:-6]))
    affine_params = params[-6:]

    if len(affine_params) != 6:
        raise ValueError("affine_params must be a 6-element array of float")

    Transform = AffineTransform(
        [
            [affine_params[0], affine_params[1], affine_params[2]],
            [affine_params[3], affine_params[4], affine_params[5]],
            [0, 0, 1],
        ]
    )

    x = np.arange(len(y))
    right_operand = np.array([x, y])
    y = Transform(right_operand.T)
    y = (y.T)[1]

    retval = None
    if rsq:
        retval = r_squared(y, y_true)
    else:
        retval = mean_squared_error(y_true=y_true, y_pred=y)
    return retval


def mean_squared_error(y_true, y_pred):
    # R-squared
    return np.mean((y_true - y_pred) ** 2)


def r_squared(y_true, y_pred):
    # R-squared
    return np.corrcoef(y_true, y_pred)[0][1] ** 2


# bycycle/recon/objs.py


# TODO: hacky fix for 2exp
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

        # popt: scipy's optimized parameters.
        # rename to p_opt
        self.popt = None
        self.loss = None
        self.rsq = None


class BycycleRecon:
    def __init__(
        self, cycles=None, affine=True, p0=None, bounds=None, param_names=None
    ):
        self.cycles = list(SIGNAL_PARAM_DICT.keys()) if cycles is None else cycles

        self.dtypes = {}
        self.bounds = BOUNDS if bounds is None else bounds
        self.p0 = P0 if p0 is None else p0
        self.param_names = (
            {k: list(SIGNAL_PARAM_DICT[k].keys()) for k in SIGNAL_PARAM_DICT}
            if param_names is None
            else param_names
        )

        self.affine = affine
        self.popt = None
        self.loss = None
        self.rsq = None

        self._bases = []
        for cyc in self.cycles:
            # self.param_names[cyc].extend(list(DEFAULT_AFFINE_PARAMS.keys()))
            self.param_names[cyc].extend("affine%d" % i for i in range(6))
            self.p0[cyc].extend(DEFAULT_AFFINE_PARAMS["p0"])
            self.dtypes[cyc] = [type(i) for i in self.p0[cyc]]
            self.bounds[cyc].extend(DEFAULT_AFFINE_PARAMS["bounds"])
            # TODO: can't use name "2exp" (rename this to something else) in attribute
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

    # add non-affine option
    def minimize_wrapper(self, y_true, cyc_sim, keys, dtypes, icyc, len_x):
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
        lfunc = lambda params: affine_transform_wrapper(
            y_true, cyc_sim, model.param_names, dtypes, *params
        )
        # TODO: expose some of the import optimization options
        res = minimize(
            lfunc, self.p0[cyc_sim], bounds=self.bounds[cyc_sim], method="L-BFGS-B"
        )

        model.popt[icyc] = res.x
        model.loss[icyc] = res.fun
        rsq = affine_transform_wrapper(
            y_true, cyc_sim, model.param_names, dtypes, *res.x, rsq=True
        )
        model.rsq[icyc] = rsq

    def fit(self, X, n_jobs=1):
        """for each cycle type, fit the best fit of an affine-transformed
        neurodsp.sim_normalized_cycle to the data.

        Parameters
        ----------
        self: BycycleRecon
        X: 2d array (likely jagged)
            Time series.
        n_jobs: int
            Number of jobs to run in parallel. Default is 1.
        """
        if n_jobs < 1:
            raise ValueError("n_jobs must be greater than 1")

        self.popt = np.zeros((len(X), len(self.cycles)))
        self.loss = np.zeros((len(X), len(self.cycles)))
        self.rsq = np.zeros((len(X), len(self.cycles)))

        dtypes = [type(i) for i in self.p0]
        if n_jobs == 1:
            for cycle_index, x in enumerate(X):
                for cyc_sim in self.cycles:
                    self.minimize_wrapper(
                        x,
                        cyc_sim,
                        self.param_names[cyc_sim],
                        self.dtypes[cyc_sim],
                        cycle_index,
                        len(X),
                    )
        else:
            # make a thread pool
            p = Pool(n_jobs)
            # count tasks, since they might run out of order
            task_count = 0
            for cycle_index, x in enumerate(X):
                for cyc_sim in self.cycles:
                    p.apply_async(
                        self.minimize_wrapper,
                        args=(
                            x,
                            cyc_sim,
                            self.param_names[cyc_sim],
                            cycle_index,
                            len(X),
                        ),
                    )
                    task_count += 1
            # no new tasks
            p.close()
            # wait for all tasks to finish
            p.join()
