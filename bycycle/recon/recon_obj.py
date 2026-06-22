# AUTHORS: Ryan Hammonds, Kenton Guarian
# DATE: August 8, 2023
# DESCRIPTION: BycycleRecon class for fitting & reconstructing bursting
# cycles. This class can fit a set of cycles to the best fit of the
# cycle types neurodsp.sim_normalized_cycle can generate, and an affine
# transform. Using that fit, it can approximately reconstruct the
# cycles from the fit parameters.

from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from skimage.transform import AffineTransform
from neurodsp.sim.cycles import sim_normalized_cycle
from bycycle.recon.defaults import DEFAULT_OPT, BOUNDS, P0

# to accelerate the fitting process
# TODO: write tests for this
from multiprocessing import Pool

def sim_and_trans(y_true, cycle_type, keys, dtypes, *params, return_pred=False):
    """Affine transformation.

    Parameters
    ----------
    y_true: 1d array
        Input signal
    return_pred : bool, optional, default: True
        Returns predicted y values.

    Returns
    -------
    affine_params: 1d array
        Affine transform parameters
    y_pred : 1d array, optionals
        Predicted values.
    """
    # Simulate
    num_keys = len(keys) - 6
    cycle_type = '2exp' if cycle_type == 'double_exp' else cycle_type
    sim_kwargs = zip(keys[:num_keys], dtypes[:num_keys], params[:num_keys])

    y = sim_normalized_cycle(
        1,
        len(y_true),
        cycle_type,
        **{k: dt(v) for k, dt, v in sim_kwargs}
    )[:len(y_true)]

    # Transform
    affine_params = params[num_keys:]

    Transform = AffineTransform(
        [
            [affine_params[0], affine_params[1], affine_params[2]],
            [affine_params[3], affine_params[4], affine_params[5]],
            [0, 0, 1],
        ]
    )

    right_operand = np.array([np.arange(len(y)), y])
    y_pred = Transform(right_operand.T)[:, 1]

    # MSE
    mse = ((y_pred-y_true)**2).mean()

    if return_pred:
        return mse, y_pred
    else:
        return mse


class Models:
    def __init__(self, bases):
        for b in bases:
            setattr(self, b.cycle, b)


class Basis:
    def __init__(self, cycle, p0, bounds, param_names):
        """Todo Docme
        """
        self.cycle = cycle
        self.p0 = p0
        self.bounds = bounds
        self.param_names = param_names

        # popt: scipy's optimized parameters.
        self.popt = None
        self.loss = None
        self.rsq = None

        self.y_true = None
        self.y_pred = None

    def plot(self, ind=None):
        """Todo Docme
        """
        if self.y_pred is None:
            raise ValueError('Basis model is not fit.')

        # Plot
        if ind is not None:
            plt.plot(self.y_true[ind], color='C0', label="True")
            plt.plot(self.y_pred[ind], color='C1', alpha=.8, label="Fit")
        else:
            for i, yt in enumerate(self.y_true):
                label = 'True' if i == len(self.y_true)-1 else None
                plt.plot(yt, color='C0', label=label)
            for i, yp in enumerate(self.y_pred):
                label = 'Fit' if i == len(self.y_pred)-1 else None
                plt.plot(yp, color='C1', alpha=.5, label=label)

        plt.legend()


class BycycleRecon:
    def __init__(
        self, cycles=None, p0=None, bounds=None, param_names=None
    ):
        """Todo Docme
        """
        self.cycles = list(DEFAULT_OPT.keys()) if cycles is None else cycles
        self.param_names = {k:list(DEFAULT_OPT[k].keys()) for k in self.cycles}
        self.dtypes = {}
        self.bounds = BOUNDS if bounds is None else bounds
        self.p0 = P0 if p0 is None else p0

        self._bases = []
        for cyc in self.cycles:

            # self.param_names[cyc].extend(list(DEFAULT_AFFINE_PARAMS.keys()))
            # self.param_names[cyc].extend("affine%d" % i for i in range(6))
            # self.p0[cyc].extend(DEFAULT_AFFINE_PARAMS["p0"])
            # self.dtypes[cyc] = [type(i) for i in self.p0[cyc]]
            # self.bounds[cyc].extend(DEFAULT_AFFINE_PARAMS["bounds"])

            self._bases.append(
                Basis(
                    cyc,
                    p0=self.p0[cyc],
                    bounds=self.bounds[cyc],
                    param_names=self.param_names[cyc],
                )
            )

        self.models = Models(self._bases)


    def minimize_wrapper(self, x, cyc_sim, dtypes, icyc, n_cycles, *params):
        """
        Minimize wrapper for multiprocessing and code separation
        """
        # Initalize arrays
        model = getattr(self.models, cyc_sim)

        if model.popt is None:
            model.popt = np.zeros((n_cycles, len(model.p0)))
            model.loss = np.zeros(n_cycles)
            model.rsq = np.zeros(n_cycles)
            model.y_true = np.zeros((n_cycles, len(x)))
            model.y_pred = np.zeros((n_cycles, len(x)))


        # Minimize loss
        dtypes = [type(i) for i in model.p0]
        lfunc = lambda params: sim_and_trans(
            x, cyc_sim, model.param_names, dtypes, *params
        )

        if cyc_sim == 'asine':
            self.p0[cyc_sim][0] = np.argmax(x) / len(x)
            self.p0[cyc_sim][1] = 1 - (np.argmax(x) / 2 / len(x))
            eps=[*[1e-2]*(len(self.p0[cyc_sim]) - 6), *([1e-10]*6)]
        else:
            eps = None

        res = minimize(
            lfunc, self.p0[cyc_sim], bounds=self.bounds[cyc_sim], method="L-BFGS-B",
            tol=1e-29, options=dict(eps=eps, ftol=1e-29, gtol=1e-29)
        )

        model.popt[icyc] = res.x
        model.loss[icyc] = res.fun

        # Predicted values and r-squared
        _, y_pred = sim_and_trans(
            x, cyc_sim, model.param_names, dtypes, *res.x, return_pred=True
        )
        model.y_pred[icyc] = y_pred
        model.y_true[icyc] = x
        model.rsq[icyc] = np.corrcoef(x, y_pred)[0][1] ** 2


    def fit(self, X, n_jobs=1):
        """Todo Docme
        """
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
                        self.p0[cyc_sim],
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
                            self.p0[cyc_sim],
                            cycle_index,
                            len(X),
                        ),
                    )
                    task_count += 1
            # no new tasks
            p.close()
            # wait for all tasks to finish
            p.join()
