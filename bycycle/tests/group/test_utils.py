"""Test group utility functions."""

from pytest import mark, param
import numpy as np

from bycycle.group.utils import progress_bar, check_kwargs_shape

###################################################################################################
###################################################################################################

@mark.parametrize("progress", [None, 'tqdm', param('invalid', marks=mark.xfail)])
def test_progress_bar(progress):

    n_iterations = 10
    iterable = [0] * n_iterations

    pbar = progress_bar(iterable, progress, n_iterations)

    assert len(pbar) == n_iterations


@mark.parametrize("axis", [0, 1, (0, 1), None, param(2, marks=mark.xfail)])
@mark.parametrize("kwargs_ndim", [1, 2])
@mark.parametrize("sigs_ndim", [2, 3])
@mark.parametrize("mismatch", [True, False])
def test_check_kwargs_shape(sim_args, axis, kwargs_ndim, sigs_ndim, mismatch):

    sigs = np.array([sim_args['sig']] * 2)
    if sigs_ndim == 3:
        sigs = np.array([sigs] * 2)

    kwargs = [{'center_extrema': 'peak'}]

    # Cases that will pass
    if ((axis == 0 or axis == 1) and kwargs_ndim == 1 and sigs_ndim == 3) or \
       (axis == None and kwargs_ndim == 1 and sigs_ndim == 2):
        kwargs = kwargs * 2
    elif (axis == (0, 1) or axis == None) and kwargs_ndim == 1 and sigs_ndim == 3:
        kwargs = kwargs * 4
    elif (axis == (0, 1) or axis == None) and kwargs_ndim == 2 and sigs_ndim == 3:
        kwargs = kwargs * 2
        kwargs = [kwargs] * 2
    else:
        mismatch = True

    # If not one of the above cases, force size mismatch
    kwargs = kwargs * 5 if mismatch else kwargs

    if mismatch is True:
        try:
            check_kwargs_shape(sigs, np.array(kwargs), axis)
            assert False
        except ValueError:
            assert True
    else:
        check_kwargs_shape(sigs, np.array(kwargs), axis)

    # Check case where to kwargs are passed
    check_kwargs_shape(sigs, {}, axis)
