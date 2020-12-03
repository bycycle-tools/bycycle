"""Test group utility functions."""

import pytest
import numpy as np

from bycycle.group.utils import progress_bar, check_kwargs_shape

###################################################################################################
###################################################################################################

@pytest.mark.parametrize("progress", [None, 'tqdm', pytest.param('invalid',
                                                                 marks=pytest.mark.xfail)])
def test_progress_bar(progress):

    n_iterations = 10
    iterable = [0] * n_iterations

    pbar = progress_bar(iterable, progress, n_iterations)

    assert len(pbar) == n_iterations


@pytest.mark.parametrize("axis", [0, 1, 2, None, pytest.param(2, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("kwargs_ndim", [1, 2])
@pytest.mark.parametrize("sigs_ndim", [2, 3])
@pytest.mark.parametrize("mismatch", [True, False])
def test_check_kwargs_shape(sim_args, axis, kwargs_ndim, sigs_ndim, mismatch):

    sigs = np.array([sim_args['sig']] * 2)
    if sigs_ndim == 3:
        sigs = np.array([sigs] * 2)

    kwargs = [{'center_extrema': 'peak'}]

    # 2D cases that will pass
    if (sigs_ndim == 2 and axis == 1 and kwargs_ndim == 1) or \
       (sigs_ndim == 2 and axis == None and kwargs_ndim == 1):
        kwargs = kwargs * 2
    # 3D cases that will pass
    elif (sigs_ndim == 3 and axis == 0 and kwargs_ndim == 1) or \
         (sigs_ndim == 3 and axis == 1 and kwargs_ndim == 1):
        kwargs =  kwargs * 2
    elif sigs_ndim == 3 and axis == 2 and kwargs_ndim == 2:
        kwargs = [kwargs * 2] * 2
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
