"""Test group utility functions."""

import time
import pytest
import numpy as np

from bycycle.group.utils import progress_bar

###################################################################################################
###################################################################################################


@pytest.mark.parametrize("progress", [None, 'tqdm', pytest.param('invalid',
                                                                 marks=pytest.mark.xfail)])
def test_progress_bard(progress):

    n_iterations = 10
    iterable = [0] * n_iterations

    pbar = progress_bar(iterable, progress, n_iterations)

    assert type(pbar) == list
