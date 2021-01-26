"""Functions and utilities for downloading example data for bycycle."""

import os
from urllib.request import urlretrieve

import numpy as np

from neurodsp.utils.download import check_data_folder, check_data_file

###################################################################################################
###################################################################################################

DATA_URL = 'https://raw.githubusercontent.com/bycycle-tools/bycycle/main/data/'


def fetch_bycycle_data(filename, folder='data', url=DATA_URL):
    """Download a data file for bycycle.

    Parameters
    ----------
    filename : str
        Name of the data file to download.
    folder : str, optional
        Name of the folder to save the datafile to.
    url : str, optional
        The URL to download the data file from.

    Notes
    -----
    This function checks if the file already exists, and downloads it if not.
    To download the file into the local folder, set folder to an empty string ('').
    """

    check_data_folder(folder)
    check_data_file(filename, folder, url)


def load_bycycle_data(filename, folder='data', url=DATA_URL):
    """Download, if not already available, and load an example data file for bycycle.

    Parameters
    ----------
    filename : str
        Name of the data file to download.
    folder : str, optional
        Name of the folder to save the datafile to.
    url : str, optional
        The URL to download the data file from.

    Returns
    -------
    data : ndarray
        Loaded data file.

    Notes
    -----
    This function assumes that data files are numpy (npy) files.
    """

    fetch_bycycle_data(filename, folder, url)
    data = np.load(os.path.join(folder, filename))

    return data
