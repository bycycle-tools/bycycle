"""Utilities for burst detection."""

###################################################################################################
###################################################################################################

def check_min_burst_cycles(is_burst, min_n_cycles=3):
    """Enforce minimum number of consecutive cycles to be considered a burst.

    Parameters
    ----------
    is_burst : 1d array
        Boolean array indicating which cycles are bursting.
    min_n_cycles : int, optional, default: 3
        The minimum number of cycles of consecutive cycles required to be considered a burst.

    Returns
    -------
    is_burst : 1d array
        Updated burst array.
    """

    temp_cycle_count = 0

    for idx, bursting in enumerate(is_burst):

        if bursting:
            temp_cycle_count += 1

        else:

            if temp_cycle_count < min_n_cycles:
                for c_rm in range(temp_cycle_count):
                    is_burst[idx - 1 - c_rm] = False

            temp_cycle_count = 0

    return is_burst