"""Test the persistence of computed features across versions of bycycle."""

from hashlib import sha1
import numpy as np

###################################################################################################
###################################################################################################

# Update when a commit alters the result of compute_features
PERSISTENT_HASH = 'de3842a725c996cfeb6eee293703179d72e954e7'

def test_persistent_features(sim_args_comb):

    # Get df from pytest fixture
    df_features = sim_args_comb['df_features']

    # Convert to np array
    arr_features = df_features.to_numpy().astype(np.float32).round(8).copy(order='C')

    # Hash
    h = sha1()
    h.update(arr_features)
    arr_hash = h.hexdigest()

    assert arr_hash == PERSISTENT_HASH
