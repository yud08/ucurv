#tests all the different functions
from ucurv.backend import ncp as _ncp_func
ncp = _ncp_func() 

def test_array_backend():
    """
    Tests that backend is working correctly, and behaves like Numpy for basic operations.
    """
    arr = ncp.ones((2, 2))
    assert arr.sum() == 4

import pytest
import ucurv
import numpy as np
from . import eps, combinations

@pytest.mark.parametrize("shape, cfg", combinations)
def test_vectorize(shape, cfg):
    """
    Tests that calling bands2vec, which flattens the dictionary of subbands into a single array, and then vec2bands, which undoes this, reverse each other perfectly.
    """
    data = np.random.rand(*shape)
    Udct = ucurv.Udct(shape, cfg)
    band = ucurv.ucurvfwd(data, Udct)
    flat = ucurv.bands2vec(band)
    unflat = ucurv.vec2bands(flat, Udct)
    recon = ucurv.ucurvinv(unflat, Udct)
    are_close = np.all(np.isclose(data, recon, atol=eps))
    assert(are_close == True)

def test_fun_meyer():
    """
    Tests that calling fun_meyer with a range of incorrect param data will correctly raise an exception
    """
    with pytest.raises(Exception):
        ucurv.fun_meyer([], [1, 3, 4])
    with pytest.raises(Exception):
        ucurv.fun_meyer([], [0, 2, 4, 5, 6])
    with pytest.raises(Exception):
        ucurv.fun_meyer([], [3, 1, 4, 2]) 
    with pytest.raises(Exception):
        ucurv.fun_meyer([], [1, 2, 4, 3.5]) 
    with pytest.raises(Exception):
        ucurv.fun_meyer([], [5.6, 3.5, 7, 8]) 