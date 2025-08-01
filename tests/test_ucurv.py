#Tests the main ucurv object creation and forward and backwards call

from ucurv.backend import ncp as _ncp_func
ncp = _ncp_func() 

import pytest
import ucurv
import numpy as np
from itertools import product
from . import eps, combinations

@pytest.mark.parametrize( #test with defaults
    "complex_, sparse, high",
    list(product([False], [False], ["wavelet"]))
)
# @pytest.mark.parametrize( # test with all combinations
#     "complex_, sparse, high",
#     list(product([False, True], [False, True], ["wavelet", "curvelet"]))
# )
@pytest.mark.parametrize("shape, cfg", combinations)
def test_ucurv(shape, cfg, complex_, sparse, high):
    """
    Tests that the forward transform is perfectly inversed by the backwards transform.
    Calls the forward and backwards on random data of multiple different shapes and multiple different configurations for the transform.
    Shapes and configurations are listed above, and all possible pairs are combined together in combinations.
    """
    data = np.random.rand(*shape)
    Udct = ucurv.Udct(shape, cfg, complex_, sparse, high)
    band = ucurv.ucurvfwd(data, Udct)
    recon = ucurv.ucurvinv(band, Udct)
    are_close = np.all(np.isclose(data, recon, atol=eps))
    assert(are_close == True)

