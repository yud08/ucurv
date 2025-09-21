import pytest
from ucurv.meyerwavelet import *
import numpy as np
@pytest.mark.parametrize("size", [64, 96, 100, 256, 512])
def test_meyer_wavelet(size):
    """
    Test meyer_wavelet function to ensure that the sum of squares of the 
    forward and inverse filters equals 1.
    """
    f1,f2 = meyer_wavelet(size)
    eps = 1e-6
    are_close = np.all(np.isclose(f1**2+f2**2, 1, atol=eps))
    assert are_close == True

@pytest.mark.parametrize("size", [64, 96, 100, 256, 512])
def test_meyer_fwd_inv1d(size):
    """
    Test that the forward and inverse 1D Meyer wavelet transforms are consistent.
    """
    x = np.random.randn(size, size)
    for dim in [0, 1]:
        low, high = meyerfwd1d(x, dim=dim)
        x_rec = meyerinv1d(low, high, dim=dim)
        eps = 1e-6
        are_close = np.all(np.isclose(x, x_rec, atol=eps))
        assert are_close == True

@pytest.mark.parametrize("size", [64, 96, 100, 256, 512])
def test_meyer_fwd_invmd(size):
    """
    Test that the forward and inverse 2D Meyer wavelet transforms are consistent.
    """
    x = np.random.randn(size, size)
    cbands = meyerfwdmd(x)
    x_rec = meyerinvmd(cbands)
    eps = 1e-6
    are_close = np.all(np.isclose(x, x_rec, atol=eps))
    assert are_close == True
