import pytest
import ucurv
import numpy as np

eps = 1e-6
shapes = [
    [[256, 256], ],
    [[32, 32, 32], ],
    [[16, 16, 16, 16], ]
]

configurations = [
    [[[3, 3]], 
     [[6, 6]],
     [[12, 12]],
     [[12, 12], [24, 24]],
     [[12, 12], [3, 3], [6, 6]],
     [[12, 12], [3, 3], [6, 6], [24, 24]], 
    ], 
    [[[3, 3, 3]], 
     [[6, 6, 6]],
     [[12, 12, 12]],
     [[12, 12, 12], [24, 24, 24]],
    # [[12, 12, 12], [3, 3, 3], [6, 6, 6]],
    # [[12, 12, 12], [3, 3, 3], [6, 6, 6], [12, 24, 24]], 
    ], 
    [[[3, 3, 3, 3]], 
    #  [[6, 6, 6, 6]],
    #  [[12, 12, 12, 12]],
    #  [[12, 12, 12, 12], [24, 24, 24, 24]],
    #  [[12, 12, 12, 12], [3, 3, 3, 3], [6, 6, 6, 6]],
    #  [[12, 12, 12, 12], [3, 3, 3, 3], [6, 6, 6, 6], [12, 24, 24, 24]], 
    ], 
]

combinations = [
    (shape, config) 
    for shape_list, config_list in zip(shapes, configurations) 
    for shape in shape_list 
    for config in config_list
]

@pytest.mark.parametrize("shape, cfg", combinations)
def test_ucurv(shape, cfg):
    data = np.random.rand(*shape)
    Udct = ucurv.Udct(shape, cfg)
    band = ucurv.ucurvfwd(data, Udct)
    recon = ucurv.ucurvinv(band, Udct)
    are_close = np.all(np.isclose(data, recon, atol=eps))
    assert(are_close == True)

@pytest.mark.parametrize("shape, cfg", combinations)
def test_vectorize(shape, cfg):
    data = np.random.rand(*shape)
    Udct = ucurv.Udct(shape, cfg)
    band = ucurv.ucurvfwd(data, Udct)
    flat = ucurv.bands2vec(band)
    unflat = ucurv.vec2bands(flat, Udct)
    recon = ucurv.ucurvinv(unflat, Udct)
    are_close = np.all(np.isclose(data, recon, atol=eps))
    assert(are_close == True)