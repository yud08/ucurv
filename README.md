# UCURV Uniform Discrete curvelet transforms.

This package implements the Uniform Discrete Curvelet Transform as described in "Uniform discrete curvelet transform" TT Nguyen, H Chauris - IEEE transactions on signal processing, 2010, following the suggestion in https://github.com/PyLops/pylops/wiki/GSoC-2023-Project-Ideas

# What are the curvelet transforms?
https://en.wikipedia.org/wiki/Curvelet

You construct a Udct object by providing the shape of the data to be transformed, and then the configuration for the transform, i.e how many directional wedges you wish to have at each axis and each scale.
Provides a forwards and backwards version of the transform. The backward inverses the forward transform.

# Example usage
```python
from ucurv import *
from ucurv.zoneplate import *
import matplotlib.pyplot as plt
sz = [512, 512]
cfg = [[3, 3], [6,3], [12, 6]]
res = len(cfg)
rsq = zoneplate(sz)
img = rsq - np.mean(rsq)

udct = Udct(sz, cfg, complex = True)

#forward transform
imband = ucurvfwd(img, udct)
#backwards transform
recon = ucurvinv(imband, udct)

err = img - recon
print(np.max(np.abs(err)))

```

# Installation guide
```bash
pip install ucurv
```

# Running tests
We use [pytest](https://docs.pytest.org/) for unit testing. 
To run tests, install Pytest in your project root:
```bash
pip install -e .[dev]
```
and then call(from the project root):

```bash
pytest
```