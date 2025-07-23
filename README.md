# UCURV: Uniform Discrete curvelet transform.

This package implements the Uniform Discrete Curvelet Transform as described in "Uniform discrete curvelet transform" TT Nguyen, H Chauris - IEEE transactions on signal processing, 2010, following the suggestion in https://github.com/PyLops/pylops/wiki/GSoC-2023-Project-Ideas

# What is a curvelet transform?
A curvelet is a mathematical building‑block, much like a wavelet but stretched, and directional, so it has an "orientation" that better represents images/signals which contain curved edges. Curvelets are multiscale, so come in many sizes, to capture both coarse and fine details. At each scale there are also many orientations, letting curvelets follow curves instead of just horizontal/vertical features. Curvelets also have anisotropy: as the scale becomes finer, their length shrinks roughly with the square root of the scale while the width shrinks as linearly, matching the geometry of smooth curves.

Because of those properties, a curvelet transform can represent objects with smooth contours using far fewer coefficients than standard wavelets, making it suitable for image denoising, compression, and edge‑aware signal analysis.

https://en.wikipedia.org/wiki/Curvelet
https://towardsdatascience.com/desmystifying-curvelets-c6d88faba0bf/
https://www.youtube.com/watch?v=jnxqHcObNK4&ab_channel=ArtemKirsanov

# Example usage
The package is implemented with a Udct object, which you first must construct, providing the shape of the data to be transformed, and then the configuration for the transform, i.e how many directional wedges you wish to have at each axis and each scale. Then to apply the forwards and backwards versions of the transform, these methods must be called with the data and the Udct object. The backward perfectly inverses the forward transform. We demonstrate this below:

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

# Documentation
Access the documentation at https://ucurv.readthedocs.io/en/latest/.

# Creating your own documentation with Sphinx
Install Sphinx in your project root(same as above):
```bash
pip install -e .[dev]
```
and then make the documentation with:

```bash
make clean html
```

Access it from `docs\build\html\index.html`.
If you wish to view it, either use the Live Server extension, or alternatively call:

```bash
python -m http.server --directory docs/build/html 8000
```
and go onto:
http://localhost:8000/ 