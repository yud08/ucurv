.. UDCT documentation master file, created by
   sphinx-quickstart on Mon Jul 21 16:14:14 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

UCURV documentation
==================

Welcome to the UDCT Documentation!

.. rubric:: UCURV: Uniform Discrete curvelet transform.

This package implements the Uniform Discrete Curvelet Transform as described in "Uniform discrete curvelet transform" TT Nguyen, H Chauris - IEEE transactions on signal processing, 2010, following the suggestion in https://github.com/PyLops/pylops/wiki/GSoC-2023-Project-Ideas

.. rubric:: What is a curvelet transform?
A curvelet is a mathematical building block, much like a wavelet but stretched, and directional, so it has an "orientation" that better represents images/signals which contain curved edges. Curvelets are multiscale, so come in many sizes, to capture both coarse and fine details. At each scale there are also many orientations, letting curvelets follow curves instead of just horizontal/vertical features. Curvelets also have anisotropy: as the scale becomes finer, their length shrinks roughly with the square root of the scale while the width shrinks as linearly, matching the geometry of smooth curves.
Because of those properties, a curvelet transform can represent objects with smooth contours using far fewer coefficients than standard wavelets, making it suitable for image denoising, compression, and edge‑aware signal analysis.

https://en.wikipedia.org/wiki/Curvelet
https://towardsdatascience.com/desmystifying-curvelets-c6d88faba0bf/
https://www.youtube.com/watch?v=jnxqHcObNK4&ab_channel=ArtemKirsanov

.. rubric:: Comparison with the matlab toolbox
This package is developed from the UDCT matlab toolbox available at https://www.mathworks.com/matlabcentral/fileexchange/50948-uniform-discrete-curvelet-transform by the orignal author of the paper.
As the first step, the main algorithms have been translated from matlab to python, and the results have been verified to be identical to those of the matlab toolbox. 
The matlab toolbox is not available under an open source license, so this package is released under the MIT license.
Some of the improvements made in this package include: (some are still in progress and not yet available)
- Support for complex-valued inputs and outputs.
- Verifiable multidimensional support (2D, 3D, etc.)
- Using Meyer wavelet transform at high resolution to reduce redundancy.
- Adapt to a linear operator to be used in Pylops toolbox

.. rubric:: Example Usage

.. code-block:: python

   from ucurv import *
   from ucurv.zoneplate import *
   import matplotlib.pyplot as plt

   sz = [512, 512]
   cfg = [[3, 3], [6,3], [12, 6]]
   res = len(cfg)
   rsq = zoneplate(sz)
   img = rsq - np.mean(rsq)

   udct = Udct(sz, cfg, complex = True)

   # forward transform
   imband = ucurvfwd(img, udct)
   # backward (perfect inverse)
   recon = ucurvinv(imband, udct)

   err = img - recon
   print(np.max(np.abs(err)))

Contents
--------

.. toctree::
   :maxdepth: 2
   gettingstarted
   :caption: Modules
   ucurv.backend
   ucurv.meyerwavelet
   ucurv.ucurv
   ucurv.util
   ucurv.zoneplate
