from .util import *
from .meyerwavelet import *
from .ucurv import *
from .zoneplate import *

"""
This package implements the Uniform Discrete Curvelet Transform as described in "Uniform discrete curvelet transform", TT Nguyen, H Chauris - IEEE transactions on signal processing, 2010.
You construct a Udct object by providing the shape of the data to be transformed, and then the configuration for the transform, i.e how many directional wedges you wish to have at each axis and each scale.
The forwards and backwards version of the transform are ucurvfwd and ucurvinv, and must be passed with the data, and the Udct object for that data. The backward inverses the forward transform.
Taken from the README.
"""

# _version is only created upon package creation
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0+unknown"