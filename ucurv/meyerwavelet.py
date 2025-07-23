from .backend import ncp as _ncp_func
ncp = _ncp_func() 
import numpy as np
from .util import fun_meyer

def meyer_wavelet(N):
    """
    Generate forward and inverse 1D Meyer wavelet filters of length N.

    Parameters
    ----------
    N : int
        Number of samples (length of the wavelet filters).

    Returns
    -------
    f1 : ndarray
        The forward Meyer wavelet filter, computed as the square root of
        the FFT-shifted Meyer window on a grid of length N.
    f2 : ndarray
        The inverse Meyer wavelet filter, computed as the square root of
        the unshifted Meyer window on the same grid.

    Notes
    -----
    This function constructs a spatial grid
        x = linspace(0, 2π, N, endpoint=False) - π/2
    and parameters
        prm = π * [-1/3, 1/3, 2/3, 4/3]
    It then calls `fun_meyer(x, prm)` to generate the Meyer window,
    applies FFT-shift for the forward filter, and takes the square roots.
    """
    step = 2*ncp.pi/N
    x = ncp.linspace(0,2*ncp.pi - step, N) - ncp.pi/2
    prm = ncp.pi*ncp.array([-1/3, 1/3 , 2/3, 4/3])
    f1 = ncp.sqrt( ncp.fft.fftshift(fun_meyer(x, prm)) )
    f2 = ncp.sqrt( fun_meyer(x, prm) )
    return f1, f2


def meyerfwd1d(img, dim):
    """
    Perform a 1-D Meyer wavelet forward transform along a specified axis.

    This applies Meyer filters in the frequency domain, then splits and
    downsamples into low and high-frequency sub-bands.

    Parameters
    ----------
    img : ndarray
        Incput array of arbitrary shape.  The transform is applied along
        one axis; all other dimensions are preserved.
    dim : int
        Axis (0 ≤ ``dim`` < ``img.ndim``) along which to compute the
        1-D Meyer transform.

    Returns
    -------
    h1 : ndarray
        The **low-frequency** sub-band.  Same shape as *img* except
        along *dim*, where the length is ``N/2`` (even samples).
    h2 : ndarray
        The high frequency sub band (odd samples); shape identical to *h1*.

    Notes
    -----
    Internally this function:

    1. Swaps axis *dim* with the last axis.
    2. Computes length ``N`` Meyer filters ``f1`` and ``f2``.
    3. FFTs the incput along that axis.
    4. Multiplies by ``f1``/``f2`` and IFFTs back.
    5. Takes the real part and downsamples by 2:
       ``h1 = [..., ::2]`` (even), ``h2 = [..., 1::2]`` (odd).
    6. Swaps axes back to restore the original layout.

    Examples
    --------
    >>> import numpy as ncp
    >>> x = ncp.random.randn(100, 200)
    >>> low, high = meyerfwd1d(x, dim=1)
    >>> low.shape, high.shape
    ((100, 100), (100, 100))
    """
    ldim = img.ndim - 1
    img = ncp.swapaxes(img, dim, ldim)
    sp = img.shape
    N = sp[-1]
    f1, f2 = meyer_wavelet(N)
    f1 = ncp.reshape(f1, (1, N))
    f2 = ncp.reshape(f2, (1, N))

    imgf = ncp.fft.fft(img, axis = ldim)
    h1 = ncp.real(ncp.fft.ifft(f1*imgf, axis = ldim))[...,::2]
    h2 = ncp.real(ncp.fft.ifft(f2*imgf, axis = ldim))[...,1::2]
    h1 = ncp.swapaxes(h1, dim, ldim)
    h2 = ncp.swapaxes(h2, dim, ldim)

    return h1, h2

def meyerinv1d(h1, h2, dim):
    """
    Inverse 1-D Meyer wavelet transform along a chosen axis.

    The routine reconstructs the original signal by interleaving the
    low- and high-frequency sub-bands, applying the Meyer synthesis
    filters in the frequency domain, and summing the results.

    Parameters
    ----------
    h1 : ndarray
        Low-frequency sub-band.  Same shape as the original incput except
        along *dim*, where its length is ``N/2`` (even samples).
    h2 : ndarray
        High-frequency sub-band; shape identical to *h1* (odd samples).
    dim : int
        Axis along which the forward transform was taken
        (``0 <= dim < h1.ndim``).

    Returns
    -------
    imrecon : ndarray
        Reconstructed array with the same shape and ``dtype`` as the
        original incput to :func:`meyerfwd1d`.

    Notes
    -----
    Internally the function proceeds as follows:

    1. Swap axis *dim* with the last axis.
    2. Build two length-``N`` arrays ``g1`` and ``g2`` by placing
       ``h1`` in the even indices (``[..., ::2]``) and ``h2`` in the
       odd indices (``[..., 1::2]``).
    3. Compute Meyer synthesis filters ``f1`` and ``f2`` of length ``N``.
    4. FFT ``g1`` and ``g2`` along the last axis, multiply by
       ``f1``/``f2``, and sum in the frequency domain.
    5. Apply the inverse FFT, take the real part, and scale by 2.
    6. Swap axes back to restore the original layout.

    Examples
    --------
    >>> import numpy as ncp
    >>> x = ncp.random.randn(64, 128)
    >>> low, high = meyerfwd1d(x, dim=1)
    >>> x_rec = meyerinv1d(low, high, dim=1)
    >>> ncp.allclose(x, x_rec, atol=1e-6)
    True
    """
    ldim = h1.ndim - 1
    h1 = ncp.swapaxes(h1, dim, ldim)
    h2 = ncp.swapaxes(h2, dim, ldim)

    sp = list(h1.shape)
    sp[-1] = 2*sp[-1]
    g1 = ncp.zeros(sp)
    g2 = ncp.zeros(sp)
    g1[...,::2] = h1
    g2[...,1::2] = h2
    N = sp[-1]
    f1, f2 = meyer_wavelet(N)
    f1 = ncp.reshape(f1, (1, N))
    f2 = ncp.reshape(f2, (1, N))
    imfsum = f1*ncp.fft.fft(g1, axis = ldim) + f2*ncp.fft.fft(g2, axis = ldim)
    imrecon = 2*ncp.real(ncp.fft.ifft(imfsum, axis = ldim))
    imrecon = ncp.swapaxes(imrecon, dim, ldim)
    return imrecon

def meyerfwdmd(img):
    """
    N-dimensional forward Meyer wavelet transform.

    The routine applies the 1-D forward Meyer filters successively along
    each axis of the incput array, splitting the data into low- and
    high-frequency sub-bands at every step.

    Parameters
    ----------
    img : ndarray
        Incput array of arbitrary dimensionality.  The transform is applied
        along axis 0, then axis 1, and so on up to ``img.ndim - 1``.

    Returns
    -------
    cband : list of ndarray
        List of length ``2**N`` containing the sub-band arrays, where
        ``N = img.ndim``.  Each decomposition level doubles the number of
        bands by splitting every current band into its low- and
        high-frequency components.

    Notes
    -----
    * **Axis 0** - the first call to :func:`meyerfwd1d` splits *img* into
      ``[h1, h2]``.
    * **Axis i ( i ≥ 1 )** - every existing band is split again along
      axis *i*, so after processing axis *i* the total number of bands
      is ``2**(i + 1)``.

    Examples
    --------
    >>> import numpy as ncp
    >>> x = ncp.random.randn(8, 8, 8)   # 3-D array (N = 3)
    >>> subbands = meyerfwdmd(x)
    >>> len(subbands)
    8
    """
    band = [img]
    dim = len(img.shape)
    for i in range(dim):
        cband = []
        for j in range(len(band)):
            h1 , h2  = meyerfwd1d(band[j], i)
            cband.append(h1)
            cband.append(h2)
        band = cband
    return cband    

def meyerinvmd(band):
    """
    Inverse *N*-dimensional Meyer wavelet transform.

    Reconstruct the original array from the ``2**N`` sub-bands returned by
    :func:`meyerfwdmd` by successively merging low- and high-frequency
    components along each axis.

    Parameters
    ----------
    band : list[numpy.ndarray]
        Sequence of ``2**N`` sub-band arrays produced by
        :func:`meyerfwdmd`.  The order **must** match exactly the order
        returned by that function for the same incput dimensions.

    Returns
    -------
    img_recon : numpy.ndarray
        Array with the same shape and ``dtype`` as the data that was
        passed to :func:`meyerfwdmd`.

    Notes
    -----
    The reconstruction proceeds in **reverse axis order**:

    1. Start with axis ``N - 1``; pair each low/high band and merge them
       with :func:`meyerinv1d` along that axis, halving the number of
       bands.
    2. Repeat the pairing-and-merge step for the next smaller axis.
    3. Continue until only one band remains—the fully reconstructed
       signal.

    Examples
    --------
    >>> import numpy as ncp
    >>> x = ncp.random.randn(8, 8, 8)
    >>> subbands = meyerfwdmd(x)
    >>> x_rec = meyerinvmd(subbands)
    >>> ncp.allclose(x, x_rec, atol=1e-6)
    True
    """
    dim = len(band[0].shape)
    for i in range(dim-1, -1, -1):
        cband = []
        for j in range(len(band)//2):
            imrecon = meyerinv1d( band[2*j] , band[2*j+1], i)
            cband.append(imrecon)
        band = cband
    return band[0]
