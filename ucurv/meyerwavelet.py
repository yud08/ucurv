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
    step = 2*np.pi/N
    x = np.linspace(0,2*np.pi - step, N) - np.pi/2
    prm = np.pi*np.array([-1/3, 1/3 , 2/3, 4/3])
    f1 = np.sqrt( np.fft.fftshift(fun_meyer(x, prm)) )
    f2 = np.sqrt( fun_meyer(x, prm) )
    return f1, f2


def meyerfwd1d(img, dim):
    """
    Perform a 1D Meyer wavelet forward transform along a specified axis.

    This applies the Meyer filters in the frequency domain, then
    splits and downsamples into low and high frequency subbands.

    Parameters
    ----------
    img : ndarray
        Input array of arbitrary shape.  The transform is applied along
        one axis, and all other dimensions are preserved.
    dim : int
        The axis (0 ≤ `dim` < `img.ndim`) along which to compute the
        1D Meyer transform.

    Returns
    -------
    h1 : ndarray
        The low# frequency subband.  This has the same shape as `img`
        except along `dim`, where its length is `N/2` (every even sample).
    h2 : ndarray
        The high frequency subband.  Same shape as `h1`, but containing
        the odd indexed samples after filtering.

    Notes
    -----
    Internally this function:
      1. Swaps axis `dim` with the last axis for convenience.
      2. Computes the length `N` Meyer wavelet filters `f1, f2`.
      3. FFTs the input along that last axis.
      4. Multiplies by `f1`/`f2`, IFFTs back to time domain.
      5. Takes real part and downsamples by 2:
         - `h1` takes the even indices (`[..., ::2]`),
         - `h2` takes the odd indices (`[..., 1::2]`).
      6. Swaps the axes back to restore the original layout.

    Examples
    --------
    import numpy as np
    x = np.random.randn(100, 200)
    # transform along axis=1 (columns)
    low, high = meyerfwd1d(x, dim=1)
    low.shape, high.shape
    ((100, 100), (100, 100))
    """
    ldim = img.ndim - 1
    img = np.swapaxes(img, dim, ldim)
    sp = img.shape
    N = sp[-1]
    f1, f2 = meyer_wavelet(N)
    f1 = np.reshape(f1, (1, N))
    f2 = np.reshape(f2, (1, N))

    imgf = np.fft.fft(img, axis = ldim)
    h1 = np.real(np.fft.ifft(f1*imgf, axis = ldim))[...,::2]
    h2 = np.real(np.fft.ifft(f2*imgf, axis = ldim))[...,1::2]
    h1 = np.swapaxes(h1, dim, ldim)
    h2 = np.swapaxes(h2, dim, ldim)

    return h1, h2

def meyerinv1d(h1, h2, dim):
    """
    Perform the 1D inverse Meyer wavelet transform along a specified axis.

    Reconstructs the original signal by interleaving low and high frequency
    subbands, applying Meyer filters in the frequency domain, and summing.

    Parameters
    ----------
    h1 : ndarray
        Low frequency subband array.  Its shape matches the original image
        except along axis `dim`, where its length is N/2.
    h2 : ndarray
        High frequency subband array, same shape as `h1`.
    dim : int
        The axis (0 ≤ `dim` < `h1.ndim`) along which the forward transform
        was applied and now is inverted.

    Returns
    -------
    imrecon : ndarray
        The reconstructed array, of the same shape as the original input
        to `meyerfwd1d`.

    Notes
    -----
    Internally this function:
      1. Swaps axis `dim` with the last axis for convenience.
      2. Creates arrays `g1` and `g2` of length N by placing `h1` into even
         indices (`[..., ::2]`) and `h2` into odd indices (`[..., 1::2]`).
      3. Computes Meyer wavelet filters `f1, f2` of length N.
      4. Transforms `g1` and `g2` via FFT along that axis, multiplies by
         `f1`/`f2`, and sums in the frequency domain.
      5. Applies the inverse FFT, takes the real part, and scales by 2.
      6. Swaps axes back to restore the original array layout.

    Examples
    --------
    import numpy as np
    x = np.random.randn(64, 128)
    low, high = meyerfwd1d(x, dim=1)
    x_rec = meyerinv1d(low, high, dim=1)
    np.allclose(x, x_rec, atol=1e-6)
    True
    """
    ldim = h1.ndim - 1
    h1 = np.swapaxes(h1, dim, ldim)
    h2 = np.swapaxes(h2, dim, ldim)

    sp = list(h1.shape)
    sp[-1] = 2*sp[-1]
    g1 = np.zeros(sp)
    g2 = np.zeros(sp)
    g1[...,::2] = h1
    g2[...,1::2] = h2
    N = sp[-1]
    f1, f2 = meyer_wavelet(N)
    f1 = np.reshape(f1, (1, N))
    f2 = np.reshape(f2, (1, N))
    imfsum = f1*np.fft.fft(g1, axis = ldim) + f2*np.fft.fft(g2, axis = ldim)
    imrecon = 2*np.real(np.fft.ifft(imfsum, axis = ldim))
    imrecon = np.swapaxes(imrecon, dim, ldim)
    return imrecon

def meyerfwdmd(img):
    """
    Perform an N-dimensional forward Meyer wavelet transform.

    Applies 1D Meyer forward filters along each axis of the input array,
    successively splitting into low- and high-frequency subbands.

    Parameters
    ----------
    img : ndarray
        Input N-dimensional array to be transformed.  The transform is applied
        along each axis in turn, starting from axis 0 up to axis N-1.

    Returns
    -------
    cband : list of ndarray
        A list of length 2**N of subband arrays.  Each level of decomposition
        doubles the number of subbands by splitting each existing band into
        its low- and high-frequency components.

    Notes
    -----
    - The first call splits the input into [h1, h2] along axis 0.
    - On each subsequent axis `i`, every current subband is further split
      by `meyerfwd1d` along axis `i`.

    Examples
    --------
    import numpy as np
    x = np.random.randn(8, 8, 8)
    subbands = meyerfwdmd(x)
    len(subbands)
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
    Perform the inverse N-dimensional Meyer wavelet transform.

    Reconstructs the original N-dimensional array from its 2**N subbands
    by iteratively merging low- and high-frequency components along each axis.

    Parameters
    ----------
    band : list of ndarray
        List of 2**N subband arrays produced by `meyerfwdmd`.  The order must
        match the output of `meyerfwdmd` for the same input dimensions.

    Returns
    -------
    img_recon : ndarray
        The reconstructed N-dimensional array, matching the shape of the original
        input passed to `meyerfwdmd`.

    Notes
    -----
    - Reconstruction proceeds in reverse axis order: starting from axis N-1
      back down to axis 0.
    - At each axis `i`, subbands are paired (low, high) and merged via `meyerinv1d`.

    Examples
    --------
    import numpy as np
    x = np.random.randn(8, 8, 8)
    subbands = meyerfwdmd(x)
    x_rec = meyerinvmd(subbands)
    np.allclose(x, x_rec, atol=1e-6)
    """
    dim = len(band[0].shape)
    for i in range(dim-1, -1, -1):
        cband = []
        for j in range(len(band)//2):
            imrecon = meyerinv1d( band[2*j] , band[2*j+1], i)
            cband.append(imrecon)
        band = cband
    return band[0]
