from .backend import ncp as _ncp_func
ncp = _ncp_func() 
import math
import numpy as np
from .util import fun_meyer
from .meyerwavelet import meyerfwdmd, meyerinvmd

def generate_combinations(dim):
    """
    Generate all the ways to choose dim - 1 elements from (0, 1, 2, ... dim - 1), 
    i.e all the combinations which exclude one index

    Parameters
    ----------
    dim : int
        The number of elements (defines the index set 0, 1, …, dim - 1).

    Returns
    -------
    List[tuple]
        A list of tuples, each of length `dim-1`, representing all ways to
        omit exactly one index from the full range. 

    Examples
    --------
    generate_combinations(4)
    [(0, 1, 2),  (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    """
    lst = []
    for i in range(dim):
        combi = tuple(j for j in range(dim) if j != i)
        lst.append(combi)
    return lst[::-1] #needs to be reversed

def tan_theta_grid(S1, S2):
    """
    Create a grid approximate the tan theta function as described in (28) of the paper
    """
    # first grid
    x1, x2 = ncp.meshgrid(S2, S1)

    t1 = ncp.zeros_like(x1)
    ind = ncp.logical_and(x2 != 0, ncp.abs(x1) <= ncp.abs(x2))
    t1[ind] = -x1[ind] / x2[ind]

    t2 = ncp.zeros_like(x1)
    ind = ncp.logical_and(x1 != 0, ncp.abs(x2) < ncp.abs(x1))
    t2[ind] = x2[ind] / x1[ind]
    t3 = t2.copy()
    t3[t2 < 0] = t2[t2 < 0] + 2
    t3[t2 > 0] = t2[t2 > 0] - 2

    M2 = t1 + t3
    M2[x2 >= 0] = -2

    return M2

def fftflip(F, dirlist = None):
    """
    Flip and circularly shift an N-D FFT array so that frequency sign is reversed.

    This routine performs an axis-wise reversal and roll on the incput array
    to map X(\\omega) to X(-\\omega) in its FFT representation.  For each
    transformed axis, elements are flipped (so that the zero-frequency
    component moves to the end), then rolled by one to restore the zero-frequency
    component to the first position.

    Parameters
    ----------
    F : ndarray
        Incput array in the frequency domain (FFT output).  Can be of any dimensionality.
    dirlist : int or sequence of ints, optional
        Axis or list of axes over which to apply the fftflip.  If None (default), all
        axes of `F` will be processed.

    Returns
    -------
    Fc : ndarray
        A new array of the same shape as `F`, with each specified axis flipped
        and rolled so that the frequency axis is negated.

    Notes
    -----
    - Reversing an axis in FFT output corresponds to replacing \\omega with -\\omega.
    - After `ncp.flip`, the zero-frequency component moves to the end of the axis.
      `ncp.roll` by +1 brings it back to index 0.
    - For multi-dimensional FFTs, reversing multiple axes implements sign
      inversion in each frequency dimension.
    """
    Fc = F.copy()
    dim = Fc.ndim
    if dirlist is None:
        dirlist = list(range(dim))
    shiftvec = ncp.zeros(dim)
    if type(dirlist) is list:
        for dir in dirlist:
            shiftvec[dir] = 1
            Fc = ncp.flip(Fc, dir)
            Fc = ncp.roll(Fc, 1, axis=dir)
    if type(dirlist) is int:
        dir = dirlist
        shiftvec[dir] = 1
        Fc = ncp.flip(Fc, dir)
        Fc = ncp.roll(Fc, 1, axis=dir)
    return Fc

def angle_fun(Mgrid, n, alpha, dir, bandpass = None):
    """
    Return the angle meyer window as in Figure 8 of the paper
    Compute directional Meyer window functions for angular decomposition.
    Parameters
    ----------
    Mgrid : ndarray
        An N-D coordinate grid (e.g., meshgrid) of angles normalized to [−1,1].
    n : int
        Total number of angular directions (must be positive and even).
    alpha : float
        Angular transition parameter controlling the width of each subband.
    dir : int or sequence of int
        Axis or axes along which to apply `fftflip` for the symmetric counterpart.
    bandpass : ndarray, optional
        An array of the same shape as `Mgrid` to modulate (mask) each window.

    Returns
    -------
    Mang : list of ndarray
        A list of length `n` containing the forward (`0 <= id < n/2`) and
        flipped inverse (`n/2 <= id < n`) Meyer windows for each angular sector.
    """
    angd = 2/n
    ang = angd*ncp.array([-alpha, alpha, 1-alpha, 1+alpha])
    Mang = [[] for i in range(n) ]
    tmp = []
    sp = ncp.array(Mgrid.shape)
    for id in range(math.ceil(n/2)):
        ang2 = -1 + id*angd + ang
        x = fun_meyer(Mgrid, ang2)
        if bandpass is not None:
            x = x*bandpass
        x = ncp.roll(x, 3*sp//4, (0,1) )
        Mang[id] = x.copy()
        Mang[n-1-id] = fftflip(x, dir)
        
    return Mang

def angle_kron(F, dr, sz):
    """
    This function replicate the 2D array F at dimension dr to match the size sz

    Constructs a multi-dimensional array by taking the Kronecker product of the
    flattened incput `F` with an all-ones array, then reshaping and moving axes
    so that the original 2D data appear along dimensions `dr` within the final
    shape `sz`.

    Parameters
    ----------
    F : ndarray
        A 2D incput array of shape (m, n) to be replicated.
    dr : tuple of int
        Length-2 tuple specifying the target axes in the output array where
        the original dimensions of `F` should be placed.
    sz : sequence of int
        Desired shape of the output array.  The product of all entries of `sz`
        must be an integer multiple of the product of `F.shape`.

    Returns
    -------
    Fk : ndarray
        Array of shape `sz` containing repeated copies of `F` along all axes
        not in `dr`, with the original 2D layout inserted at positions `dr`.

    Raises
    ------
    ValueError
        If `ncp.prod(sz)` is not divisible by `ncp.prod(F.shape)`.
    """
    sp = list(F.shape)
    sp2 = int(ncp.prod(ncp.array(sz))/ncp.prod(ncp.array(F.shape)))
    # remove dr element from sz
    sz2 = list(sz)
    sz2 = [i for j, i in enumerate(sz) if j not in dr]
    # now append sz2 to sp
    sp = sp + sz2
    # Fk is the replicated version of F, along the dimension other than dr[0,1]
    Fk = ncp.reshape(ncp.kron(F.flatten(), ncp.ones( sp2 )),  sp)

    # Move the dimensions 0,1 to dr
    Fk = ncp.moveaxis(Fk, [0 ,1] , dr)
    return Fk

def downsamp(band, samp, shift = None):
    """
    Downsample a N-D array by length N of power-2 integers 
    """
    if shift is None:
        shift = ncp.zeros(len(band.shape), dtype = int)
    if len(samp) == 2:
        return band[shift[0]::samp[0], shift[1]::samp[1]]
    if len(samp) == 3:
        return band[shift[0]::samp[0], shift[1]::samp[1], shift[2]::samp[2]]
    if len(samp) == 4:
        return band[shift[0]::samp[0], shift[1]::samp[1], shift[2]::samp[2], shift[3]::samp[3]]
    if len(samp) == 5:
        return band[shift[0]::samp[0], shift[1]::samp[1], shift[2]::samp[2], shift[3]::samp[3], shift[4]::samp[4]]

def upsamp(band, samp, shift = None):
    """
    Upsample a N-D array by length N of power-2 integers 
    """    
    if shift is None:
        shift = ncp.zeros(len(band.shape), dtype = int)
    sp = ncp.array(band.shape)*samp
    shape = tuple(int(x) for x in sp)   # works whether sp is list, np.ndarray, or cp.ndarray
    bandup = ncp.zeros(shape, dtype=complex)
    if len(samp) == 2:
        bandup[shift[0]::samp[0], shift[1]::samp[1]] = band
    if len(samp) == 3:
        bandup[shift[0]::samp[0], shift[1]::samp[1], shift[2]::samp[2]] = band
    if len(samp) == 4:
        bandup[shift[0]::samp[0], shift[1]::samp[1], shift[2]::samp[2], shift[3]::samp[3]] = band
    if len(samp) == 5:
        bandup[shift[0]::samp[0], shift[1]::samp[1], shift[2]::samp[2], shift[3]::samp[3], shift[4]::samp[4]] = band

    return bandup

r = ncp.pi*ncp.array([1/3, 2/3, 2/3, 4/3])
alpha = 0.1

####  class to hold all curvelet windows and other based on transform configuration
class Udct:
    def __init__(self, sz, cfg, complex = False, sparse = False, high = 'curvelet'):
        self.name = "ucurv"
        # 
        if high != 'curvelet':
            self.sz = tuple(ncp.array(sz)//2)
        else:
            self.sz = tuple(sz)

        self.cfg = tuple(cfg)
        self.complex = complex
        self.sparse = sparse
        self.high = high
        self.dim = len(sz)
        self.res = len(cfg)

        dim = len(sz)
        res = len(cfg)

        self.Sampling = {}
        # calculate output len
        clen = ncp.prod(ncp.array(self.sz))//((2**self.dim)**(self.res-1))
        self.len = clen
        for i in range(self.res):
            clen = clen*((2**self.dim)**i)
            self.len = self.len + clen*self.dim*3**(self.dim-1)//2**(self.dim-1)

        # create the subsampling vectors
        self.Sampling[(0)]  = 2**(res-1)*ncp.ones(dim, dtype = int) 
        for rs in range(res):
            for ipyr in range(dim):
                dmat = []
                for idir in range(dim):
                    if idir == ipyr:
                        dmat.append(2**(res-rs))
                    else:
                        dmat.append(2*(cfg[rs][idir]//3)*2**(res-rs-1))  
                self.Sampling[(rs,ipyr)] = ncp.array(dmat, dtype = int)

        Sgrid = [ [] for i in range(dim) ]

        for ind in range(dim):
            Sgrid[ind] = ncp.linspace(-1.5 * ncp.pi, 0.5 * ncp.pi - ncp.pi / (self.sz[ind]  / 2), self.sz[ind]) 

        f1d = {}
        # print(f1d)
        for ind in range(dim):
            for rs in range(res):
                f1d[ (rs, ind) ] = fun_meyer(ncp.abs(Sgrid[ind]), [-2, -1, r[0]/2**(res-1-rs), r[1]/2**(res-1-rs)] )

            f1d[ (res, ind )] = fun_meyer(ncp.abs(Sgrid[ind]), [-2, -1, r[2], r[3] ])

        SLgrid = [ [] for i in range(dim) ]
        for ind in range(dim):
            SLgrid[ind] = ncp.linspace(-ncp.pi,  ncp.pi - ncp.pi / (self.sz[ind]  / 2), self.sz[ind])

        # fl1d = []
        FL = ncp.ones([1])
        for ind in range(dim):
            fl1d = fun_meyer(ncp.abs(SLgrid[ind]), [-2, -1, r[0]/2**(res-1), r[1]/2**(res-1)] ) 
            FL = ncp.kron(FL, fl1d.flatten() )
            # print(FL.shape)
        FL = FL.reshape(self.sz)

        # Mang2 will contain all the 2D angle functions needed to create dim-dimension 
        # angle pyramid. As such it is a 4D dictionary 2D angle funtions. The dimension are
        # Resolution - Dimension (number of hyper pyramid) - Dimension-1 (number of angle
        # function in each pyramid ) - Number of angle function in that particular resolution-direction
        #
        Mang2 = {}
        for rs in range(res):
            # For each resolution we loop through each pyramid
            for ind in range(dim):
                # For each pyramid we try to collect all the 2D angle function so that we can build the dim 
                # dim-dimension angle functions
                for idir in range(dim):
                    if idir == ind : # skip the dimension that is the same as the pyramid
                        continue
                    
                    # print(ndir, cfg[rs][ idir] )
                    Mg0 = tan_theta_grid(Sgrid[ind], Sgrid[idir] )

                    # create the bandpass function
                    BP1 = ncp.outer(f1d[(rs,ind)], f1d[(rs,idir)] )
                    BP2 = ncp.outer(f1d[(rs+1,ind)], f1d[(rs+1,idir)] )
                    bandpass = (BP2 - BP1)**(1./(dim-1.))

                    # create the 2D angle function, in the vertical 2D pyramid
                    Mang2[(rs, ind, idir)]  = angle_fun( Mg0, cfg[rs][ idir] , alpha, 1, bandpass)

        #################################
        Msubwin = {}
        cnt = 0

        for rs in range(res):
            dlists = generate_combinations(dim)[::-1]
            #print(dlists)
            id_angle_lists = []
            for x in dlists:
                new_list = [[i] for i in range(cfg[rs][x[0]])]
                for i in range(1, len(x)):
                    new_list = [z + [j] for z in new_list for j in range(cfg[rs][x[i]])] 
                id_angle_lists.append(new_list)
            #print(dlists)
            #print(id_angle_lists)
            for ipyr in range(dim):
                # for each resolution-pyramid, id_angle_list is the angle combinaion within that pyramid
                # for instance, (5,5) would be the last angle of a (6,6) 3D pyramid
                # and dlist is the list of the dimension of that pyramid, 
                # for instance (0,2) would be the list of pyramid of dimension 1 in 3D case
                id_angle_list = id_angle_lists[ipyr]
                dlist = list(dlists[ipyr])

                for alist in id_angle_list:
                    subband = ncp.ones(self.sz)
                    for idir, aid in enumerate(alist):
                        angkron = angle_kron(Mang2[(rs, ipyr, dlist[idir])][aid] , [ipyr, dlist[idir]], self.sz)
                        subband = subband*angkron
                        cnt += 1

                    Msubwin[tuple([rs, ipyr] + alist)] = subband.copy()

        #################################
        sumall = ncp.zeros(self.sz)
        for id, subwin in Msubwin.items():
            sumall = sumall + subwin
            # print(id, ncp.max(subwin), ncp.max(sumall))

        sumall = sumall + fftflip(sumall)
        sumall = sumall + FL

        self.Msubwin = {}
        for id, subwin in Msubwin.items():
            win = ncp.fft.fftshift(ncp.sqrt(2*ncp.prod(self.Sampling[(id[0], id[1])]) *subwin / sumall))
            if sparse:
                self.Msubwin[id] = ( ncp.nonzero(win), win[ncp.nonzero(win)] )
            else:
                self.Msubwin[id] = win
        win  = ncp.sqrt(ncp.prod(self.Sampling[(0)]))*ncp.fft.fftshift(ncp.sqrt(FL/sumall))        
        if sparse:
            self.FL = ( ncp.nonzero(win), win[ncp.nonzero(win)] )
        else:
            self.FL = win


def ucurvfwd(img, udct):
    if udct.high == 'curvelet':
        assert img.shape == udct.sz
    Msubwin = udct.Msubwin
    # FL = udct.FL
    Sampling = udct.Sampling
    if udct.sparse:
        FL = ncp.zeros(udct.sz)
        FL[udct.FL[0]] = udct.FL[1]
    else:
        FL = udct.FL    
    imband = {}
    if udct.high == 'wavelet':
        band = meyerfwdmd(img)
        for i, band in enumerate(band):
            if i == 0:
                imf = ncp.fft.fftn(band)
            else:    
                imband[(udct.res, i)] = band
    else:
        imf = ncp.fft.fftn(ncp.array(img))

    if udct.complex:
        bandfilt = ncp.fft.ifftn(imf*FL)
        imband[(0,)] = downsamp(bandfilt, Sampling[(0)])
        for id, subwin in Msubwin.items():
            if udct.sparse:
                sbwin = ncp.zeros(udct.sz)
                sbwin[subwin[0]] = subwin[1]
                subwin = sbwin
            bandfilt = ncp.sqrt(0.5)*ncp.fft.ifftn(imf *subwin)
            imband[id] = downsamp(bandfilt, Sampling[(id[0], id[1])])
            id2 = list(id)
            id2[1] = id2[1] + udct.dim
            bandfilt = ncp.sqrt(0.5)*ncp.fft.ifftn(imf *fftflip(subwin))
            imband[tuple(id2)] = downsamp(bandfilt, Sampling[(id[0], id[1])])

    else:    
        bandfilt = ncp.real(ncp.fft.ifftn(imf*FL))
        imband[(0,)] = downsamp(bandfilt, Sampling[(0)]) # ncp.real(ncp.fft.ifftn(imf*FL))
        for id, subwin in Msubwin.items():
            if udct.sparse:
                sbwin = ncp.zeros(udct.sz)
                sbwin[subwin[0]] = subwin[1]
                subwin = sbwin

            bandfilt = ncp.fft.ifftn(imf *subwin)
            # samp = Sampling[(id[0], id[1])]
            # imband[id] = bandfilt[::samp[0], ::samp[1]]
            imband[id] = downsamp(bandfilt, Sampling[(id[0], id[1])])
            # print(bandfilt.shape, Sampling[(id[0], id[1])], imband[id].shape)

    return imband    

##############
def ucurvinv(imband, udct):
    Msubwin = udct.Msubwin
    Sampling = udct.Sampling
    # imlow = imband[0]
    imlow = upsamp(imband[(0,)], Sampling[(0)])

    if udct.sparse:
        FL = ncp.zeros(udct.sz)
        FL[udct.FL[0]] = udct.FL[1]
    else:
        FL = udct.FL    

    if udct.complex:
        recon = ncp.fft.ifftn( ncp.fft.fftn(imlow) * FL)
    else:
        recon = ncp.real(ncp.fft.ifftn( ncp.fft.fftn(imlow) * FL) )
    for id, subwin in Msubwin.items():
        #
        if udct.high != 'curvelet' and id[0] == udct.res :
            continue

        if udct.sparse:
            sbwin = ncp.zeros(udct.sz)
            sbwin[subwin[0]] = subwin[1]
            subwin = sbwin

        if udct.complex:
            bandup = upsamp(imband[id], Sampling[(id[0], id[1])])
            recon = recon + ncp.sqrt(0.5)*ncp.fft.ifftn( ncp.fft.fftn(bandup) * subwin )
            id2 = list(id)
            id2[1] = id2[1] + udct.dim
            bandup = upsamp(imband[tuple(id2)], Sampling[(id[0], id[1])])
            recon = recon + ncp.sqrt(0.5)*ncp.fft.ifftn( ncp.fft.fftn(bandup) * fftflip(subwin) )
        else:
            bandup = upsamp(imband[id], Sampling[(id[0], id[1])])
            recon = recon + ncp.real(ncp.fft.ifftn( ncp.fft.fftn(bandup) * subwin ))
            
    if udct.high == 'wavelet':
        band = [recon]
        for id, suband in imband.items():
            if id[0] == udct.res:
                band.append(suband)

        recon = meyerinvmd(band)
    
    return recon

    