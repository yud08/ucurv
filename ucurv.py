import math
import numpy as np

import math
import numpy as np
import matplotlib.pyplot as plt

def fun_meyer(x, param):

    p = np.array([-20,70,-84, 35, 0, 0, 0, 0])
    # x = np.linspace(0,5)
    y = np.ones_like(x)

    y[x <= param[0] ] = 0.
    y[x >= param[3] ] = 0.
    xx = (x[ (x >= param[0]) & (x <= param[1]) ] -param[0] ) /(param[1]-param[0])
    y[ (x >= param[0]) & (x <= param[1]) ] = np.polyval( p, xx)
    xx = (x[ (x >= param[2]) & (x <= param[3]) ] - param[3] ) /(param[2]-param[3])
    y[ (x >= param[2]) & (x <= param[3]) ] = np.polyval( p, xx)
    return y.reshape(x.shape)


def adapt_grid(S1, S2):
    x1, x2 = np.meshgrid(S2, S1)

    # scale the grid approximate the tan theta function ------------------
    # create two scale grid for mostly horizontal and vertical direction
    # first grid
    t1 = np.zeros_like(x1)
    ind = np.logical_and(x1 != 0, np.abs(x2) <= np.abs(x1))
    t1[ind] = -x2[ind] / x1[ind]

    t2 = np.zeros_like(x1)
    ind = np.logical_and(x2 != 0, np.abs(x1) < np.abs(x2))
    t2[ind] = x1[ind] / x2[ind]
    t3 = t2.copy()
    t3[t2 < 0] = t2[t2 < 0] + 2
    t3[t2 > 0] = t2[t2 > 0] - 2

    M1 = t1 + t3
    M1[x1 >= 0] = -2

    # second grid
    t1 = np.zeros_like(x1)
    ind = np.logical_and(x2 != 0, np.abs(x1) <= np.abs(x2))
    t1[ind] = -x1[ind] / x2[ind]

    t2 = np.zeros_like(x1)
    ind = np.logical_and(x1 != 0, np.abs(x2) < np.abs(x1))
    t2[ind] = x2[ind] / x1[ind]
    t3 = t2.copy()
    t3[t2 < 0] = t2[t2 < 0] + 2
    t3[t2 > 0] = t2[t2 > 0] - 2

    M2 = t1 + t3
    M2[x2 >= 0] = -2

    return M2, M1

def fftflip(F, dirlist):
    Fc = F.copy()
    dim = Fc.ndim
        
    shiftvec = np.zeros(dim)
    if type(dirlist) is list:
        for dir in dirlist:
            shiftvec[dir] = 1
            Fc = np.flip(Fc, dir)
            Fc = np.roll(Fc, 1, axis=dir)
    if type(dirlist) is int:
        dir = dirlist
        shiftvec[dir] = 1
        Fc = np.flip(Fc, dir)
        Fc = np.roll(Fc, 1, axis=dir)
    return Fc

def angle_fun(Mgrid, n, alpha, dir, bandpass = None):
    # angle meyer window
    angd = 2/n
    ang = angd*np.array([-alpha, alpha, 1-alpha, 1+alpha])
    Mang = [[] for i in range(n) ]
    tmp = []
    sp = np.array(Mgrid.shape)
    for id in range(math.ceil(n/2)):
        ang2 = -1 + id*angd + ang
        x = fun_meyer(Mgrid, ang2)
        if bandpass is not None:
            x = x*bandpass
        x = np.roll(x, 3*sp//4, (0,1) )
        Mang[id] = x.copy()
        Mang[n-1-id] = fftflip(x, dir)
        
    return Mang


def angle_kron(F, dr, sz):
    sp = list(F.shape)
    sp2 = int(np.prod(sz)/np.prod(F.shape))
    sp.append(sp2)
    #print(sp, sp2)
    Fk = np.reshape(np.kron(F.flatten(), np.ones( sp2 )) ,  sp)

    #print(Fk.shape)
    Fk = np.moveaxis(Fk, [0 ,1] , dr)
    #print(Fk.shape)
    #Fk =  np.swapaxes(Fk, 1, 2)
    # print("ff",Fk.shape)
    return Fk

def downsamp(band, samp):
    if len(samp) == 2:
        return band[::samp[0], ::samp[1]]
    if len(samp) == 3:
        return band[::samp[0], ::samp[1], ::samp[2]]

def upsamp(band, samp):
    sp = np.array(band.shape)*samp
    bandup = np.zeros(sp, dtype = complex)
    if len(samp) == 2:
        bandup[::samp[0], ::samp[1]] = band
        
    if len(samp) == 3:
        bandup[::samp[0], ::samp[1], ::samp[2]] = band
    return bandup


r = [1.0472, 2.0944, 2.0944, 4.1888]
alpha = 0.15

####  class to hold all curvelet windows and other based on transform configuration
class ucurv:
    def __init__(self, sz, cfg):
        self.name = "ucurv"
        # 
        dim = len(sz)
        res = len(cfg)

        self.Sampling = {}
        for rs in range(res):
            for ipyr in range(dim):
                dmat = []
                for idir in range(dim):
                    if idir == ipyr:
                        dmat.append(2**(res-rs))
                    else:
                        dmat.append(2*(cfg[rs][idir]//3)*2**(res-rs-1))  
                self.Sampling[(rs,ipyr)] = np.array(dmat, dtype = int)

        Sgrid = [ [] for i in range(dim) ]

        for ind in range(dim):
            Sgrid[ind] = np.linspace(-1.5 * np.pi, 0.5 * np.pi - np.pi / (sz[ind]  / 2), sz[ind]) 

        f1d = {}
        # print(f1d)
        for ind in range(dim):
            for rs in range(res):
                f1d[ (rs, ind) ] = fun_meyer(np.abs(Sgrid[ind]), [-2, -1, r[0]/2**(res-1-rs), r[1]/2**(res-1-rs)] )

            f1d[ (res, ind )] = fun_meyer(np.abs(Sgrid[ind]), [-2, -1, r[2], r[3] ])

        SLgrid = [ [] for i in range(dim) ]
        for ind in range(dim):
            SLgrid[ind] = np.linspace(-np.pi,  np.pi - np.pi / (sz[ind]  / 2), sz[ind])

        # fl1d = []
        FL = np.ones([1])
        for ind in range(dim):
            fl1d = fun_meyer(np.abs(SLgrid[ind]), [-2, -1, r[0]/2**(res-1), r[1]/2**(res-1)] ) 
            FL = np.kron(FL, fl1d.flatten() )
            # print(FL.shape)
        FL = FL.reshape(sz)

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
                    
                    ndir = np.array([ind, idir]) # ndir are the dimension in the pyramid 
                    # print(ndir, cfg[rs][ idir] )
                    Mg0, Mg1 = adapt_grid(Sgrid[ndir[0]], Sgrid[ndir[1] ] )

                    # create the bandpass function
                    BP1 = np.outer(f1d[(rs,ind)], f1d[(rs,idir)] )
                    BP2 = np.outer(f1d[(rs+1,ind)], f1d[(rs+1,idir)] )
                    bandpass = (BP2 - BP1)**(1./(dim-1.))

                    # create the 2D angle function, in the vertical 2D pyramid
                    Mang2[(rs, ind, idir)]  = angle_fun( Mg0, cfg[rs][ idir] , alpha, 1, bandpass)

        #################################
        Msubwin = {}
        cnt = 0
        for rs in range(res):
            for ipyr in range(dim):
                id_angle_list = []
                dlist = []
                for idir2 in range(dim):
                    if ipyr == idir2:
                        continue
                    else:
                        dlist.append(idir2)
                        # print(cfg[rs][idir2])
                        if len(id_angle_list) == 0:
                            id_angle_list = [[i] for i in range(cfg[rs][idir2]) ]
                        else:
                            new_list = []
                            for i in range(len(id_angle_list)):
                                for j in range(cfg[rs][idir2]):
                                    new_list.append((id_angle_list[i] + [j]))
                            id_angle_list = new_list.copy()

                # print(id_angle_list)
                
                for alist in id_angle_list:
                    subband = np.ones(sz)
                    for idir, aid in enumerate(alist):
                        angkron = np.squeeze(angle_kron(Mang2[(rs, ipyr, dlist[idir])][aid] , [ipyr, dlist[idir]], sz))
                        subband = subband*angkron
                        cnt += 1

                    Msubwin[tuple([rs, ipyr] + alist)] = subband.copy()

        #################################
        sumall = np.zeros(sz)
        for id, subwin in Msubwin.items():
            sumall = sumall + subwin
            # print(id, np.max(subwin), np.max(sumall))

        sumall = sumall + fftflip(sumall, list(range(dim)))
        sumall = sumall + FL

        self.Msubwin = {}
        for id, subwin in Msubwin.items():
            self.Msubwin[id] = np.fft.fftshift(np.sqrt(2*np.prod(self.Sampling[(id[0], id[1])]) *subwin / sumall))
        self.Sampling[(0)]  = 2**(res-1)*np.ones(dim, dtype = int) 
        self.FL = np.sqrt(np.prod(self.Sampling[(0)]))*np.fft.fftshift(np.sqrt(FL/sumall))


def ucurvfwd(img, udct):
    Msubwin = udct.Msubwin
    FL = udct.FL
    Sampling = udct.Sampling

    imf = np.fft.fftn(img)

    imband = {}
    bandfilt = np.real(np.fft.ifftn(imf*FL))
    print(bandfilt.shape, Sampling[(0)])
    imband[0] = downsamp(bandfilt, Sampling[(0)]) # np.real(np.fft.ifftn(imf*FL))
    for id, subwin in Msubwin.items():
        bandfilt = np.fft.ifftn(imf *subwin)
        # samp = Sampling[(id[0], id[1])]
        # imband[id] = bandfilt[::samp[0], ::samp[1]]
        imband[id] = downsamp(bandfilt, Sampling[(id[0], id[1])])
        # print(bandfilt.shape, Sampling[(id[0], id[1])], imband[id].shape)

    return imband    


##############
def ucurvinv(imband, udct):
    Msubwin = udct.Msubwin
    FL = udct.FL
    Sampling = udct.Sampling
    # imlow = imband[0]
    imlow = upsamp(imband[0], Sampling[(0)])
    recon = np.real(np.fft.ifftn( np.fft.fftn(imlow) * FL) )
    for id, subwin in Msubwin.items():
        # bandup = np.zeros_like(imf)
        # samp = Sampling[(id[0], id[1])]
        # bandup[::samp[0], ::samp[1]] = imband[id]
        bandup = upsamp(imband[id], Sampling[(id[0], id[1])])
        recon = recon + np.real(np.fft.ifftn( np.fft.fftn(bandup) * subwin ))
    
    return recon        
