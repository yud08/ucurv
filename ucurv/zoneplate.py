import numpy as np

#used as example data for ucurv backwards and forwards transform
def zoneplate(sz):
    """
    Generate an N‑dimensional zone plate pattern.

    Parameters
    ----------
    sz : sequence of int
        A sequence of length 2, 3, or 4 specifying the size along each dimension.
        For example, (nx, ny) for 2D, (nx, ny, nz) for 3D, or (nx, ny, nz, np) for 4D.

    Returns
    -------
    rsq : ndarray
        An N‑dimensional array of shape `tuple(sz)` containing the zone plate values
        computed as

            cos(pi / max(sz) * (x_1**2 + x_2**2 + ... + x_N**2)),

        where each coordinate x_i is linearly spaced from `-sz[i]/2` to `sz[i]/2`.

    Raises
    ------
    ValueError
        If `sz` does not have length 2, 3, or 4.
        Higher dimensions than 4 are not implemented.
    """
    if len(sz) == 1:
        raise ValueError("Zoneplate does not work with 1D")
    if len(sz) == 2:
        x_ = np.linspace(-sz[0]/2, sz[0]/2, sz[0] )
        y_ = np.linspace(-sz[1]/2, sz[1]/2, sz[1] )
        x, y = np.meshgrid(x_, y_, indexing='ij')
        rsq = np.cos(np.pi/np.max(sz)*(x**2 + y**2) )
    if len(sz) == 3:
        x_ = np.linspace(-sz[0]/2, sz[0]/2, sz[0] )
        y_ = np.linspace(-sz[1]/2, sz[1]/2, sz[1] )
        z_ = np.linspace(-sz[2]/2, sz[2]/2, sz[2] )
        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        rsq = np.cos(np.pi/np.max(sz)*(x**2 + y**2 + z**2 ) )
    if len(sz) == 4:
        x_ = np.linspace(-sz[0]/2, sz[0]/2, sz[0] )
        y_ = np.linspace(-sz[1]/2, sz[1]/2, sz[1] )
        z_ = np.linspace(-sz[2]/2, sz[2]/2, sz[2] )
        p_ = np.linspace(-sz[3]/2, sz[3]/2, sz[3] )
        x, y, z, p = np.meshgrid(x_, y_, z_, p_, indexing='ij')
        rsq = np.cos(np.pi/np.max(sz)*(x**2 + y**2 + z**2 + p**2 ) )
    if len(sz) > 4:
        raise ValueError("Zoneplate is not implemented for higher dimensions")
        
    return rsq  