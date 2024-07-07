
import numpy as np

def zoneplate(sz):
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
        
        
    return rsq  