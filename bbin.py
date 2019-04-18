"""Module to perform buoyancy-binning of 2D dedalus tracer-release
experiments

Ryan Holmes

"""

import numpy as np
from scipy.sparse import spdiags,eye,csr_matrix
from scipy.optimize import leastsq, minimize
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt

def trbin(y,z,tr,B,N2,theta,Lz,Ly,q0):
    """Bin the 2D tracer distribution tr(y,z) into buoyancy bins using
    original method
    """

    # Define buoyancy grid:
    b = np.linspace(0,N2*np.cos(theta)*Lz + N2*np.sin(theta)*Ly,256);db = b[1]-b[0]
    ba = (b[:-1]+b[1:])/2
    
    dy = y[1:]-y[:-1];dz = z[1:]-z[:-1];
    dA = np.tile(dy,[len(z)-1,1]).T*np.tile(dz,[len(y)-1,1]);
    B = (B[1:,:] + B[:-1,:])/2
    B = (B[:,1:] + B[:,:-1])/2

    tr = (tr[1:,:] + tr[:-1,:])/2
    tr = (tr[:,1:] + tr[:,:-1])/2
    tr = tr*dA
    trC = [np.sum(tr[np.logical_and(B>=bi,B<(bi+db))])/db for bi in b[:-1]]

    return(ba,trC)

def trbinI(y,z,tr,N2,theta,Lz,Ly,q0,Kinf,K0,d,SPru0i=0.):
    """Bin the 2D tracer distribution tr(y,z) into buoyancy bins by first
    interpolating onto a high-resolution y-z grid.

    """

    # Define buoyancy grid:
    b = np.linspace(0,N2*np.cos(theta)*Lz + N2*np.sin(theta)*Ly,256);db = b[1]-b[0]
    ba = (b[:-1]+b[1:])/2

    # Define high-res y,z grid:
    yhr = np.linspace(0.,Ly,512)
    zhr = np.linspace(0.,Lz,512)

    f = interp2d(y,z,tr.T)
    trhr = f(yhr,zhr)

    [ym,zm] = np.meshgrid(yhr,zhr)
    # Bhr = N2*np.sin(theta)*ym + N2*np.cos(theta)*(zm + np.exp(-q0*zm)*np.cos(q0*zm)/q0)
    Bhr = N2*np.sin(theta)*ym + N2*np.cos(theta)/(1.+SPru0i)*(zm +
                np.exp(-q0*zm)*np.cos(q0*zm)/q0*(1.+SPru0i*Kinf/K0) +
                SPru0i*d*np.log(1.+Kinf/K0*(np.exp(zm/d)-1.)))

    dy = yhr[1:]-yhr[:-1];dz = zhr[1:]-zhr[:-1];
    dA = np.tile(dy,[len(zhr)-1,1]).T*np.tile(dz,[len(yhr)-1,1]);
    Bhr = (Bhr[1:,:] + Bhr[:-1,:])/2
    Bhr = (Bhr[:,1:] + Bhr[:,:-1])/2

    trhr = (trhr[1:,:] + trhr[:-1,:])/2
    trhr = (trhr[:,1:] + trhr[:,:-1])/2
    trhr = trhr*dA
    trC = [np.sum(trhr[np.logical_and(Bhr>=bi,Bhr<(bi+db))])/db for bi in b[:-1]]

    return(ba,trC)
    
