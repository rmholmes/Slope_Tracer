"""Module to solve the 1D advection-diffusion equation for fitting
diffusivities and velocities to the 2D results

Ryan Holmes

"""

import numpy as np
from scipy.sparse import spdiags,eye,csr_matrix

def solve(Lh,n,dt,nt,K0,Kh,w,trI):

    # Define h:
    h = np.linspace(-Lh/2.,Lh/2.,n)
    dh = h[1]-h[2]

    # Make Matrix:
    K = np.maximum( K0 + h*Kh , np.zeros_like(h) )
    A = csr_matrix(np.tile(K,(n,1)).T)
    A = dt * A.multiply(d2(n,dh)) # Kappa term
    A += - dt * (w - Kh) * d1(n,dh) # First derivative term

    # Advance in time:
    tr = np.copy(trI)
    for t in range(nt):
        tr += ( A @ tr )
    
    return(tr)

def chisq(f1,f2):
    """Calculated L2 norm of difference"""
    chisq = np.sum((f1-f2)**2)
    return(chisq)

def d2(m,dh):
    """Centered 2nd derivative"""
    ec=([-1.] + (m-2)*[-2.] + [-1.])
    em=[1.]*m
    ep=[1.]*m
    A=spdiags([em,ec,ep],[-1,0,1],m,m)
    A/=dh**2
    return A

def d1(m,dh):
    """Centered 1st derivative"""
    ec=([1] + (m-2)*[0] + [-1.])
    em=[1.]*m
    ep=[-1.]*m
    A=spdiags([em,ec,ep],[-1,0,1],m,m)
    A/=dh
    return A
