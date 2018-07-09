"""Module to perform a three-parameter fit to the 2D dedalus
simulations using a one-dimensional advection-diffusion equation.

Follows the technique discussed briefly in Ledwell et al. 1995, 1998.

Ryan Holmes

"""

import numpy as np
from scipy.sparse import spdiags,eye,csr_matrix
from scipy.optimize import minimize

# Grid parameters:
Lh = 2000. # Max size of h-grid
n = 200 # Number of h-points
dt = 1e5 # Time-step

h = np.linspace(-Lh/2.,Lh/2.,n)
dh = h[1]-h[2]

# Initial guess:
# [K0, Kh, w]
x0 = [1., 0., 0.]

# Scales
xsc = [1.e-4,1.e-7,1.e-7]

def fit3par(zF,trF,tf,sz):
    """Return parameters K0, Kh and w that best match the tracer
    profile trF as a function of zF at time tf where the initial
    tracer distribution is a gaussian about zF=0 with standard
    deviation sz"""

    # Initial tracer distribution:
    trI = np.exp(-h**2/2/sz**2)/np.sqrt(2*np.pi*sz**2) 

    nt = tf / dt

    res = minimize(cost, x0, method = 'nelder-mead',
                   options = {'xtol': 1e-2, 'disp': True})

    return(res)
    
def cost(x):

    """ Cost function """
    
    A = operator(x[0]*xsc[0],x[1]*xsc[1],x[2]*xsc[2])
    tr = solve(A,nt,trI)
    
    return(chisq(tr,trF))

def operator(K0,Kh,w):
    """ Construct the operator matrix for given inputs """
    
    K = np.maximum( K0 + h*Kh , np.zeros_like(h) )
    A = csr_matrix(np.tile(K,(n,1)).T)
    A = A.multiply(D2)
    A += (Kh - w) * D1

    return(A)

def solve(A,nt,trI):
    """ Take an initial concentration to a final concentration using
    the matrix operator A and return the final concentration."""
    
    tr = np.copy(trI)
    for t in range(nt):
        tr += dt * ( A @ tr )
    
    return(tr)

def chisq(f1,f2):
    """Calculated L2 norm of difference between two concentration
    profiles"""
    chisq = np.sum((f1-f2)**2)
    return(chisq)

def D2:
    """Centered 2nd derivative matrix"""
    ec=([-1.] + (n-2)*[-2.] + [-1.])
    em=[1.]*n
    ep=[1.]*n
    A=spdiags([em,ec,ep],[-1,0,1],n,n)
    A/=dh**2
    return A

def D1:
    """Centered 1st derivative matrix"""
    ec=([1] + (n-2)*[0] + [-1.])
    em=[1.]*n
    ep=[-1.]*n
    A=spdiags([em,ec,ep],[-1,0,1],n,n)
    A/=dh
    return A
