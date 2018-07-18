"""Module to perform a three-parameter fit to the 2D dedalus
simulations using a one-dimensional advection-diffusion equation.

Follows the technique discussed briefly in Ledwell et al. 1995, 1998.

Ryan Holmes

"""

import numpy as np
from scipy.sparse import spdiags,eye,csr_matrix
#from scipy.optimize import minimize
from scipy.optimize import leastsq, minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Grid parameters:
n = 100 # Number of h-points
hper = 0.99 # High-percentile
lper = 0.01 # Low-percentile
buf = 500. # Buffer (m)
dt = int(1e5) # Time-step

# Initial guess:
# [K0, Kh, w]
x0 = [1., 0., 1.]

# Scales
xsc = [1.e-4,1.e-7,1.e-7]

# Method:
method = 0 # 0 = leastsq, 1 = minimize

## Least Squares method:
def fit3par(zF,trF,trINI,tf):
    """Return parameters K0, Kh and w that best match the tracer
    profile trF as a function of zF at time tf where trINI is the
    initial tracer distribution"""

    h = get_grid(zF,trF)
    trI, tmp = trOBS(zF,trINI,h)
    trO, tmp = trOBS(zF,trF,h)

    nt = tf // dt
    if (method == 0):
        res = leastsq(cost, x0, args= (nt,trI,trO,h),full_output=1)
        class resout:
            pass
        reso = resout()
        reso.fun = costCHISQ(res[0], nt, trI, trO, h)
        reso.x = [res[0][0]*xsc[0],res[0][1]*xsc[1],res[0][2]*xsc[2]]
        reso.success = res[-1]
        reso.msg = res[-2]
        if (res[1] is not None):
            pcov = res[1] * ( reso.fun / (len(h)-3))
            reso.xerr = []
            for i in range(3):
                reso.xerr.append(xsc[i]*np.absolute(pcov[i][i])**0.5)
        else:
            reso.xerr = [np.inf] * 3
        res = reso
    else:
        res = minimize(costCHISQ, x0, args = (nt,trI, trO, h), method = 'nelder-mead',
                       options = {'xtol': 1e-5})
        res.x[0] = res.x[0]*xsc[0]
        res.x[1] = res.x[1]*xsc[1]
        res.x[2] = res.x[2]*xsc[2]
    
    return(res)

def cost(x, nt, trI, trO,h):
    """ Cost function """

    tr = solve(x[0]*xsc[0],x[1]*xsc[1],x[2]*xsc[2],nt,trI,h)
    
    return(tr-trO)

def costCHISQ(x, nt, trI, trO, h):
    """ Cost function """
    tr = solve(x[0]*xsc[0],x[1]*xsc[1],x[2]*xsc[2],nt,trI,h)
    
    chisq = np.sum((tr-trO)**2)
    return(chisq)

def solve(K0,Kh,w,nt,trI,h):
    """ Take an initial concentration to a final concentration."""
    
    K = np.maximum( K0 + h*Kh , np.zeros_like(h) )
    A = csr_matrix(np.tile(K,(n,1)).T)
    A = A.multiply(D2(h[1]-h[0]))
    A = A + (Kh - w) * D1(h[1]-h[0])

    tr = np.copy(trI)
    for t in range(nt):
        tr += dt * ( A * tr )
    
    return(tr)

def trMOD(x,zF,trF,trINI,tf):
    """Return tracer profile for given parameters and time on input
    grid."""

    # Plot modelled and observed solutions:
    h = get_grid(zF,trF)
    trI, scl = trOBS(zF,trINI,h)
    trM = solve(x[0],x[1],x[2],tf // dt, trI, h)

    trMscl = trSCL(zF,trM,h,scl)
    
    return(trMscl)

def plot(x,zF,trF,trINI,tf):
    """ Plot initial, final and modelled tracer profiles for given
    parameters."""

    # Plot modelled and observed solutions:
    h = get_grid(zF,trF)
    trI, tmp = trOBS(zF,trINI,h)
    trO, tmp = trOBS(zF,trF,h)
    trM = solve(x[0],x[1],x[2],tf // dt, trI, h)

    f = plt.figure(figsize=(5,5),facecolor='white')
    plt.plot(trI,h,'-r',label='Initial')
    plt.plot(trO,h,'-k',label='Observed')
    plt.plot(trM,h,'-b',label='Modelled')
    plt.legend()
    plt.xlabel('Tracer Concentration')
    plt.ylabel('Depth above target surface')
    plt.xlim([0., np.max(trO)*1.1])
    plt.ylim([np.min(h),np.max(h)])
    plt.title('Time = %04d, $\kappa_0= $ %.2e, $\kappa_h = $ %.2e, $w = $, %.2e'  % (tf // 1e5, x[0],x[1],x[2]))
    
def get_grid(zF,trF):
    """ Determine h-grid to use"""
    csum = np.cumsum(trF)
    f = interp1d(csum,zF)
    hmax = f(hper*csum[-1])+buf
    hmin = f(lper*csum[-1])-buf
    
    h = np.linspace(hmin,hmax,n)
    return(h)

def trOBS(zF,trF,h):
    """ Return observed final tracer distribution on h grid"""
    dh = h[1]-h[0]
    f = interp1d(zF,trF,fill_value=0.)
    trO = np.zeros_like(h)
    in_range = np.logical_and(h>=np.min(zF),h <= np.max(zF))
    trO[in_range] = f(h[in_range])

    scl = np.sum(trO*dh)
    trO = trO / scl

    return (trO, scl)

def trSCL(zF,trM,h,scl):
    """ Return modelled final tracer distribution on orginal grid"""
    dh = h[1]-h[0]
    f = interp1d(h,trM,fill_value=0.)
    trO = np.zeros_like(zF)
    in_range = np.logical_and(zF>=np.min(h),zF <= np.max(h))
    trO[in_range] = f(zF[in_range])

    trO = trO * scl

    return (trO)

def D2(dh):
    """Centered 2nd derivative matrix"""
    ec=([-1.] + (n-2)*[-2.] + [-1.])
    em=[1.]*n
    ep=[1.]*n
    A=spdiags([em,ec,ep],[-1,0,1],n,n)
    A/=dh**2
    return A

def D1(dh):
    """Centered 1st derivative matrix"""
    ec=([1] + (n-2)*[0] + [-1.])
    em=[1.]*n
    ep=[-1.]*n
    A=spdiags([em,ec,ep],[-1,0,1],n,n)
    A/=dh
    return A
