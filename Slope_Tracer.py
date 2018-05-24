"""
Dedalus script for 2D tracer advection-diffusion on a slope

This script uses a Fourier basis in the y direction with periodic boundary
conditions.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 igw.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5

"""
import numpy as np
import h5py
import matplotlib
import shutil
from mpi4py import MPI
from scipy.special import erf
import time
from IPython import display
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools import post
import pathlib
from dedalus.extras import plot_tools
import logging
logger = logging.getLogger(__name__)

# Input Grids and parameters ------------------------------------------------------------
Ly, Lz = (800000., 1500.) # units = 1m

# Create bases and domain
y_basis = de.Fourier('y', 256, interval=(0, Ly))#, dealias=3/2)
z_basis = de.Chebyshev('z', 128, interval=(0, Lz))#, dealias=3/2)
domain = de.Domain([y_basis, z_basis], grid_dtype=np.float64)

# Input fields --------------------------------------------------------------------------
y = domain.grid(0)
z = domain.grid(1)

N2 = 1.0e-6
slope = 1/400.0
Pr0 = 1.0
theta = np.arctan(slope)

# Isotropic Diffusivity
K = domain.new_field()
K.meta['y']['constant'] = True
Kz = domain.new_field()
Kz.meta['y']['constant'] = True

Kinf = 1.0e-3
K0 = 1.0e-3
d = 500.0
K['g'] = Kinf + (K0-Kinf)*np.exp(-z/d)
K.differentiate('z',out=Kz)

# Isopycnal Diffusivity
AH = 10.0

# Upslope Velocity
PSI = domain.new_field()
PSI.meta['y']['constant'] = True
V = domain.new_field()
V.meta['y']['constant'] = True

q0 = (N2*np.sin(theta)*np.sin(theta)/4.0/Pr0/K0/K0)**(1.0/4.0)

PSI['g'] = np.cos(theta)/np.sin(theta)*(1.0-np.exp(-q0*z)*(np.cos(q0*z)+np.sin(q0*z)))

#PSI['g'] = PSI['g']*K['g']  # SML + BBL
#PSI['g'] = PSI['g']*K0 # BBL
PSI['g'] = 0.0              # No V

PSI.differentiate('z',out=V)

# Buoyancy field:
By = N2*np.sin(theta)
Bz = domain.new_field()
Bz.meta['y']['constant'] = True
B = domain.new_field()
#B['g'] = N2*np.sin(theta)*y + N2*np.cos(theta)*(z + np.exp(-q0*z)*np.cos(q0*z)/q0)
#Bz['g'] = N2*np.cos(theta)*(1.-np.exp(-q0*z)*(np.cos(q0*z)+np.sin(q0*z)))
B['g'] = N2*np.sin(theta)*y + N2*np.cos(theta)*z
Bz['g'] = N2*np.cos(theta)

# Equations and Solver
problem = de.IVP(domain, variables=['tr','trz'])
problem.meta[:]['z']['dirichlet'] = True

problem.parameters['K'] = K
problem.parameters['Kz'] = Kz
problem.parameters['AH'] = AH
problem.parameters['V'] = V
problem.parameters['Bz'] = Bz
problem.parameters['By'] = By
problem.parameters['B'] = B
problem.substitutions['GB2'] = "(Bz**2.+By**2.)"
problem.substitutions['By2'] = "(By**2.)"
problem.substitutions['Bz2'] = "(Bz**2.)"
#problem.add_equation("dt(tr) - E0*d(tr,y=2) - K0*dz(trz) = Kv*dz(trz) + Kvz*trz + Ev*d(tr,y=2) - V*dy(tr)")
#problem.add_equation("dt(tr) - E0*d(tr,y=2) - K0*dz(trz) -Kvref*dz(trz) - Kvzref*trz = (Kv-Kvref)*dz(trz) + (Kvz-Kvzref)*trz + Ev*d(tr,y=2) - V*dy(tr)")
problem.add_equation("dt(tr) + V*dy(tr) - (AH*Bz2/GB2 + K)*d(tr,y=2) + 2*AH*By*Bz/GB2*dy(trz) + AH*By*dz(Bz/GB2)*dy(tr) - (AH*By2*dz(1./GB2)+Kz)*trz - (AH*By2/GB2+K)*dz(trz) = 0.")
problem.add_equation("trz - dz(tr) = 0")
problem.add_bc("left(trz) = 0")
problem.add_bc("right(trz) = 0")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial condition:
tr = solver.state['tr']
trz = solver.state['trz']

# Gaussian blob:
#sy = Ly/80.0;sz = Lz/40.0;cy = Ly/3.0;cz = 2*d
sy = Ly/40.0;sz = Lz/20.0;cy = Ly/2.0;cz = 1.5*d
tr['g'] = np.exp(-(z-cz)**2/2/sz**2 -(y-cy)**2/2/sy**2)
#tr['g'] = np.exp(-(y-cy)**2/2/sy**2)
#tr['g'] = np.exp(-(z-cz)**2/2/sz**2)

# Function of buoyancy:
# tr['g'] = 0*z
# tr['g'][np.logical_and(B['g']/N2/np.cos(theta)>=Lz/4,B['g']/N2/np.cos(theta)<=Lz/4+200.)] = 1

tr.differentiate('z',out=trz)

# Integration parameters
lday = 1.0e5 # A "long-day" unit (86400 ~= 100000)
dt=8*lday
solver.stop_sim_time = np.inf
solver.stop_wall_time = np.inf 
solver.stop_iteration = 200*lday/dt

# Save parameters:
np.savez('runparams',Ly=Ly,Lz=Lz,N2=N2,slope=slope,theta=theta,Pr0=Pr0,Kinf=Kinf,K0=K0,d=d,AH=AH,
         q0=q0,By=By,sy=sy,sz=sz,cy=cy,cz=cz,lday=lday,dt=dt)

## Analysis
# Input fields file:
ifields = solver.evaluator.add_file_handler('ifields', iter=1e50, max_writes=1)
ifields.add_task("B", layout='g', name = 'B')
ifields.add_task("K", layout='g', name = 'K')
ifields.add_task("V", layout='g', name = 'V')
ifields.add_task("Bz", layout='g', name = 'Bz')

# Snapshots file:
snapshots = solver.evaluator.add_file_handler('snapshots', iter=2, max_writes=500)
snapshots.add_system(solver.state, layout='g')
snapshots.add_task("integ(tr,'y')", layout='g', name = 'ym0')
snapshots.add_task("integ(tr*y,'y')", layout='g', name = 'ym1')
snapshots.add_task("integ(tr*y*y,'y')", layout='g', name = 'ym2')
snapshots.add_task("integ(tr,'z')", layout='g', name = 'zm0')
snapshots.add_task("integ(tr*z,'z')", layout='g', name = 'zm1')
snapshots.add_task("integ(tr*z*z,'z')", layout='g', name = 'zm2')
snapshots.add_task("integ(integ(tr,'y'),'z')", layout = 'g', name = 'trT')
snapshots.add_task("integ(integ(tr*B,'y'),'z')", layout = 'g', name = 'bm1i')
snapshots.add_task("integ(integ(tr*B*B,'y'),'z')", layout = 'g', name = 'bm2i')
snapshots.add_task("integ(integ(tr*y,'y'),'z')", layout='g', name = 'ym1i')
snapshots.add_task("integ(integ(tr*y*y,'y'),'z')", layout='g', name = 'ym2i')
snapshots.add_task("integ(integ(tr*z,'z'),'y')", layout='g', name = 'zm1i')
snapshots.add_task("integ(integ(tr*z*z,'z'),'y')", layout='g', name = 'zm2i')


# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        #        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
