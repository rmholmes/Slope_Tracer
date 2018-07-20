"""
Dedalus scripts for 2D tracer advection-diffusion on a slope

This code uses a Fourier basis in the y direction with periodic boundary
conditions.

The functions can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.

"""
import numpy as np
import h5py
import matplotlib
import shutil
import time
from mpi4py import MPI
from scipy.special import erf
import os
from IPython import display
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools import post
import pathlib
from dedalus.extras import plot_tools
import obsfit1d
import logging
logger = logging.getLogger(__name__)

# Default parameters ------------------------------------------------------------------------
# Input Grids
Ly, Lz = (1500000., 3000.) # units = 1m
ny, nz = (384, 192)

# Physical parameters
N2 = 1.0e-6
slope = 1/400.0
Pr0 = 1.0

Kinf = 1.0e-5
K0 = 1.0e-3
d = 500.0

AH = 0.0

# Initial tracer parameters
trItype = 1 # 1 = point, 2 = layer
z0  = 0.5  # units of d
sz0 = 3.   # units of Lz/nz

c0  = 0.5  # units of Ly (point only)
sy0 = 3.   # units of Ly/ny (point only)

mxy0 = 0.  # units of Ly (layer only)
mny0 = 1.  # units of Ly (layer only)

# Advection type
ADV = 2

# Timing information
lday = 1.0e5 # A "long-day" unit (86400 ~= 100000)
dt=8*lday
Ttot = 3200
sfreq = 2

accept_list = ['Ly','Lz','ny','nz','N2','slope','Pr0',
               'Kinf','K0','d','AH','trItype','z0','sz0',
               'c0','sy0','mxy0','mny0','ADV','lday','dt',
               'Ttot','sfreq']
default_input_dict = {}
for i in accept_list:
    exec('default_input_dict[i] = %s' % i)

# Run a simulation -----------------------------------------------------------------------------------------
def run_sim(rundir,Ly,Lz,ny,nz,N2,slope,Pr0,
               Kinf,K0,d,AH,trItype,z0,sz0,
               c0,sy0,mxy0,mny0,ADV,lday,dt,
               Ttot,sfreq):

    # Create bases and domain
    y_basis = de.Fourier('y', ny, interval=(0, Ly))#, dealias=3/2)
    z_basis = de.Chebyshev('z', nz, interval=(0, Lz))#, dealias=3/2)
    domain = de.Domain([y_basis, z_basis], grid_dtype=np.float64)
    y = domain.grid(0)
    z = domain.grid(1)

    # Create input fields 

    # Isotropic Diffusivity
    K = domain.new_field()
    K.meta['y']['constant'] = True
    Kz = domain.new_field()
    Kz.meta['y']['constant'] = True
    K['g'] = Kinf + (K0-Kinf)*np.exp(-z/d)
    K.differentiate('z',out=Kz)

    # Upslope Velocity
    PSI = domain.new_field()
    PSI.meta['y']['constant'] = True
    V = domain.new_field()
    V.meta['y']['constant'] = True

    theta = np.arctan(slope)
    q0 = (N2*np.sin(theta)*np.sin(theta)/4.0/Pr0/K0/K0)**(1.0/4.0)

    PSI['g'] = np.cos(theta)/np.sin(theta)*(1.0-np.exp(-q0*z)*(np.cos(q0*z)+np.sin(q0*z)))

    if ADV == 2:
        PSI['g'] = PSI['g']*K['g']  # SML + BBL
    elif ADV == 1:
        PSI['g'] = PSI['g']*K0      # BBL
    else:
        PSI['g'] = 0.0              # No ADV

    PSI.differentiate('z',out=V)

    # Buoyancy field:
    By = N2*np.sin(theta)
    Bz = domain.new_field();Bz.meta['y']['constant'] = True
    B = domain.new_field()
    B['g'] = N2*np.sin(theta)*y + N2*np.cos(theta)*(z + np.exp(-q0*z)*np.cos(q0*z)/q0)
    f = domain.new_field();f.meta['y']['constant'] = True
    f['g'] = np.exp(-q0*z)*(np.cos(q0*z)+np.sin(q0*z))
    Bz['g'] = N2*np.cos(theta)*(1.-f['g'])

    # Equations and Solver
    problem = de.IVP(domain, variables=['tr','trz'])
    problem.meta[:]['z']['dirichlet'] = True

    problem.parameters['K'] = K
    problem.parameters['Kz'] = Kz
    problem.parameters['AH'] = AH
    problem.parameters['V'] = V
    problem.parameters['By'] = By
    problem.parameters['Bz'] = Bz
    problem.parameters['B'] = B

    # # Full Equation with analytic derivatives:
    # problem.parameters['tanth'] = np.tan(theta)
    # problem.parameters['cotth'] = 1/np.tan(theta)
    # problem.parameters['f'] = f
    # problem.parameters['fz'] = fz
    # problem.substitutions['Bz2d'] = '(1-f)**2./(tanth**2.+(1-f)**2.)'
    # problem.substitutions['ByBzd'] = '(1-f)/(tanth+cotth*(1-f)**2.)'
    # problem.substitutions['By2d'] = '1/(1+cotth**2.*(1-f)**2.)'
    # problem.substitutions['ByBzdd'] = 'fz*(cotth*(1-f)**2.-tanth)/(tanth+cotth*(1-f)**2.)**2.'
    # problem.substitutions['By2dd'] = '2*(1-f)*fz/(1+cotth**2.*(1-f)**2.)**2.'
    # problem.add_equation("dt(tr) + V*dy(tr) - (AH*Bz2d + K)*d(tr,y=2) + 2*AH*ByBzd*dy(trz) + AH*ByBzdd*dy(tr) - (AH*By2dd+Kz)*trz - (AH*By2d+K)*dz(trz) = 0.")
    # Comments:
    # - Try formulating equation with the flux
    # - Try the trick for each individual coefficient.
    # # Only Interior buoyancy influences AH (but full B used for binning):
    problem.parameters['costh'] = np.cos(theta)
    problem.parameters['sinth'] = np.sin(theta)
    problem.add_equation("dt(tr) + V*dy(tr) - (AH*costh**2. + K)*d(tr,y=2) + 2*AH*sinth*costh*dy(trz) - Kz*trz - (AH*sinth**2.+K)*dz(trz) = 0.")

    problem.add_equation("trz - dz(tr) = 0")
    problem.add_bc("left(trz) = 0")
    problem.add_bc("right(trz) = 0")

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK222)
    logger.info('Solver built')

    # Initial condition:
    tr = solver.state['tr']
    trz = solver.state['trz']
    
    if (trItype == 1):
        # Gaussian blob:
        cz = d*z0;sy = Ly/ny*sy0;sz = Lz/nz*sz0;cy = Ly*c0;
        tr['g'] = np.exp(-(z-cz)**2/2/sz**2 -(y-cy)**2/2/sy**2)
    elif (trItype == 2):
        # Function of buoyancy:
        hvs = np.ones_like(y);hvs[y <= mny0*Ly] = 0.;hvs[y >= mxy0*Ly] = 0.
        tr['g'] = 0*z
        tr['g'] = np.exp(-(B['g']/N2/np.cos(theta) - cz)**2/2/sz**2)*hvs
    else:
        "ERROR: Must pick a valid initial tracer distribution"
        return
    tr.differentiate('z',out=trz)

    # Integration parameters
    solver.stop_sim_time = np.inf
    solver.stop_wall_time = np.inf 
    solver.stop_iteration = Ttot*lday/dt
    Itot = solver.stop_iteration

    # Save parameters:
    np.savez(rundir + 'runparams',Ly=Ly,Lz=Lz,N2=N2,slope=slope,theta=theta,Pr0=Pr0,Kinf=Kinf,K0=K0,d=d,AH=AH,
             q0=q0,By=By,sy=sy,sz=sz,cy=cy,cz=cz,lday=lday,dt=dt,sfreq=sfreq,Itot=Itot,Ttot=Ttot,mxy0=mxy0,mny0=mny0)

    ## Analysis
    # Input fields file:
    ifields = solver.evaluator.add_file_handler(rundir + 'ifields', iter=5000000000000, max_writes=20000)
    ifields.add_task("B", layout='g', name = 'B')
    ifields.add_task("K", layout='g', name = 'K')
    ifields.add_task("V", layout='g', name = 'V')
    ifields.add_task("Bz", layout='g', name = 'Bz')

    # Snapshots file:
    snapshots = solver.evaluator.add_file_handler(rundir + 'snapshots', iter=sfreq, max_writes=20000)
    snapshots.add_system(solver.state, layout='g')
    snapshots.add_task("integ(tr,'y')", layout='g', name = 'ym0')
    snapshots.add_task("integ(tr*y,'y')", layout='g', name = 'ym1')
    snapshots.add_task("integ(tr*y*y,'y')", layout='g', name = 'ym2')
    snapshots.add_task("integ(tr,'z')", layout='g', name = 'zm0')
    snapshots.add_task("integ(tr*z,'z')", layout='g', name = 'zm1')
    snapshots.add_task("integ(tr*z*z,'z')", layout='g', name = 'zm2')
    snapshots.add_task("integ(integ(tr,'y'),'z')", layout = 'g', name = 'trT')
    snapshots.add_task("integ(integ(trz,'y'),'z')", layout = 'g', name = 'trzT')
    snapshots.add_task("integ(integ(abs(trz),'y'),'z')", layout = 'g', name = 'atrzT')
    snapshots.add_task("integ(integ(trz*z,'y'),'z')", layout = 'g', name = 'trzzT')
    snapshots.add_task("integ(integ(trz*y,'y'),'z')", layout = 'g', name = 'trzyT')
    snapshots.add_task("integ(integ(tr*B,'y'),'z')", layout = 'g', name = 'bm1i')
    snapshots.add_task("integ(integ(tr*B*B,'y'),'z')", layout = 'g', name = 'bm2i')
    snapshots.add_task("integ(integ(tr*B*B*B,'y'),'z')", layout = 'g', name = 'bm3i')
    snapshots.add_task("integ(integ(tr*B*B*B*B,'y'),'z')", layout = 'g', name = 'bm4i')
    snapshots.add_task("integ(integ(tr*y,'y'),'z')", layout='g', name = 'ym1i')
    snapshots.add_task("integ(integ(tr*y*y,'y'),'z')", layout='g', name = 'ym2i')
    snapshots.add_task("integ(integ(tr*z,'z'),'y')", layout='g', name = 'zm1i')
    snapshots.add_task("integ(integ(tr*z*z,'z'),'y')", layout='g', name = 'zm2i')
    snapshots.add_task("integ(integ(tr*z*y,'z'),'y')", layout='g', name = 'yzmi')
    snapshots.add_task("integ(integ(tr*K,'z'),'y')", layout='g', name = 'Ktr')
    snapshots.add_task("integ(integ(trz*K,'z'),'y')", layout='g', name = 'Ktrz')
    snapshots.add_task("integ(integ(abs(trz)*K,'z'),'y')", layout='g', name = 'Katrz')
    snapshots.add_task("integ(integ(trz*K*y,'z'),'y')", layout='g', name = 'Kytrz')
    snapshots.add_task("integ(integ(tr*V,'z'),'y')", layout='g', name = 'Vtr')
    snapshots.add_task("integ(integ(tr*V*y,'z'),'y')", layout='g', name = 'Vytr')
    snapshots.add_task("integ(integ(tr*V*z,'z'),'y')", layout='g', name = 'Vztr')

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


def merge_move(rundir,outdir):

    # Merge snapshots from different processes:
    post.merge_process_files(rundir + "snapshots", cleanup=True)
    set_paths = list(pathlib.Path(rundir + "snapshots").glob("snapshots_s*.h5"))
    post.merge_sets(rundir + "snapshots/snapshots.h5", set_paths, cleanup=True)
    # ifields
    post.merge_process_files(rundir + "ifields", cleanup=True)
    set_paths = list(pathlib.Path(rundir + "ifields").glob("ifields_s*.h5"))
    post.merge_sets(rundir + "ifields/ifields.h5", set_paths, cleanup=True)

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank   = comm.Get_rank()
rundir = '/home/z3500785/dedalus_rundir/';
outbase = '/srv/ccrc/data03/z3500785/dedalus_Slope_Tracer/saveRUNS/';

# Test runs:
AHs = [5.,5.,10.,10.,50.,50.]
ADVs = [0,1,0,1,0,1]
outfold = outbase + 'prodruns_wide30-5-19/'
for ii in range(len(AHs)):
    
    input_dict = default_input_dict.copy()
    input_dict['AH'] = AHs[ii]
    input_dict['ADV'] = ADVs[ii]
    run_sim(rundir,**input_dict)
    outdir = outfold + 'z0_0p5000_AH_%03d_ADV_%01d/' % (AHs[ii],ADVs[ii])
    print(outdir)
    merge_move(rundir,outdir)

    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
        shutil.move(rundir + 'snapshots/snapshots.h5',outdir + 'snapshots.h5');
        shutil.move(rundir + 'ifields/ifields.h5',outdir + 'ifields.h5');
        shutil.move(rundir + 'runparams.npz',outdir + 'runparams.npz');
        shutil.rmtree(rundir + 'snapshots/');
        shutil.rmtree(rundir + 'ifields/');


