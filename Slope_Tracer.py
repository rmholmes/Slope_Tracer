
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
import matplotlib.pyplot as plt
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
Prv0 = 1.0 # upslope Pr (in q0)
SPru0i = 0.0 # across-slope (S*Pru0)^(-1) (interior db/dz reduction)

Kinf = 1.0e-5
K0 = 1.0e-3
d = 500.0
Kred = 0 # 1 = reduce to zero at bottom with scale Kreds, 0 = don't.
Kreds = 0.5 # scale of exponential reduction (units of q0^{-1}).

AH = 0.0
AHvar = 1 # 1 = Reduced in BBL, 0 = Constant
AHfull = 0 # 0 = horizontal diffusion, 1 = along-isopycnal diffusion

# Initial tracer parameters
trItype = 1 # 1 = point, 2 = layer
z0  = 0.5  # units of d
sz0 = 3.   # units of Lz/nz

c0  = 0.5  # units of Ly (point only)
sy0 = 3.   # units of Ly/ny (point only)

mny0 = 0.  # units of Ly (layer only)
mxy0 = 1.  # units of Ly (layer only)

# Advection type
ADV = 2

# Timing information
lday = 1.0e5 # A "long-day" unit (86400 ~= 100000)
dt=8*lday
Ttot = 3200
sfreq = 2

accept_list = ['Ly','Lz','ny','nz','N2','slope','Prv0','SPru0i',
               'Kinf','K0','d','AH','AHvar','AHfull','trItype','z0','sz0',
               'c0','sy0','mxy0','mny0','ADV','lday','dt',
               'Ttot','sfreq','Kred','Kreds']
default_input_dict = {}
for i in accept_list:
    exec('default_input_dict[i] = %s' % i)

# Run a simulation -----------------------------------------------------------------------------------------
def run_sim(rundir,Ly,Lz,ny,nz,N2,slope,Prv0,SPru0i,
               Kinf,K0,d,AH,AHvar,AHfull,trItype,z0,sz0,
               c0,sy0,mxy0,mny0,ADV,lday,dt,
               Ttot,sfreq,Kred,Kreds,plot):

    # Create bases and domain
    y_basis = de.Fourier('y', ny, interval=(0, Ly))#, dealias=3/2)
    z_basis = de.Chebyshev('z', nz, interval=(0, Lz))#, dealias=3/2)
    domain = de.Domain([y_basis, z_basis], grid_dtype=np.float64)
    y = domain.grid(0)
    z = domain.grid(1)

    # Notes: dealias (n+1)/2 where n is the RHS non-linearity order (e.g. n2).
    # so dealias = 1 if no non-linearities

    # Create input fields

    # Isotropic Diffusivity
    K = domain.new_field()
    K.meta['y']['constant'] = True
    Kz = domain.new_field()
    Kz.meta['y']['constant'] = True
    K['g'] = Kinf + (K0-Kinf)*np.exp(-z/d)
    #    K.set_scales(domain.dealias)
    K.differentiate('z',out=Kz)
    
    # Upslope Velocity
    PSI = domain.new_field()
    PSI.meta['y']['constant'] = True
    V = domain.new_field()
    V.meta['y']['constant'] = True
    PSIbbl = domain.new_field()
    PSIbbl.meta['y']['constant'] = True
    Vbbl = domain.new_field()
    Vbbl.meta['y']['constant'] = True

    theta = np.arctan(slope)
    q0 = (N2*np.sin(theta)*np.sin(theta)/4.0/Prv0/K0/K0*(1.+SPru0i))**(1.0/4.0)

    PSI['g'] = np.cos(theta)/np.sin(theta)/(1.+SPru0i)*(1.0-np.exp(-q0*z)*(np.cos(q0*z)+np.sin(q0*z)))
    PSIbbl['g'] = PSI['g']*(K0+SPru0i*Kinf)

    if ADV == 2:
        PSI['g'] = PSI['g']*(K['g']+SPru0i*Kinf)  # SML + BBL
    elif ADV == 1:
        PSI['g'] = PSI['g']*(K0+SPru0i*Kinf)      # BBL
    else:
        PSI['g'] = 0.0              # No ADV
        PSIbbl['g'] = 0.0

    PSIbbl.differentiate('z',out=Vbbl)
    PSI.differentiate('z',out=V)

    # Depth-dependent horizontal diffusivity
    AHdd = domain.new_field()
    AHdd.meta['y']['constant'] = True
    if AHvar == 1:
        AHdd['g'] = AH*(1.-np.exp(-q0*z))
    else:
        AHdd['g'] = AH
        
    # BBL fluxes from thickness criteria:
    hvs = np.ones_like(z);
    hvs[z > np.pi/q0] = 0.
    Hbbl = domain.new_field()
    Hbbl.meta['y']['constant'] = True
    Hbbl['g'] = hvs

    # Buoyancy field:
    By = N2*np.sin(theta)
    Bz = domain.new_field();Bz.meta['y']['constant'] = True
    Bzp = domain.new_field();Bzp.meta['y']['constant'] = True
    BzpSML = domain.new_field();BzpSML.meta['y']['constant'] = True
    B = domain.new_field()
    B['g'] = N2*np.sin(theta)*y + N2*np.cos(theta)/(1.+SPru0i)*(z +
                np.exp(-q0*z)*np.cos(q0*z)/q0*(1.+SPru0i*Kinf/K0) +
                SPru0i*d*np.log(1.+Kinf/K0*(np.exp(z/d)-1.)))
    f = domain.new_field();f.meta['y']['constant'] = True
    f['g'] = np.exp(-q0*z)*(np.cos(q0*z)+np.sin(q0*z))
    Bzp['g'] = -N2*np.cos(theta)/(1+SPru0i)*f['g']
    Bz['g'] = (N2*np.cos(theta)/(1+SPru0i) + Bzp['g'])*(1+SPru0i*Kinf/K['g'])
    BzpSML['g'] = -N2*np.cos(theta)*(K0-Kinf)*np.exp(-z/d)*SPru0i/((1+SPru0i)*K['g'])
    # NOTE: SPru0i non-zero case only works with Kinf not equal to 0.

    # Artifically reduce K through BBL:
    if Kred == 1:
        K['g'] = K['g']*(1.-np.exp(-q0*z/Kreds))
        K.differentiate('z',out=Kz)

    # Equations and Solver
    problem = de.IVP(domain, variables=['tr','trz'], ncc_cutoff=1e-20)
    problem.meta[:]['z']['dirichlet'] = True

    problem.parameters['N2'] = N2
    problem.parameters['K'] = K
    problem.parameters['Kz'] = Kz
    problem.parameters['AHdd'] = AHdd
    problem.parameters['V'] = V
    problem.parameters['Vbbl'] = Vbbl
    problem.parameters['By'] = By
    problem.parameters['Bz'] = Bz
    problem.parameters['Bzp'] = Bzp
    problem.parameters['BzpSML'] = BzpSML
    problem.parameters['B'] = B
    problem.parameters['Hbbl'] = Hbbl
    problem.parameters['costh'] = np.cos(theta)
    problem.parameters['sinth'] = np.sin(theta)
    problem.parameters['SPru0i'] = SPru0i

    # Flux-formulation:
    # Advection and isotropic diffusion fluxes:
    problem.substitutions['Fy'] = "V*tr - K*dy(tr)" 
    problem.substitutions['Fz'] = "     - K*trz"

    # LHS K-tensor terms:
    if AHfull == 1:                  # Full along-isopycnal diffusion
        problem.parameters['f'] = f
    else:                            # Horizontal diffusion
        problem.parameters['f'] = 0.
    
    problem.substitutions['GB2'] = "1. + costh**2.*f*(f-2)"
    problem.substitutions['KHyy'] = "costh**2.*(1-f)**2./GB2"
    problem.substitutions['KHyz'] = "-costh*sinth*(1-f)/GB2"
    problem.substitutions['KHzz'] = "sinth**2./GB2"

    # LHS fluxes:
    problem.substitutions['FHy']   = "-AHdd*(KHyy*dy(tr) + KHyz*trz)"
    problem.substitutions['FHz']   = "-AHdd*(KHyz*dy(tr) + KHzz*trz)"

    problem.add_equation("dt(tr) + dy(Fy + FHy) + dz(Fz + FHz) = 0.")
    problem.add_equation("trz - dz(tr) = 0")
    problem.add_bc("left(trz) = 0")
    problem.add_bc("right(trz) = 0")

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK222)
    logger.info('Solver built')

    # Initial condition:
    tr = solver.state['tr']
    trz = solver.state['trz']
    
    cz = d*z0;sy = Ly/ny*sy0;sz = Lz/nz*sz0;cy = Ly*c0;
    if (trItype == 1):
        # Gaussian blob:
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
    np.savez(rundir + 'runparams',Ly=Ly,Lz=Lz,N2=N2,slope=slope,theta=theta,Prv0=Prv0,SPru0i=SPru0i,Kinf=Kinf,K0=K0,d=d,AH=AH,AHvar=AHvar,AHfull=AHfull,
             q0=q0,By=By,sy=sy,sz=sz,cy=cy,cz=cz,lday=lday,dt=dt,sfreq=sfreq,Itot=Itot,Ttot=Ttot,mxy0=mxy0,mny0=mny0,Kred=Kred,Kreds=Kreds)

    ## Analysis
    # Input fields file:
    ifields = solver.evaluator.add_file_handler(rundir + 'ifields', iter=5000000000000, max_writes=20000)
    ifields.add_task("B", layout='g', name = 'B')
    ifields.add_task("K", layout='g', name = 'K')
    ifields.add_task("AHdd", layout='g', name = 'AHdd')
    ifields.add_task("V", layout='g', name = 'V')
    ifields.add_task("Bz", layout='g', name = 'Bz')
    ifields.add_task("Bzp", layout='g', name = 'Bzp')
    ifields.add_task("BzpSML", layout='g', name = 'BzpSML')
    ifields.add_task("1. + costh**2.*f*(f-2)", layout='g', name='GB2')
    ifields.add_task("costh**2.*(1-f)**2./(1. + costh**2.*f*(f-2))", layout='g', name='KHyy')
    ifields.add_task("-costh*sinth*(1-f)/(1. + costh**2.*f*(f-2))", layout='g', name='KHyz')
    ifields.add_task("sinth**2./(1. + costh**2.*f*(f-2))", layout='g', name='KHzz')

    # Snapshots file:
    snapshots = solver.evaluator.add_file_handler(rundir + 'snapshots', iter=sfreq, max_writes=20000)
    snapshots.add_system(solver.state, layout='g')
    snapshots.add_task("tr*V", layout='g', name = 'advFy')
    snapshots.add_task("integ(tr,'y')", layout='g', name = 'ym0')
    snapshots.add_task("integ(tr*y,'y')", layout='g', name = 'ym1')
    snapshots.add_task("integ(tr*y*y,'y')", layout='g', name = 'ym2')
    snapshots.add_task("integ(tr,'z')", layout='g', name = 'zm0')
    snapshots.add_task("integ(tr*z,'z')", layout='g', name = 'zm1')
    snapshots.add_task("integ(tr*z*z,'z')", layout='g', name = 'zm2')
    snapshots.add_task("integ(integ(tr,'y'),'z')", layout = 'g', name = 'trT')
    snapshots.add_task("integ(integ(tr*y,'y'),'z')", layout='g', name = 'ym1i')
    snapshots.add_task("integ(integ(tr*z,'z'),'y')", layout='g', name = 'zm1i')


    # Moment time series file:
    moments = solver.evaluator.add_file_handler(rundir + 'moments', iter=1, max_writes=20000)

    moments.add_task("integ(integ(tr,'y'),'z')", layout = 'g', name = 'trT')

    moments.add_task("integ(integ(trz,'y'),'z')", layout = 'g', name = 'trzT')
    moments.add_task("integ(integ(abs(trz),'y'),'z')", layout = 'g', name = 'atrzT')
    moments.add_task("integ(integ(trz*z,'y'),'z')", layout = 'g', name = 'trzzT')
    moments.add_task("integ(integ(trz*y,'y'),'z')", layout = 'g', name = 'trzyT')
    
    moments.add_task("integ(integ(tr*B,'y'),'z')", layout = 'g', name = 'bm1i')
    moments.add_task("integ(integ(tr*B*B,'y'),'z')", layout = 'g', name = 'bm2i')
    moments.add_task("integ(integ(tr*B*B*B,'y'),'z')", layout = 'g', name = 'bm3i')
    moments.add_task("integ(integ(tr*B*B*B*B,'y'),'z')", layout = 'g', name = 'bm4i')
    
    moments.add_task("integ(integ(tr*y,'y'),'z')", layout='g', name = 'ym1i')
    moments.add_task("integ(integ(tr*y*y,'y'),'z')", layout='g', name = 'ym2i')
    moments.add_task("integ(integ(tr*z,'z'),'y')", layout='g', name = 'zm1i')
    moments.add_task("integ(integ(tr*z*z,'z'),'y')", layout='g', name = 'zm2i')
    moments.add_task("integ(integ(tr*z*y,'z'),'y')", layout='g', name = 'yzmi')
    
    moments.add_task("integ(integ(tr*K,'z'),'y')", layout='g', name = 'Ktr')
    moments.add_task("integ(integ(trz*K,'z'),'y')", layout='g', name = 'Ktrz')
    moments.add_task("integ(integ(abs(trz)*K,'z'),'y')", layout='g', name = 'Katrz')
    moments.add_task("integ(integ(trz*K*y,'z'),'y')", layout='g', name = 'Kytrz')
    
    moments.add_task("integ(integ(tr*V*y,'z'),'y')", layout='g', name = 'Vytr')
    moments.add_task("integ(integ(tr*V*z,'z'),'y')", layout='g', name = 'Vztr')

    # Z VAR terms:
    moments.add_task("integ(integ(z*Kz*tr,'z'),'y')", layout='g', name = 'zKztr')

    # B COM terms:
    moments.add_task("integ(integ(tr*V*By,'z'),'y')", layout='g', name = 'VtrBy')
    moments.add_task("integ(integ(tr*Vbbl*By,'z'),'y')", layout='g', name = 'VbbltrBy')
    moments.add_task("integ(integ(tr*V*Hbbl*By,'z'),'y')", layout='g', name = 'VbblTtrBy')

    moments.add_task("integ(integ(K*dy(tr)*By,'z'),'y')", layout='g', name = 'KtryBy')
    moments.add_task("integ(integ(K*Hbbl*dy(tr)*By,'z'),'y')", layout='g', name = 'KbblTtryBy')
    moments.add_task("integ(integ(Kz*tr*N2*costh,'z'),'y')", layout='g', name = 'KztrBZ')
    moments.add_task("integ(integ(K*trz*Bz,'z'),'y')", layout='g', name = 'KtrzBz')
    moments.add_task("integ(integ(K*trz*Bzp,'z'),'y')", layout='g', name = 'KtrzBzp')
    moments.add_task("integ(integ(K*trz*BzpSML,'z'),'y')", layout='g', name = 'KtrzBzpSML')
    moments.add_task("integ(integ(K*Hbbl*trz*Bz,'z'),'y')", layout='g', name = 'KbblTtrzBz')

    moments.add_task("integ(integ(trz*Bzp,'z'),'y')", layout='g', name = 'trzBzp')
    moments.add_task("integ(integ(trz*BzpSML,'z'),'y')", layout='g', name = 'trzBzpSML')

    # B VAR terms:
    moments.add_task("integ(integ(tr*B*V*By,'z'),'y')", layout='g', name = 'VtrBBy')
    moments.add_task("integ(integ(tr*B*Vbbl*By,'z'),'y')", layout='g', name = 'VbbltrBBy')
    moments.add_task("integ(integ(tr*B*V*Hbbl*By,'z'),'y')", layout='g', name = 'VbblTtrBBy')
    moments.add_task("integ(integ(B*K*dy(tr)*By,'z'),'y')", layout='g', name = 'KtryBBy')
    moments.add_task("integ(integ(B*K*trz*Bz,'z'),'y')", layout='g', name = 'KtrzBBz')
    moments.add_task("integ(integ(B*K*trz*Bzp,'z'),'y')", layout='g', name = 'KtrzBBzp')
    moments.add_task("integ(integ(tr*K*Bzp*N2*costh,'z'),'y')", layout='g', name = 'KtrBZBzp')
    moments.add_task("integ(integ(B*K*trz*BzpSML,'z'),'y')", layout='g', name = 'KtrzBBzpSML')
    moments.add_task("integ(integ(tr*K*BzpSML*N2*costh,'z'),'y')", layout='g', name = 'KtrBZBzpSML')
    moments.add_task("integ(integ(B*Kz*tr*N2*costh,'z'),'y')", layout='g', name = 'KztrBBZ')
    moments.add_task("integ(integ(B*K*Hbbl*dy(tr)*By,'z'),'y')", layout='g', name = 'KbblTtryBBy')
    moments.add_task("integ(integ(B*K*Hbbl*trz*Bz,'z'),'y')", layout='g', name = 'KbblTtrzBBz')

    # Approximation check terms:
    moments.add_task("integ(integ(AHdd*sinth*Bzp*(trz*sinth-dy(tr)*costh),'z'),'y')", layout='g', name = 'AHbm1')
    moments.add_task("integ(integ(AHdd*sinth*B*Bzp*(trz*sinth-dy(tr)*costh),'z'),'y')", layout='g', name = 'AHbm2')
    
    # Plotting:
    if plot:
        f, ax = plt.subplots(figsize=(10,5))
        f.set_facecolor('white')
        ax.set_xlabel('true y (km)');ax.set_ylabel('true z (m)')
        y = domain.grid(0,scales=domain.dealias)
        z = domain.grid(1,scales=domain.dealias)
        ym, zm = np.meshgrid(y,z)
        zt = np.cos(theta)*zm + np.sin(theta)*ym
        yt = -np.sin(theta)*zm + np.cos(theta)*ym
        p = ax.pcolormesh(yt/1.0e3, zt, tr['g'].T/np.max(tr['g']), cmap='RdBu_r', vmin=-1., vmax=1.);
        Buo = N2*np.sin(theta)*ym + N2*np.cos(theta)*(zm + np.exp(-q0*zm)*np.cos(q0*zm)/q0)
        # NOTE: This plotting only works with SPru0i=0
        ax.contour(yt/1.0e3, zt, Buo, 30, colors='k')
        ax.plot(y/1.0e3, slope*y,'k-', linewidth=4)
        plt.colorbar(p, ax = ax)
        ax.set_xlim([0,Ly/1.e3]);ax.set_ylim([0.,Lz + slope*Ly]);
        ax.set_facecolor('k')
        ax.set_title('Normalized Tracer Concentration')

    # Main loop
    try:
        logger.info('Starting loop')
        start_time = time.time()
        while solver.ok:
            #        dt = CFL.compute_dt()
            solver.step(dt)
            if (plot and (solver.iteration-1) % 5 == 0):
                p.set_array(np.ravel(tr['g'][:-1,:-1].T/np.max(tr['g'])))
                display.clear_output()
                display.display(f)
            if (solver.iteration-1) % 2 == 0:
                logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            assert (np.max(tr['g'][:,:])<10.),'blow-up'
            
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        end_time = time.time()
        if plot:
            p.set_array(np.ravel(tr['g'][:-1,:-1].T/np.max(tr['g'])))
            display.clear_output()
            display.display(f)

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
    # moments
    post.merge_process_files(rundir + "moments", cleanup=True)
    set_paths = list(pathlib.Path(rundir + "moments").glob("moments_s*.h5"))
    post.merge_sets(rundir + "moments/moments.h5", set_paths, cleanup=True)

    
if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank   = comm.Get_rank()
    rundir = '/short/e14/rmh561/dedalus/Slope_Tracer/rundir/';
    outbase = '/g/data/e14/rmh561/Slope_Tracer/saveRUNS/';
    # outfold = outbase + 'prodruns_layer30-08-18/'
    outfold = outbase + 'prodruns24-08-18/'

    plot = False

    # # Production runs Point-Release -------------------
    # # AH=0:
    # ADVs   = [0,0,0,1,1,1,2,2,2]
    # Kinfs  = [1.e-5,1.e-4,1.e-3] *3
    # z0s    = [0.5] * 9
    # slopes = [1./400.] * 9

    # slopes.extend([1./200.]* 3 + [1./100.]*3)
    # ADVs.extend([0,1,2] * 2)
    # Kinfs.extend([1.e-5] * 6)
    # z0s.extend([0.5] * 6)

    # z0s.extend([0.125,0.25,1.,2.])
    # ADVs.extend([2]*4)
    # Kinfs.extend([1.e-5]*4)
    # slopes.extend([1./400.]*4)

    # AHs    = [0.] * len(ADVs)

    # More z0 runs Point-Release -------------------

    # z0s = [0.0625, 0.375, 0.75, 1.5]
    # AHs = [0.] * len(z0s)
    
    # z0s = [0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.5, 2.]
    # AHs = [100.] * 8

    # ADVs   = [2] * len(z0s)
    # Kinfs  = [1.e-5] * len(z0s)
    # slopes = [1./400.] * len(z0s)

    # Production runs Layer-Release -----------------
    # if end spacing is dz (e.g. 500m), then mny0 = (z0*d-dz)/(slope*Ly)
    # mny0s = [1.4/3.,1.6/3.]
    # AHs = [10.,10.]

    # AHs = [0.,20.,30.,40.,50.,60.,70.,80.,90.]
    # mny0s = [1.6/3.] * len(AHs)

    # # Larger Ly runs:
    # AHs = [100.,125.,150.,175.,200.]
    # mny0s = [1.6/3.] * len(AHs)

#    for ii in range(len(AHs)):
    for ii in range(1):#len(AHs)):

        input_dict = default_input_dict.copy()

        # # Production isoAH (true along-isopycnal) point-release runs:
        # input_dict['dt'] = 4*lday
        # input_dict['sfreq'] = 4
        # input_dict['nz'] = 768
        # input_dict['ny'] = 576
        # input_dict['sz0'] = 3.*768./192.
        # input_dict['sy0'] = 3.*576./384.
        # input_dict['AH'] = AHs[ii]
        # input_dict['AHvar'] = 0
        # input_dict['AHfull'] = 1
        # outdir = outfold + 'z0_0p5000_AH_%03d_ADV_2_Kinf_m5_slope_400_isoAH/' % (AHs[ii])

#         # Production isoAH (true along-isopycnal) layer-release runs:
#         input_dict['dt'] = 4*lday
#         input_dict['sfreq'] = 4
# #        input_dict['nz'] = 768
#         input_dict['ny'] = 576
#         input_dict['AHvar'] = 0
#         input_dict['AHfull'] = 1

#         input_dict['slope'] = 1/200.        
#         input_dict['AH'] = AHs[ii]
#         input_dict['mny0'] = mny0s[ii]

#         input_dict['z0'] = 10.
#         input_dict['trItype'] = 2
#         input_dict['sz0'] = 3.*768./192.

#         input_dict['Lz'] = 4000.
#         input_dict['nz'] = 1024

#         mny0str  = ('%0.4f' % mny0s[ii]).replace('.','p')
#         outdir = outfold + 'AH_%03d_ADV_2_Kinf_m5_mny0_%s_slope_200_isoAH_Lz4000/' % (AHs[ii],mny0str)

        # BBTRE run:
        input_dict['dt']     = 4*lday
        input_dict['Ttot']   = 6400
        input_dict['sfreq']  = 4
        input_dict['ny']     = 576
        input_dict['nz']     = 1024
        input_dict['Lz']     = 4000.
        input_dict['AHvar']  = 0
        input_dict['AHfull'] = 1

        input_dict['slope']  = 1/500.        
        input_dict['AH']     = 100.
        input_dict['N2']     = 1.69e-6
        input_dict['d']      = 230.
        input_dict['SPru0i'] = 1.95
        input_dict['K0']     = 1.8e-3
        input_dict['Kinf']   = 5.2e-5

        input_dict['z0']     = 4.3
        input_dict['sz0']    = 3.*1024./192.
        input_dict['sy0']    = 3.*576./384.

        outdir = outfold + 'BBTRE/'

        # # K BBL variation testing:
        # input_dict['dt']     = 4*lday
        # input_dict['sfreq']  = 4
        # input_dict['ny']     = 384
        # input_dict['nz']     = 768

        # input_dict['sz0']    = 3.*2.
        # input_dict['sy0']    = 3.*2.
        # input_dict['Kred']   = 1

        # outdir = outfold + 'Kredtest_Kred_Kreds_0p5/'

        
        # z0str = ('%1.4f' % z0s[ii]).replace('.','p')
        # Kinfstr = ('%01d' % np.log10(Kinfs[ii])).replace('-','m')
        # slopestr = '%03d' % (1./slopes[ii])

        run_sim(rundir,plot=plot,**input_dict)
        print(outdir)
        merge_move(rundir,outdir)
        
        if rank == 0:
            os.makedirs(outdir, exist_ok=True)
            shutil.move(rundir + 'snapshots/snapshots.h5',outdir + 'snapshots.h5');
            shutil.move(rundir + 'moments/moments.h5',outdir + 'moments.h5');
            shutil.move(rundir + 'ifields/ifields.h5',outdir + 'ifields.h5');
            shutil.move(rundir + 'runparams.npz',outdir + 'runparams.npz');
            shutil.rmtree(rundir + 'snapshots/');
            shutil.rmtree(rundir + 'ifields/');
            shutil.rmtree(rundir + 'moments/');


