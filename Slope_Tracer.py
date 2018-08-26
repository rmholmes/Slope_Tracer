
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
Pr0 = 1.0

Kinf = 1.0e-5
K0 = 1.0e-3
d = 500.0

AH = 0.0
AHvar = 1 # 1 = Reduced in BBL, 0 = Constant

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

accept_list = ['Ly','Lz','ny','nz','N2','slope','Pr0',
               'Kinf','K0','d','AH','AHvar','trItype','z0','sz0',
               'c0','sy0','mxy0','mny0','ADV','lday','dt',
               'Ttot','sfreq']
default_input_dict = {}
for i in accept_list:
    exec('default_input_dict[i] = %s' % i)

# Run a simulation -----------------------------------------------------------------------------------------
def run_sim(rundir,Ly,Lz,ny,nz,N2,slope,Pr0,
               Kinf,K0,d,AH,AHvar,trItype,z0,sz0,
               c0,sy0,mxy0,mny0,ADV,lday,dt,
               Ttot,sfreq,plot):

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
    q0 = (N2*np.sin(theta)*np.sin(theta)/4.0/Pr0/K0/K0)**(1.0/4.0)

    PSI['g'] = np.cos(theta)/np.sin(theta)*(1.0-np.exp(-q0*z)*(np.cos(q0*z)+np.sin(q0*z)))
    PSIbbl['g'] = PSI['g']*K0

    if ADV == 2:
        PSI['g'] = PSI['g']*K['g']  # SML + BBL
    elif ADV == 1:
        PSI['g'] = PSI['g']*K0      # BBL
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
    B = domain.new_field()
    B['g'] = N2*np.sin(theta)*y + N2*np.cos(theta)*(z + np.exp(-q0*z)*np.cos(q0*z)/q0)
    f = domain.new_field();f.meta['y']['constant'] = True
    f['g'] = np.exp(-q0*z)*(np.cos(q0*z)+np.sin(q0*z))
    Bzp['g'] = -N2*np.cos(theta)*f['g']
    Bz['g'] = N2*np.cos(theta) + Bzp['g']

    # Equations and Solver
    problem = de.IVP(domain, variables=['tr','trz'])
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
    problem.parameters['B'] = B
    problem.parameters['Hbbl'] = Hbbl
    problem.parameters['costh'] = np.cos(theta)
    problem.parameters['sinth'] = np.sin(theta)

    # Only Interior buoyancy influences AH (but full B used for binning):
    # problem.add_equation("dt(tr) + V*dy(tr) - (AH*costh**2. + K)*d(tr,y=2) + 2*AH*sinth*costh*dy(trz) - Kz*trz - (AH*sinth**2.+K)*dz(trz) = 0.")

    # Flux-formulation:
    # Advection and isotropic diffusion fluxes:
    problem.substitutions['Fy'] = "V*tr - K*dy(tr)" 
    problem.substitutions['Fz'] = "     - K*trz"
    # LHS K-tensor terms:
    # problem.parameters['f'] = f
    # problem.substitutions['GB2'] = "1 + costh**2.*f*(f-2)"
    # problem.substitutions['KHyy'] = "sinth**2./GB2"
    # problem.substitutions['KHyz'] = "-costh*sinth*(1-f)/GB2"
    # problem.substitutions['KHzz'] = "costh**2.*(1-f)**2./GB2"
    problem.substitutions['KHyy'] = "costh**2"
    problem.substitutions['KHyz'] = "-costh*sinth"
    problem.substitutions['KHzz'] = "sinth**2."

    # Two options:
    # - Try uping the level of non-constant coefficients cutoff (how many coefficient it uses to expand exponentials.
    # - Adding diffusion to prevent negative diffusivities (look for theory on diagnoally dominant matrices). (off-diagnal can be negative
    # - Approximate every single term with a polynomial (and make sure matrix is positive definite). Make sure polynomials are less than half resolution.
    
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
    np.savez(rundir + 'runparams',Ly=Ly,Lz=Lz,N2=N2,slope=slope,theta=theta,Pr0=Pr0,Kinf=Kinf,K0=K0,d=d,AH=AH,AHvar=AHvar,
             q0=q0,By=By,sy=sy,sz=sz,cy=cy,cz=cz,lday=lday,dt=dt,sfreq=sfreq,Itot=Itot,Ttot=Ttot,mxy0=mxy0,mny0=mny0)

    ## Analysis
    # Input fields file:
    ifields = solver.evaluator.add_file_handler(rundir + 'ifields', iter=5000000000000, max_writes=20000)
    ifields.add_task("B", layout='g', name = 'B')
    ifields.add_task("K", layout='g', name = 'K')
    ifields.add_task("AHdd", layout='g', name = 'AHdd')
    ifields.add_task("V", layout='g', name = 'V')
    ifields.add_task("Bz", layout='g', name = 'Bz')

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

    # B COM terms:
    moments.add_task("integ(integ(tr*V*By,'z'),'y')", layout='g', name = 'VtrBy')
    moments.add_task("integ(integ(tr*Vbbl*By,'z'),'y')", layout='g', name = 'VbbltrBy')
    moments.add_task("integ(integ(tr*V*Hbbl*By,'z'),'y')", layout='g', name = 'VbblTtrBy')

    moments.add_task("integ(integ(K*dy(tr)*By,'z'),'y')", layout='g', name = 'KtryBy')
    moments.add_task("integ(integ(K*Hbbl*dy(tr)*By,'z'),'y')", layout='g', name = 'KbblTtryBy')
    moments.add_task("integ(integ(Kz*tr*N2*costh,'z'),'y')", layout='g', name = 'KztrBZ')
    moments.add_task("integ(integ(K*trz*Bz,'z'),'y')", layout='g', name = 'KtrzBz')
    moments.add_task("integ(integ(K*trz*Bzp,'z'),'y')", layout='g', name = 'KtrzBzp')
    moments.add_task("integ(integ(K*Hbbl*trz*Bz,'z'),'y')", layout='g', name = 'KbblTtrzBz')

    moments.add_task("integ(integ(trz*Bzp,'z'),'y')", layout='g', name = 'trzBzp')

    # B VAR terms:
    moments.add_task("integ(integ(tr*B*V*By,'z'),'y')", layout='g', name = 'VtrBBy')
    moments.add_task("integ(integ(tr*B*Vbbl*By,'z'),'y')", layout='g', name = 'VbbltrBBy')
    moments.add_task("integ(integ(tr*B*V*Hbbl*By,'z'),'y')", layout='g', name = 'VbblTtrBBy')
    moments.add_task("integ(integ(B*K*dy(tr)*By,'z'),'y')", layout='g', name = 'KtryBBy')
    moments.add_task("integ(integ(B*K*trz*Bz,'z'),'y')", layout='g', name = 'KtrzBBz')
    moments.add_task("integ(integ(B*K*trz*Bzp,'z'),'y')", layout='g', name = 'KtrzBBzp')
    moments.add_task("integ(integ(tr*K*Bzp*N2*costh,'z'),'y')", layout='g', name = 'KtrBZBzp')
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
        p = ax.pcolormesh(yt/1.0e3, zt, tr['g'].T/np.max(tr['g']), cmap='RdBu_r', vmin=0., vmax=1.);
        Buo = N2*np.sin(theta)*ym + N2*np.cos(theta)*(zm + np.exp(-q0*zm)*np.cos(q0*zm)/q0)
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
    rundir = '/home/z3500785/dedalus_rundir/';
    outbase = '/srv/ccrc/data03/z3500785/dedalus_Slope_Tracer/saveRUNS/';
    outfold = outbase + 'prodruns24-08-18/'

    plot = False
    # Production runs -------------------
    # AH=0:
    ADVs   = [0,0,0,1,1,1,2,2,2]
    Kinfs  = [1.e-5,1.e-4,1.e-3] *3
    z0s    = [0.5] * 9
    slopes = [1./400.] * 9

    slopes.extend([1./200.]* 3 + [1./100.]*3)
    ADVs.extend([0,1,2] * 2)
    Kinfs.extend([1.e-5] * 6)
    z0s.extend([0.5] * 6)

    z0s.extend([0.125,0.25,1.,2.])
    ADVs.extend([2]*4)
    Kinfs.extend([1.e-5]*4)
    slopes.extend([1./400.]*4)

    AHs    = [0.] * len(ADVs)

    # AH non-zero:
    AHs.extend([10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,125.,150.,175.,200.] * 3)
    ADVs.extend([0] * 14 + [1]*14 + [2]*14)
    z0s.extend([0.5] * 14 * 3)
    slopes.extend([1./400.] * 14 * 3)
    Kinfs.extend([1.e-5] * 14 * 3)

    AHs.extend([10.,50.,100.,150.])
    Kinfs.extend([1.e-3] * 4)
    ADVs.extend([0] * 4)
    z0s.extend([0.5] * 4)
    slopes.extend([1./400.] * 4)

    # # Test runs:
    # AHs = [0.,0.,0.,100.,100.]
    # ADVs = [0,0,2,0,2]
    # Kinfs = [1.e-3] + [1.e-5] * 4
    # slopes = [1./400.] * 5
    # z0s = [0.5] * 5

    for ii in range(len(AHs)):

        input_dict = default_input_dict.copy()
        input_dict['z0'] = z0s[ii]
        input_dict['AH'] = AHs[ii]
        input_dict['ADV'] = ADVs[ii]
        input_dict['Kinf'] = Kinfs[ii]
        input_dict['slope'] = slopes[ii]
        run_sim(rundir,plot=plot,**input_dict)
        z0str = ('%1.4f' % z0s[ii]).replace('.','p')
        Kinfstr = ('%01d' % np.log10(Kinfs[ii])).replace('-','m')
        slopestr = '%03d' % (1./slopes[ii])
        outdir = outfold + 'z0_%s_AH_%03d_ADV_%01d_Kinf_%s_slope_%s/' % (z0str,AHs[ii],ADVs[ii],Kinfstr,slopestr)
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


