import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u

from multiprocessing import Pool

import forward_modelling as fm
from gmode_series import asymptotic
from observations import pulsations as obspuls

"""
    A collection of various testing routines / demos of the subroutines and classes in AMIGO.
"""

def sampling_test():
    """
        A routine to test (and demonstrate) how to use the forward_modelling.sampling() subroutine
    """
    
    parameters = ['mass','fov','mlt','f_rot']
    minbound = np.array([1.2, 0.001,1.2,0.0])
    maxbound = np.array([2.4,0.041,2.4,2.5])
    regular_step = np.array([0.4,0.008,0.4,0.1])
    
    method = ['regular','sobol','sobol','random']
    
    sample = fm.sampling(parameters,minbound,maxbound,method=method,regular_step=regular_step,Nsobol=100,Nrandom=4)
    
    # Plotting the samples in the parameter space
    plt.figure(1)
    plt.subplot(221)
    plt.plot(sample[parameters[0]],sample[parameters[2]],'b.')
    plt.xlabel(parameters[0])
    plt.ylabel(parameters[2])
    
    plt.subplot(222)
    plt.plot(sample[parameters[0]],sample[parameters[3]],'r.')  
    plt.xlabel(parameters[0])
    plt.ylabel(parameters[3])
    
    
    plt.subplot(223)
    plt.plot(sample[parameters[1]],sample[parameters[2]],'c.')
    plt.xlabel(parameters[1])
    plt.ylabel(parameters[2])
    
    plt.subplot(224)
    plt.plot(sample[parameters[1]],sample[parameters[3]],'k.',c='orange')
    plt.xlabel(parameters[1])
    plt.ylabel(parameters[3])
    
    plt.show()



def chi_squared_test(ind):
    """A simple chi^2 calculation, used in the demo in __main__."""
    
    pattern = k0m1.uniform_pattern(sample['f_rot'][ind]/u.day,sample['Pi0'][ind]*u.s)
    chisq = np.sum((np.diff(obsstar.period) - np.interp(obsstar.period[:-1],pattern[:-1],np.diff(pattern))*u.day)**2./(obsstar.e_period[1:]**2. + obsstar.e_period[:-1]**2.))
    
    return chisq
        

if(__name__ == '__main__'):
    
    # Testing the forward_modelling.sampling() subroutine
    sampling_test()
    
    ### A simple test/demo on how to combine the routines to do model fitting
    ### At the moment, this still uses a simple chi^2; AIM: switch to Aerts et al. 2018.
    
    # Again, the sampling
    parameters = ['Pi0','f_rot']
    minbound = np.array([2300., 0.]) 
    maxbound = np.array([5600.,2.5]) 
    regular_step = np.array([10.,0.005]) 
    method = 'regular'
    
    sample = fm.sampling(parameters,minbound,maxbound,method=method,regular_step=regular_step)
    
    # asymptotic object initialisation
    gyre_dir = '/lhome/timothyv/Bin/mesa/mesa-10108/gyre/gyre/'
    kval = 0
    mval = 1
    
    k0m1 = asymptotic(gyre_dir,kval=kval,mval=mval)
    
    # The observations
    targetname = 'kic11721304'
    patternfile = '/lhome/timothyv/gamma_Doradus/Data/Found_Series_newformat/Kepler11721304_spacings.dat'
    
    obsstar = obspuls(targetname,patternfile)
    
    # Parallellised calculations
    nthreads = 10
    ppool = Pool(nthreads)
    
    reg_chi2 = ppool.map(chi_squared_test,np.arange(len(sample['f_rot'])))
    
    # Plotting the results
    plt.figure(2)
    plt.subplot(211)
    plt.plot(sample['f_rot'],reg_chi2,'k.')
    plt.xlabel(r'$\sf f_{\sf rot}$ [$\sf d^{-1}$]')
    plt.ylabel(r'$\chi^2$')
    plt.subplot(212)
    plt.plot(sample['Pi0'],reg_chi2,'k.')
    plt.xlabel(r'$\sf \Pi_{\sf 0}$ [s]')
    plt.ylabel(r'$\chi^2$')
    
    plt.show()
    
    
    
