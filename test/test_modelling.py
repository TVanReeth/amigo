import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u

from multiprocessing import Pool

import amigo.forward_modelling as fm
from amigo.gmode_series import asymptotic
from amigo.observations import pulsations as obspuls

"""
    A collection of various testing routines / demos of the subroutines and classes in AMIGO.
"""



def chi_squared_test(ind):
    """A simple chi^2 calculation, used in the demo in __main__."""
    
    pattern = k0m1.uniform_pattern(sample['f_rot'][ind]/u.day,sample['Pi0'][ind]*u.s)
    chisq = np.sum((np.diff(obsstar.period) - np.interp(obsstar.period[:-1],pattern[:-1],np.diff(pattern))*u.day)**2./(obsstar.e_period[1:]**2. + obsstar.e_period[:-1]**2.))
    
    return chisq
        

if(__name__ == '__main__'):
    
    
    ### A simple test/demo on how to combine the routines to do model fitting
    ### At the moment, this still uses a simple chi^2; AIM: switch to Aerts et al. 2018.
    
    # Again, the sampling
    parameters = ['Pi0','f_rot']
    minbound = np.array([2300., 0.]) # minimum values of Pi0 and f_rot
    maxbound = np.array([5600.,2.5]) # maximum values of Pi0 and f_rot
    regular_step = np.array([10.,0.005]) 
    method = 'regular'
    
    sample = fm.sampling(parameters,minbound,maxbound,method=method,regular_step=regular_step)
    
    # asymptotic object initialisation
    gyre_dir = '/lhome/timothyv/Bin/mesa/mesa-12778/gyre/gyre/'
    kval = 0
    mval = 1
    
    k0m1 = asymptotic(gyre_dir,kval=kval,mval=mval)
    
    # The observations
    targetname = 'kic11721304'
    patternfile = './test_data/Kepler11721304_spacings.dat'
    
    obsstar = obspuls(targetname,patternfile)
    
    # Parallellised calculations
    nthreads = 3
    ppool = Pool(nthreads)
    
    reg_chi2 = ppool.map(chi_squared_test,np.arange(len(sample['f_rot'])))
    
    # Plotting the results
    plt.figure(2)
    plt.subplot(211)
    plt.semilogy(sample['f_rot'],reg_chi2,'k.')
    plt.xlabel(r'$\sf f_{\sf rot}$ [$\sf d^{-1}$]')
    plt.ylabel(r'$\chi^2$')
    plt.subplot(212)
    plt.semilogy(sample['Pi0'],reg_chi2,'k.')
    plt.xlabel(r'$\sf \Pi_{\sf 0}$ [s]')
    plt.ylabel(r'$\chi^2$')
    
    plt.show()
    
    
    