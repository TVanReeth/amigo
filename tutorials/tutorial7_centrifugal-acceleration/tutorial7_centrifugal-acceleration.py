#!/usr/bin/env python3
#
# File: tutorial7_centrifugal-acceleration.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Calculate asymptotic gravito-inertial modes for a centrifugally
#              deformed rotating star



import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u

from amigo.asymptotic_theory import gravity_modes as gm
from amigo.stellar_model import stellar_model

mpl.rc('font', size=14)


def read_config_file():
    """
        Read in the configuration file of AMiGO.
        
        Parameters:
            NA
        
        Returns:
            gyre_dir:        string
                             path to the GYRE installation directory
            nthreads:        integer
                             number of threads to be used in the parallellised 
                             parts of the code
    """
    
    config_file = f'{os.path.dirname(__file__)}/../../defaults/config.dat'
    
    ### Reading in the main config file
    file = open(config_file,'r')
    lines = file.readlines()
    file.close()
    
    gyre_dir = ''
    nthreads = 1
    
    for line in lines:
        if(line.split('=')[0].strip() == 'gyre_dir'):
            gyre_dir = line.split('=')[1].split('#')[0].strip()
        elif(line.split('=')[0].strip() == 'nthreads'):
            nthreads = int(line.split('=')[1].split('#')[0].strip())
    
    return gyre_dir, nthreads
    


def read_mesaprofile(mesaprofile):
    """
        A basic routine to read in a precalculated MESA profile (in GYRE  
        format), and provide the required quantities (with astropy units) to
        calculate the centrifugal deformation and its effect on g-mode 
        pulsations.

        Parameters:
            mesaprofile: string
                         The MESA profile that we will read in.
        
        Returns:
            radius:      astropy quantity array (quantity = 'length')
                         the radial coordinates of the stellar structure profile
            N2:          astropy quantity array (unit = 'rad^2/s^2').
                         the squared Brunt-Vaisala frequency profile of the 
                         stellar model, with values given at the radial 
                         coordinates given in 'radius'.
            mass:        astropy quantity array (unit = 'grams')
                         the mass coordinates of the stellar structure profile
            pressure:    astropy quantity array (unit = 'Ba')
                         the pressure profile of the given stellar structure 
                         model
            density:     astropy quantity array (unit = 'g / cm^3')
                         the density profile of the given stellar structure 
                         model
            Gamma1:      numpy array (dtype = float)
                         the adiabatic temperature gradient profile of the given
                         stellar structure model
    """

    profile = np.loadtxt(mesaprofile, skiprows=1)
    radius = profile[:,1] * u.cm
    N2 = profile[:,8] * (u.rad / u.s)**2.
    
    mass = profile[:,2] * u.g
    pressure = profile[:,4] * u.Ba
    density = profile[:,6] * u.g / u.cm**3
    Gamma1 = profile[:,9]

    return radius, N2, mass, pressure, density, Gamma1
    




if __name__ == "__main__":
    
    
    ###
    ### Some required ( & optional) variables
    ###
    
    gyre_dir, nthreads = read_config_file()
    
    ode_method = 'BDF' # the numerical method that we use to solve the ODE
    nthreads = 8       # the number of threads used in the parallellisation
    
    
    # Geometric mode identifications (n,k,m) for the g-modes, where n is the 
    # radial order, k is the meridional degree), and m is the azimuthal order. 
    k1 = 0      # meridional degree of the first pattern
    m1 = 1      # azimuthal order of the first pattern
    k2 = -2     # meridional degree of the second pattern
    m2 = -1     # azimuthal order of the second pattern
    
    nmin = 1   # min. radial order that is considered. (optional; default = 1)
    nmax = 150 # max. radial order that is considered. (optional; default = 150)
    
    
    ### stellar input parameters
    omrot_frac = 0.40           # rotation frequency (as a fraction of the 
                                # critical Roche rotation rate)
    alpha_g = 0.50              # phase term dependent on mode cavity boundaries 
                                # (optional; default = 0.5)
    pattern_unit = 'days'       # output units 
                                # (optional; default = 'cycle_per_day')
    dp_unit = u.s               # different unit for the period spacings in the 
                                # tutorial figure (to help make it pretty)
    
    # The MESA input model for GYRE that is used in this tutorial
    mesaprofile = f'{os.path.dirname(__file__)}/M150Z0140_fov150_Xc50.data.GYRE'
    
    # Reading in the necessary quantities of the MESA model
    radius, N2, mass, pressure, density, Gamma1 = read_mesaprofile(mesaprofile)
    
    # Initialising the stellar model
    star = stellar_model(radius, N2, mass=mass, pressure=pressure, \
                                                 density=density, Gamma1=Gamma1)
    Pi0 = star.Pi0()
    
    # Initialising the gravity_modes class objects
    gmode_k0m1 = gm(gyre_dir, kval=k1, mval=m1, nmin=nmin, nmax=nmax)
    rmode_k2m1 = gm(gyre_dir, kval=k2, mval=m2, nmin=nmin, nmax=nmax)
    
    # calculating the centrifugal deformation
    omrot = omrot_frac * star.omegacrit_roche[-1]
    frot = omrot / (2.*np.pi*u.rad)
    star.calculate_centrifugal_deformation(omrot, nthreads=nthreads)
    
    # Calculating the patterns assuming spherical symmetry as benchmarks
    gmode_unif = gmode_k0m1.uniform_pattern(frot, Pi0, unit='days')
    rmode_unif = rmode_k2m1.uniform_pattern(frot, Pi0, unit='days')
    
    # Calculating the patterns accounting for centrifugal deformation
    gmode_cntr = gmode_k0m1.centrifugal_pattern(frot, star, unit='days')
    rmode_cntr = rmode_k2m1.centrifugal_pattern(frot, star, unit='days')
    
    
    
    # Plotting the results (spacing as a function of period)
    fig7 = plt.figure('Tutorial 7', figsize=(6.4,10))
    
    # the deformed N^2-profiles, at different values of mu = cos(theta)
    plt.subplot(311)
    
    mus = np.array([1.00, 0.67, 0.33, 0.00])
    mulbls = ['1.00','0.67','0.33','0.00']
    slct = np.r_[star.N2 > 0.]
    plt.plot(star.radius[slct], np.log10(star.N2.value[slct]), c='r', \
                                     label=r'$\Omega_{\sf rot} = 0$', zorder=10)
    
    for imu,mu in enumerate(mus):
        pseudo_rad = star.centrifugal_radius(mu)
        pseudo_N2 = star.centrifugal_N2profile(mu)
        clr = f'{np.round(0.8 * (1. - imu/len(mus)),3)}'
        txtlbl = r'$\Omega_{\sf rot} = {omrot_frac}\Omega_{\sf crit,K}$, ' \
                 + r'$\mu = $'+f'{mulbls[imu]}'
        slct = np.r_[pseudo_N2 > 0.]
        plt.plot(pseudo_rad[slct],np.log10(pseudo_N2.value[slct]), c=clr, \
                                                label=txtlbl, zorder=int(9-imu))
    
    plt.legend(loc = 'upper right')
    plt.xlabel(fr"radius $r(\mu)$ ({star.radius.unit.to_string('latex')})")
    plt.ylabel(r'log $N^2$ ($rad^2$ $s^{-2}$)')
    plt.ylim(-6.5,-3.5)
    plt.xlim(0.,np.amax(pseudo_rad.value))
    
    # the g-mode patterns
    plt.subplot(312) 
    plt.plot(gmode_unif[:-1], np.diff(gmode_unif.to(dp_unit)), 'r-', marker='.')
    plt.plot(gmode_cntr[:-1], np.diff(gmode_cntr.to(dp_unit)), 'k-', marker='.')
    plt.title('(k,m) = (0,1)')
    plt.xlabel(fr"Period P ({gmode_unif.unit.to_string('latex')})")
    plt.ylabel(fr"Period spacing $\Delta$P ({dp_unit.to_string('latex')})")
    
    # the r-mode patterns
    plt.subplot(313)
    plt.plot(np.abs(rmode_unif[:-1]), np.diff(rmode_unif.to(dp_unit)), 'r-', \
                                                                     marker='.')
    plt.plot(np.abs(rmode_cntr[:-1]), np.diff(rmode_cntr.to(dp_unit)), 'k-', \
                                                                     marker='.')
    plt.title('(k,m) = (-2,-1)')
    plt.xlabel(fr"Period P ({gmode_unif.unit.to_string('latex')})")
    plt.ylabel(fr"Period spacing $\Delta$P ({dp_unit.to_string('latex')})")
    
    plt.suptitle(fr"Tutorial 7: g-modes with centrifugal acceleration")
    
    plt.tight_layout()
    
    plt.show()
