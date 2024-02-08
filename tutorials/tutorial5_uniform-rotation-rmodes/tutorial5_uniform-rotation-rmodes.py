#!/usr/bin/env python3
#
# File: tutorial5_uniform-rotation-rmodes.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Calculate an r-mode model pattern (with (k,m) = (-2,-1)) for a
#              uniformly rotating star, given an input value for the rotation
#              frequency frot and calculating the buoyancy radius Pi0 from a 
#              MESA input model for GYRE.

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
        format), and provide the stellar radius and N^2-profile with astropy
        units.

        Parameters:
            mesaprofile: string
                         The MESA profile that we will read in.
        
        Returns:
            radius:      astropy quantity array (quantity = 'length')
                         the radial coordinates of the stellar structure profile
            brunt_N2:    astropy quantity array (unit = 'rad^2/s^2').
                         the squared Brunt-Vaisala frequency profile of the 
                         stellar model, with values given at the radial 
                         coordinates given in 'radius'.
    """

    profile = np.loadtxt(mesaprofile, skiprows=1)
    radius = profile[:,1] * u.cm
    brunt_N2 = profile[:,8] * (u.rad / u.s)**2.

    return radius, brunt_N2





if __name__ == "__main__":
    
    
    ###
    ### Some required ( & optional) variables
    ###
    
    gyre_dir, nthreads = read_config_file()
    
    
    # Geometric mode identification (n,k,m) for the r-modes, where n is the 
    # radial order, k is the meridional degree, and m is the azimuthal order. 
    # Because r-modes only exist in rotating stars, k < 0, and because they are
    # retrograde, m < 0 (following the convention in AMiGO).
    k = -2     # meridional degree
    m = -1     # azimuthal order
    nmin = 1   # min. radial order that is considered. (optional; default = 1)
    nmax = 150 # max. radial order that is considered. (optional; default = 150)
    
    
    ### stellar input parameters
    frot = 1.35 / u.day             # The rotation frequency (with astropy unit)
    alpha_g = 0.75                  # phase term dependent on the mode cavity 
                                    # boundaries. (optional; default = 0.5)
    pattern_unit = 'days'  # output units. (optional; default = 'cycle_per_day')
    dp_unit = u.s          # different unit for the period spacings in the 
                           # tutorial figure (to help make it pretty)
        
    # The MESA input model for GYRE that is used in this tutorial
    mesaprofile = f'{os.path.dirname(__file__)}/M150Z0140_fov150_Xc50.data.GYRE'
    
    # Reading in the necessary quantities of the MESA model
    radius, brunt_N2 = read_mesaprofile(mesaprofile)
    
    # Initialising the stellar model
    star = stellar_model(radius, brunt_N2)
    
    # Calculating Pi0
    Pi0 = star.Pi0(unit='seconds')
    
    # Initialising the gravity_modes class object
    rmode_k2m1 = gm(gyre_dir, kval=k, mval=m, nmin=nmin, nmax=nmax)
    
    # Calculating the r-mode pattern
    rmode_series = rmode_k2m1.uniform_pattern(frot, Pi0, alpha_g=alpha_g)


    # Plotting the results:
    # R modes are always retrograde, and have frequencies in the co-rotating
    # frame that are smaller than the stellar rotation frequency. As a result,
    # to an observer in the inertial reference frame, they look like they are
    # prograde modes, with frequencies between (|m|-1)*f_rot and |m|*f_rot.
    # In AMiGO, we follow the convention that m < 0 for retrograde modes, and
    # that pulsation frequencies in the corotating frame are positive. As a 
    # result, calculated r-mode frequencies in the inertial reference frame are
    # negative. To match these asymptotic patterns to observed r-mode patterns,
    # we therefore have to take their absolute values (since observed 
    # frequencies are always positive.
    
    fig5 = plt.figure('Tutorial 5')
    
    ax_nfreq = fig5.add_subplot(211)
    plt.plot(rmode_k2m1.nvals, rmode_series, 'k-', marker='.')
    plt.xlabel(fr"radial order $n$")
    plt.ylabel(fr"frequency $f$ " +\
               fr"({rmode_series.unit.to_string('latex')})")
    
    ax_pdp = fig5.add_subplot(212)
    rmode_periods = np.sort(np.abs(1./rmode_series))
    plt.plot(rmode_periods[:-1], np.diff(rmode_periods.to(dp_unit)), 'k-', \
                                                                   marker = '.')
    plt.xlabel(fr"period |P| ({rmode_periods.unit.to_string('latex')})")
    plt.ylabel(fr"period spacing $\Delta$|P| ({dp_unit.to_string('latex')})")
    
    plt.suptitle(fr"(k,m)=({k},{m}) pattern with " +
              fr"$\Pi_0$ = {int(Pi0.value)}{Pi0.unit.to_string('latex')} "+
              fr"and $f_r$ = {frot.value}{frot.unit.to_string('latex')}")
    
    plt.tight_layout()
    
    plt.show()
