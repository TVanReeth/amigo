#!/usr/bin/env python3
#
# File: tutorial4_uniform-rotation-gmodes.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Calculate g-mode model patterns (with (k,m) = (0,1)) for a
#              uniformly rotating star, given an input value for the 
#              stellar buoyancy radius Pi0 and assuming different rotation 
#              frequencies frot.



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
    




if __name__ == "__main__":
    
    
    ###
    ### Some required ( & optional) variables
    ###
    
    gyre_dir, nthreads = read_config_file()
    
    
    # Geometric mode identification (n,k,m) for the g-modes, where n is the 
    # radial order, k is the meridional degree, which for gravito-inertial
    # modes is k = |l| - m (where l is the spherical degree), and m is the
    # azimuthal order. In AMiGO, prograde modes are indicated with m > 0, and
    # retrograde modes with m < 0. 
    k = 0      # meridional degree
    m = 1      # azimuthal order
    nmin = 1   # min. radial order that is considered. (optional; default = 1)
    nmax = 150 # max. radial order that is considered. (optional; default = 150)
    
    
    ### stellar input parameters
    frot = np.linspace(0.0, 1., 6) / u.d  # rotation frequencies (with astropy 
                                          # unit)
    Pi0 = 4200. * u.s                     # buoyancy radius (with astropy unit)
    alpha_g = 0.50                        # phase term dependent on mode cavity 
                                          # boundaries (optional; default = 0.5)
    pattern_unit = 'days'                 # output units (optional; 
                                          # default = 'cycle_per_day')
    dp_unit = u.s                         # different unit for the period 
                                          # spacings in the tutorial figure (to 
                                          # help make it pretty)
    
    
    # Initialising the gravity_modes class object
    gmode_k0m1 = gm(gyre_dir, kval=k, mval=m, nmin=nmin, nmax=nmax)
    
    # Calculating the patterns for each of the assumed uniform rotation rates
    k0m1_series = []
    
    for i_frot in frot:
        i_pattern = gmode_k0m1.uniform_pattern(i_frot, Pi0, alpha_g=alpha_g, \
                                                              unit=pattern_unit)
        k0m1_series.append(i_pattern)


    # Plotting the results (spacing as a function of period)
    fig4 = plt.figure('Tutorial 4')
    
    for i_frot, i_pattern in zip(frot, k0m1_series):
        i_frot_label = fr"$f_r$ = {np.around(i_frot.value,decimals=1)} " + \
                       fr"{i_frot.unit.to_string('latex')}"
        i_clr = f'{0.9 * (1.-i_frot.value/np.amax(frot.value))}'
        plt.plot(i_pattern[:-1], np.diff(i_pattern.to(dp_unit)), 'k-', \
                                        c=i_clr, marker='.', label=i_frot_label)
    
    plt.xlim(0., 1.05*np.amax(k0m1_series[1].value))
    plt.ylim(0., 1.05*np.amax(np.diff(k0m1_series[0].to(dp_unit)).value))
    
    plt.legend(loc = 'upper right')
    plt.xlabel(fr"Period P ({k0m1_series[0].unit.to_string('latex')})")
    plt.ylabel(fr"Period spacing $\Delta$P ({dp_unit.to_string('latex')})")
    plt.title(fr"Tutorial 4: (k,m)=({k},{m}) patterns with " +
              fr"$\Pi_0$ = {int(Pi0.value)}{Pi0.unit.to_string('latex')}")
    
    plt.tight_layout()
    
    plt.show()
