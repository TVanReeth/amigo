# 
#  A basic demonstration of some of the main routines of the AMIGO python package.
#  Specifically, how to calculate period-spacing patterns using the TAR and assuming uniform rotation, and plot them.
#   
#  author: Timothy Van Reeth (KU Leuven)
#          timothy.vanreeth@kuleuven.be
#

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u

from amigo.gmode_series import asymptotic as asymp_gmodes



if __name__ == "__main__":
    ### Some required ( & optional) quantities

    # Input for the asymptotic class object
    gyre_dir = '/STER/timothyv/Bin/gyre-52/'  # GYRE installation directory, v5.x or v6.x is required.
    kval = 0    # For normal g-modes, k = |l| - m; optional. Default = 0.
    mval = 1    # The azimuthal order; optional. Default = 1.
    nmin = 1    # The minimum radial order for which we want to compute patterns; optional. Default = 1.
    nmax = 150  # The maximum radial order for which we want to compute patterns; optional. Default = 150.
    
    ### Demo...
    # Initialising the asymptotic class object
    asym_k0m1 = asymp_gmodes(gyre_dir,kval=kval, mval=mval, nmin=nmin, nmax=nmax)

    ### Calculating a first pattern assuming uniform rotation
    # Additional input
    frot = 0.5 / u.day               # The rotation frequency (with astropy unit!)
    Pi0 = 4200. * u.s                # The asymptotic spacing / buoyancy radius (with astropy unit)
    alpha_g = 0.50                   # The phase term dependent on the boundaries of the mode cavity; optional. Default = 0.5
    pattern_unit1 = 'days'           # The unit in which we want to have the pulsation periods or frequencies of the pattern. Default = 'days'.
    
    spacing_unit = u.s               # Different unit for the period spacings in the figure (to help make it pretty)

    # Calculating the pattern assuming uniform rotation
    demo_pattern = asym_k0m1.uniform_pattern(frot,Pi0,alpha_g=alpha_g,unit=pattern_unit1)

     # Plotting the result (spacing as a function of period)
    plt.figure(1)
    plt.plot(demo_pattern[:-1], np.diff(demo_pattern.to(spacing_unit)), 'k-', marker='.')
    plt.xlabel(fr"Period P [{demo_pattern.unit.to_string('latex')}]")
    plt.ylabel(fr"Period spacing $\Delta$P [{spacing_unit.to_string('latex')}]")
    plt.title(fr"g-mode pulsation pattern with $\Pi_0$ = {int(Pi0.value)}{Pi0.unit.to_string('latex')} and $f_r$ = {frot.value}{frot.unit.to_string('latex')}")
    
    plt.show()
