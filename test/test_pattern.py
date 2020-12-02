# 
#  A basic demonstration of some of the main routines of the AMIGO python package.
#  Specifically, how to calculate period-spacing patterns using the TAR and assuming uniform rotation.
#   
#  author: Timothy Van Reeth (KU Leuven)
#          timothy.vanreeth@kuleuven.be
#

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u

from amigo.gmode_series import asymptotic
from amigo.stellar_model import stellar_model

def read_mesaprofile(mesaprofile):
    """
        A wrapper routine to read in a precalculated MESA profile, and provide the stellar radius and N^2-profile
        with astropy units.

        Parameters:
            mesaprofile: string
                         The MESA profile that we will read in.
        
        Returns:
            radius:  array, dtype = astropy quantity 'length'.
            brunt_N2:  array, dtype = astropy quantity with unit 'rad^2/s^2'.
    """

    profile = np.genfromtxt(mesaprofile, names=True, skip_header=5)
    radius = profile['radius_cm'] * u.cm
    brunt_N2 = profile['brunt_N2'] * (u.rad / u.s)**2.

    return radius, brunt_N2



if __name__ == "__main__":
    ### Some required ( & optional) quantities

    # Input for the asymptotic class object
    gyre_dir = '/lhome/timothyv/Bin/mesa/mesa-12778/gyre/gyre/'  # GYRE installation directory, v5.x or v6.x is required.
    kval = 0    # For normal g-modes, k = |l| - m; optional. Default = 0.
    mval = 1    # The azimuthal order; optional. Default = 1.
    nmin = 1    # The minimum radial order for which we want to compute patterns; optional. Default = 1.
    nmax = 150  # The maximum radial order for which we want to compute patterns; optional. Default = 150.
    
    ### Demo...
    # Initialising the asymptotic class object
    asym_k0m1 = asymptotic(gyre_dir,kval=kval, mval=mval, nmin=nmin, nmax=nmax)

    

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




    ### Calculating a second pattern assuming uniform rotation, with Pi0 computed for a given MESA model
    # Additional input
    frot = 1.35 / u.day               # The rotation frequency (with astropy unit!)
    alpha_g = 0.75                    # The phase term dependent on the boundaries of the mode cavity; optional. Default = 0.5
    pattern_unit2 = 'cycle_per_day'   # The unit in which we want to have the pulsation periods or frequencies of the pattern. For the 2nd demo...
    mesaprofile = './test_data/M1p5_Z0014_X30_Dmix10_fov15.data'
    

    # Calculating Pi0
    radius, brunt_N2 = read_mesaprofile(mesaprofile)
    star = stellar_model(radius, brunt_N2)
    Pi0_mesa = star.buoyancy_radius(unit='seconds')
    

    # Calculating the pattern using the info from the MESA profile directly
    mesa_pattern = asym_k0m1.uniform_pattern(frot,Pi0_mesa,alpha_g=alpha_g,unit=pattern_unit2)
    
    
    # Plotting the result (frequency as a function of radial order)
    plt.figure(2)
    plt.plot(np.linspace(nmin, nmax, nmax-nmin+1), mesa_pattern, 'k-', marker='.')
    plt.xlabel(fr"radial order $n$")
    plt.ylabel(fr"frequency $f$ [{mesa_pattern.unit.to_string('latex')}]")
    plt.title(fr"g-mode pattern (for a MESA profile) with $\Pi_0$ = {int(Pi0_mesa.value)}{Pi0_mesa.unit.to_string('latex')} and $f_r$ = {frot.value}{frot.unit.to_string('latex')}")
    
    
    plt.show()
