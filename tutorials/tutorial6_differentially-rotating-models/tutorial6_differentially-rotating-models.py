#!/usr/bin/env python3
#
# File: tutorial6_differentially-rotating-models.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Calculate asymptotic gravito-inertial modes for a radially
#              differentially rotating star



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



def taylor_coeff(star, frot_profile):
    """
        Routine to calculate the coefficients of a rewritten Taylor expansion 
        for a given stellar structure model and radially differential rotation 
        profile.
        
        Parameters:
            star:         stellar_model object
                          the (non-rotating) stellar model
            frot_profile: array, dtype=astropy quantity (type: frequency)
                          the rotation frequency profile of the star
        
        Returns:
            rs:           astropy quantity
                          estimated radial coordinate at which the g-mode  
                          pulsations are most sensitive.
            frot_s:       astropy quantity
                          the stellar rotation frequency at the radial 
                          coordinate rs
            dfrot_s:      astropy quantity
                          the derivative of the stellar rotation profile with
                          respect to the radius r, at the stellar coordinate rs
            d2frot_s:     astropy quantity
                          the second derivative of the stellar rotation profile 
                          with respect to the radius r, at the stellar 
                          coordinate rs
            c1:           float
                          the first coefficient of the expansion
            c2:           float
                          the second coefficient of the expansion
    """
    
    def sloc(star, qtty):
        """
            calculate the average value of the astrophysical quantity qtty,
            weighted using the function N/r (where N and r are the Brunt-Vaisala
            frequency and stellar radius), or in other words, weighted according
            to the sensitivity of the g-modes in the star (ignoring the Coriolis
            force).
        """
        
        qtty_sloc = np.trapz(qtty * star.N/star.radius,x=star.radius) \
                                    / np.trapz(star.N/star.radius,x=star.radius)
        return qtty_sloc
    
    
    rs = sloc(star, star.radius)
    r_min_rs = sloc(star, star.radius - rs)
    r_min_rs_sq = sloc(star, (star.radius - rs)**2.)
    
    d_frot_profile = np.gradient(frot_profile)/np.gradient(star.radius)
    d2_frot_profile = np.gradient(d_frot_profile)/np.gradient(star.radius)
    
    frot_s = np.interp(rs, star.radius, frot_profile)
    dfrot_s = np.interp(rs, star.radius, d_frot_profile)
    d2frot_s = np.interp(rs, star.radius, d2_frot_profile)
    
    ## In the interest of clarity, the full expression for the coefficient c1
    ## is provided (and commented out) below. In the current implementation,
    ## the variable r_min_rs should always be 0, and a simplified expression
    ## for c1 is actually used.
    
    # c1 = ( dfrot_s / frot_s * r_min_rs ) \
    #       + ( 0.5 * d2frot_s / frot_s * r_min_rs_sq)
    
    c1 = 0.5 * d2frot_s / frot_s * r_min_rs_sq
    c2 = 0.5 * (dfrot_s / frot_s)**2. * r_min_rs_sq
    
    c1 = c1.value
    c2 = c2.value
    
    return rs, frot_s, dfrot_s, d2frot_s, c1, c2
    




if __name__ == "__main__":
    
    
    ###
    ### Some required ( & optional) variables
    ###
    
    gyre_dir, nthreads = read_config_file()
    
    
    # Geometric mode identifications (n,k,m) for the g-modes, where n is the 
    # radial order, k is the meridional degree), and m is the azimuthal order. 
    k1 = 0      # meridional degree of the first pattern
    m1 = 1      # azimuthal order of the first pattern
    k2 = -2     # meridional degree of the second pattern
    m2 = -1     # azimuthal order of the second pattern
    
    nmin = 1   # min. radial order that is considered. (optional; default = 1)
    nmax = 150 # max. radial order that is considered. (optional; default = 150)
    
    
    ### stellar input parameters
    frot_s = 2.0 / u.d          # rotation frequency (with astropy unit)
    df = np.linspace(0.1,0.2,3) # relative differential rotation rates
    alpha_g = 0.50              # phase term dependent on mode cavity boundaries 
                                # (optional; default = 0.5)
    pattern_unit = 'days'       # output units 
                                # (optional; default = 'cycle_per_day')
    dp_unit = u.s               # different unit for the period spacings in the 
                                # tutorial figure (to help make it pretty)
    
    # The MESA input model for GYRE that is used in this tutorial
    mesaprofile = f'{os.path.dirname(__file__)}/M150Z0140_fov150_Xc50.data.GYRE'
    
    # Reading in the necessary quantities of the MESA model
    radius, brunt_N2 = read_mesaprofile(mesaprofile)
    
    # Initialising the stellar model
    star = stellar_model(radius,brunt_N2)
    Pi0 = star.Pi0()
    
    # Initialising the gravity_modes class objects
    gmode_k0m1 = gm(gyre_dir, kval=k1, mval=m1, nmin=nmin, nmax=nmax)
    rmode_k2m1 = gm(gyre_dir, kval=k2, mval=m2, nmin=nmin, nmax=nmax)
    
    # Calculating the patterns for uniform rotation as benchmarks
    gmode_unif = gmode_k0m1.uniform_pattern(frot_s, Pi0, unit='days')
    rmode_unif = rmode_k2m1.uniform_pattern(frot_s, Pi0, unit='days')
    
    
    
    
    
    ###
    ### Tutorial 6A: calculating asymptotic g-mode patterns for multiple
    ###              differential rotation profiles
    ###
    
    frot_profiles = []
    gmode_series = []
    rmode_series = []

    # Looping over the multiple differential rotation rates
    for i_df in df:

        # Calculating a radially differential rotation profile
        frot_prof, rs = star.brunt_rot_profile(frot_s, i_df)
        frot_profiles.append(frot_prof)
    
        # Calculating the patterns for the assumed differential rotation rates
        gmode_diff = gmode_k0m1.differential_pattern_integral(star, frot_prof, \
                                                                    unit='days')
        rmode_diff = rmode_k2m1.differential_pattern_integral(star, frot_prof, \
                                                                    unit='days')
        gmode_series.append(gmode_diff)
        rmode_series.append(rmode_diff)
    
    
    
    # Plotting the results (spacing as a function of period)
    fig6a = plt.figure('Tutorial 6A', figsize=(6.4,10))
    
    # the rotation profiles
    plt.subplot(311) 
    plt.plot((star.radius[0].value, star.radius[-1].value), \
          (frot_s.value, frot_s.value), 'k-', c='0.8', label=r'$\delta$f = 0.0')
    
    for ii, i_df, frot_prof in zip(np.arange(1.,1.+len(df)), df, frot_profiles):
        clr = f'{np.round(0.8 * (1. - ii/len(df)),3)}'
        plt.plot(star.radius, frot_prof, 'k-', c=clr, \
                                       label=fr'$\delta$f = {np.round(i_df,2)}')
    plt.xlim(star.radius[0].value, star.radius[-1].value)
    
    plt.legend(loc = 'upper right')
    plt.xlabel(fr"radius ({star.radius.unit.to_string('latex')})")
    plt.ylabel(fr"rotation frequency ({frot_prof.unit.to_string('latex')})")
    
    # the g-mode patterns
    plt.subplot(312) 
    plt.plot(gmode_unif[:-1], np.diff(gmode_unif.to(dp_unit)), 'k-', c='0.8', \
                                                                     marker='.')
    
    for ii, gmode_diff in zip(np.arange(1.,1.+len(df)), gmode_series):
        clr = f'{np.round(0.8 * (1. - ii/len(df)),3)}'
        plt.plot(gmode_diff[:-1], np.diff(gmode_diff.to(dp_unit)), 'k-', c=clr,\
                                                                     marker='.')
    plt.title('(k,m) = (0,1)')
    plt.xlabel(fr"Period P ({gmode_unif.unit.to_string('latex')})")
    plt.ylabel(fr"Period spacing $\Delta$P ({dp_unit.to_string('latex')})")
    
    # the r-mode patterns
    plt.subplot(313)
    plt.plot(np.abs(rmode_unif[:-1]), np.diff(rmode_unif.to(dp_unit)), 'k-', \
                                                            c='0.8', marker='.')
    for ii, rmode_diff in zip(np.arange(1.,1.+len(df)), rmode_series):
        clr = f'{np.round(0.8 * (1. - ii/len(df)),3)}'
        plt.plot(np.abs(rmode_diff[:-1]), np.diff(rmode_diff.to(dp_unit)), \
                                                        'k-', c=clr, marker='.')
    plt.title('(k,m) = (-2,-1)')
    plt.xlabel(fr"Period P ({gmode_unif.unit.to_string('latex')})")
    plt.ylabel(fr"Period spacing $\Delta$P ({dp_unit.to_string('latex')})")
    
    plt.suptitle(fr"Tutorial 6A: g-mode patterns with differential rotation")
    
    plt.tight_layout()
    
    
    
    
    
    
    ###
    ### Tutorial 6B: comparing the "proper" asymptotic g-mode patterns for
    ###              a differential rotation profile with the Taylor expansion
    ###
    
    frot_prof = frot_profiles[-1]
    df_s = df[-1]
    
    gmode_diff = gmode_series[-1]
    rmode_diff = rmode_series[-1]
    
    # Calculating the (local) Taylor expansions of the period-spacing  
    # patterns. For the purpose of this tutorial we simply calculate the  
    # theoretical coefficients for the Taylor expansion for the 
    # precalculated rotation profile.
    rs, frot_s, dfrot_s, d2frot_s, c1, c2 = taylor_coeff(star, frot_prof)
    
    gmode_exp = gmode_k0m1.differential_pattern_taylor(frot_s, Pi0, \
                                                    coeffs=[c1,c2], unit='days')
    rmode_exp = rmode_k2m1.differential_pattern_taylor(frot_s, Pi0, \
                                                    coeffs=[c1,c2], unit='days')
        
    
    
    
    # Plotting the results (spacing as a function of period)
    fig6b = plt.figure('Tutorial 6B', figsize=(6.4,10))
    
    # the rotation profiles
    plt.subplot(311)
    plt.plot((star.radius[0].value, star.radius[-1].value), \
          (frot_s.value, frot_s.value), 'k-', c='0.8', label=r'$\delta$f = 0.0')
    plt.plot(star.radius, frot_prof, 'k-', c='0.4', \
                                       label=fr'$\delta$f = {np.round(df_s,1)}')
    
    frot_expansion = frot_s + dfrot_s*(star.radius-rs) \
                                             + 0.5*d2frot_s*(star.radius-rs)**2.
    plt.plot(star.radius, frot_expansion, 'k--', label='Taylor expansion')
    plt.xlim(star.radius[0].value, star.radius[-1].value)
    
    plt.legend(loc = 'upper right')
    plt.xlabel(fr"radius ({star.radius.unit.to_string('latex')})")
    plt.ylabel(fr"rotation frequency ({frot_prof.unit.to_string('latex')})")
        
    # the g-mode patterns
    plt.subplot(312) 
    plt.plot(gmode_unif[:-1], np.diff(gmode_unif.to(dp_unit)), 'k-', c='0.8', \
                                                                     marker='.')
    
    plt.plot(gmode_diff[:-1], np.diff(gmode_diff.to(dp_unit)), 'k-', c='0.4',\
                                                                     marker='.')
    plt.plot(gmode_exp[:-1], np.diff(gmode_exp.to(dp_unit)), 'k-', marker='.')
    
    plt.title('(k,m) = (0,1)')
    plt.xlabel(fr"Period P ({gmode_unif.unit.to_string('latex')})")
    plt.ylabel(fr"Period spacing $\Delta$P ({dp_unit.to_string('latex')})")
        
    # the r-mode patterns
    plt.subplot(313) 
    plt.plot(np.abs(rmode_unif[:-1]), np.diff(rmode_unif.to(dp_unit)), 'k-', \
                                                            c='0.8', marker='.')
    
    plt.plot(np.abs(rmode_diff[:-1]), np.diff(rmode_diff.to(dp_unit)), 'k-', \
                                                            c='0.4', marker='.')
    plt.plot(np.abs(rmode_exp[:-1]), np.diff(rmode_exp.to(dp_unit)), 'k-', \
                                                                     marker='.')
    
    plt.title('(k,m) = (-2,-1)')
    plt.xlabel(fr"Period P ({gmode_unif.unit.to_string('latex')})")
    plt.ylabel(fr"Period spacing $\Delta$P ({dp_unit.to_string('latex')})")
    
    plt.suptitle(fr"Tutorial 6B: Taylor expansion of g-mode patterns")
    
    plt.tight_layout()
    
    
    plt.show()
