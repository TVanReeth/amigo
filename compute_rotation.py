#!/usr/bin/env python3
#
# File: compute_rotation.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Module to fit asymptotic g-mode patterns to observed period
#              spacings and derive the near-core rotation rate and the 
#              buoyancy travel time.
 
import numpy as np
from multiprocessing import Pool
from functools import partial

from scipy.stats import chi2

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

mpl.rc('font',size=14)

import amigo.forward_modelling as fm
from amigo.gmode_series import asymptotic as asymp_gmodes
from amigo.observations import pulsations as obspuls

from lmfit import Parameters, Minimizer
import astropy
import astropy.units as u


def read_inlist():
    """
        Reading in the inlist UserInput_rotation.dat.

        Parameters:
            N/A
        
        Returns:
            gyre_dir:            string
                                 GYRE v5.x or v6.x installation directory of the user
            star:                string
                                 name of the analysed star
            observationfile:     string
                                 absolute path to the file with detected period spacings
            optimisation_method: string
                                 method used to sample and evaluate the parameter space of
                                 the near-core rotation rate and the buoyancy travel time.
                                 Valid options are:
                                     grid: the parameter space is sampled according to the 
                                           values set by the user in the inlist
                                     lmfit: the optimisation is done using the Levenberg-Marquardt
                                            algorithm, as implemented in lmfit in the 'leastsq' routine 
            nthreads:            integer
                                 the number of CPU threads used in the parallellised calculations
            frot_bnd:            numpy array
                                 minimum, maximum and (for the grid-based evaluation) the step of the
                                 evaluated rotation rates (in cycle per day)
            Pi0_bnd:             numpy array
                                 minimum, maximum and (for the grid-based evaluation) the step of the 
                                 evaluated buoyancy travel times (in seconds)
            kvals:               numpy array
                                 meridional degrees of the calculated asymptotic models
            mvals:               numpy array
                                 (corresponding) azimuthal orders of the calculated asymptotic models
    """
    
    gyre_dir = ''

    star = ''
    observationfile = ''
    
    # A few default values: should be okay for most computers
    optimisation_method = 'grid'      # 'grid' or 'lmfit'
    nthreads = 2
    
    frot_bnd = [0., 2.5, 0.0025]       # min, max, step; in cycle_per_day
    Pi0_bnd  = [2300., 5600., 10.]     # min, max, step; in seconds

    kvals = []
    mvals = []
    
    file = open('./UserInput_rotation.dat','r')
    lines = file.readlines()
    file.close()
    
    for line in lines:
        if(line.isspace()):
            continue

        if(line.strip()[0] == '#'):
            continue
    
        (par,val1) = line.strip().split('=')
        val = val1.split('#')[0].strip()

        if(par.strip() == 'gyre_dir'):
            gyre_dir = val.strip()
        if(par.strip() == 'patterns'):
            observationfile = val.strip()
        
        if(par.strip() == 'optimisation_method'):
            optimisation_method = val.strip()
        if(par.strip() == 'nthreads'):
            nthreads = int(val.strip())
        
        if(par.strip() == 'frot_bnd'):
            frot_bnd = np.array(val.strip().split(),dtype=float)
        if(par.strip() == 'Pi0_bnd'):
            Pi0_bnd = np.array(val.strip().split(),dtype=float)
        if(par.strip() == 'kvals'):
            kvals = np.array(val.strip().split(),dtype=int)
        if(par.strip() == 'mvals'):
            mvals = np.array(val.strip().split(),dtype=int)
  
    kvals = np.array(kvals)
    mvals = np.array(mvals)
    
    return gyre_dir, star, observationfile, optimisation_method, nthreads, frot_bnd, Pi0_bnd, kvals, mvals



def asymptotic_fit(obsstar, modes, optimisation_method='grid', Pi0min=2300., Pi0max=5600., Pi0step=10., frotmin=0.001, frotmax=2.5, frotstep=0.001,nthreads=2):
    """
        Calculate the near-core rotation rate and buoyancy travel time for the best-fitting asymptotic models
        to a (set of) observed g-mode period-spacing pattern(s), assuming a provided (set of) mode identification(s).

        Parameters:
            obsstar:              amigo pulsations object
                                  contains the observed period-spacing patterns
            modes:                list of amigo asymptotic objects
                                  each asymptotic object describes the possible theoretical (asymptotic)
                                  period-spacing patterns for a given mode identification (k,m), and 
                                  corresponds to one of the observed period-sapcing patterns
            optimisation_method:  string; optional
                                  method used to sample and evaluate the parameter space (grid (default) or lmfit):
                                  grid:  the parameter space is sampled on a grid according to the given values
                                  lmfit: the optimisation is done using the Levenberg-Marquardt algorithm, as 
                                         implemented in lmfit in the 'leastsq' routine
            Pi0min:               float; optional
                                  the minimum considered value of the buoyancy travel time (in seconds; default value = 2300s)
            Pi0max:               float; optional
                                  the maximum considered value of the buoyancy travel time (in seconds; default value = 5600s)
            Pi0step:              float; optional
                                  the step size of the buoyancy travel time (in seconds) in the grid-based evaluation.
                                  Ignored when optimisation_method = 'lmfit'. (default value = 10s)
            frotmin:              float; optional
                                  the minimum considered value of the near-core rotation rate (in cycles per day; default value = 0 c/d)
            frotmax:              float; optional
                                  the maximum considered value of the near-core rotation rate (in cycles per day; default value = 2.5 c/d)
            frotstep:             float; optional
                                  the step size of the near-core rotation rate (in cycles per day) in the grid-based evaluation.
                                  Ignored when optimisation_method = 'lmfit'. (default value = 0.001 c/d)
            nthreads:             integer; optional
                                  the number of CPU threads used in the calculations (default value = 2).
        
        Returns:
            fin_frot:             astropy quantity
                                  near-core rotation rate of the best-fitting models
            fin_e_frot:           astropy quantity
                                  1-sigma error margin on the near-core rotation rate of the best-fitting models
            fin_Pi0:              astropy quantity
                                  buoyancy travel time of the best-fitting models
            fin_e_Pi0:            astropy quantity
                                  1-sigma error margin on the buoyancy travel time of the best-fitting models
            sample:               (structured) numpy array
                                  the evaluated samples in the parameter space (in the grid-based evaluation)
            chi2_vals:            numpy array
                                  chi2 values for the samples in the parameter space (in the grid-based evaluation)
            degree_freedom:       integer
                                  number of degrees of freedom of the fitted period-spacing patterns
    """
    
    degree_freedom = np.nansum(np.r_[np.diff(obsstar.seqid()) == 1]) - 2
        
    # fitting the models
    if(optimisation_method == 'grid'):
        # Again, the sampling
        parameters = ['Pi0','f_rot']
        minbound = np.array([Pi0min, frotmin]) # minimum values of Pi0 and f_rot
        maxbound = np.array([Pi0max,frotmax]) # maximum values of Pi0 and f_rot
        regular_step = np.array([Pi0step,frotstep]) 
        method = 'regular'
    
        sample = fm.sampling(parameters,minbound,maxbound,method=method,regular_step=regular_step)
        
        # Parallellised calculations
        ppool = Pool(nthreads)
        chi2_vals = ppool.starmap(partial(residual_sum_squares, obsstar, modes),zip(sample['f_rot'],sample['Pi0']))
        chi2_vals = np.array(chi2_vals)
        invalid = np.r_[chi2_vals > 10.**59.]
        valid = ~invalid
        if(invalid.any() & valid.any()):
            chi2_vals[invalid] = np.amax(chi2_vals[~invalid])
        chi2_vals /= degree_freedom
        
        unique_frot = np.unique(sample['f_rot'])
        unique_Pi0 = np.unique(sample['Pi0'])
        
        chi2_frot = np.array([np.nanmin(chi2_vals[np.r_[sample['f_rot'] == ifrot]]) for ifrot in unique_frot])
        chi2_Pi0 = np.array([np.nanmin(chi2_vals[np.r_[sample['Pi0'] == iPi0]]) for iPi0 in unique_Pi0])
        sel_chi2_frot = np.r_[chi2_frot <= np.amin(chi2_frot)*chi2.ppf(0.99,degree_freedom)/degree_freedom]
        sel_chi2_Pi0 = np.r_[chi2_Pi0 <= np.amin(chi2_Pi0)*chi2.ppf(0.99,degree_freedom)/degree_freedom]
        par_frot = np.polyfit(unique_frot[sel_chi2_frot], chi2_frot[sel_chi2_frot], deg=2)
        par_Pi0 = np.polyfit(unique_Pi0[sel_chi2_Pi0], chi2_Pi0[sel_chi2_Pi0], deg=2)

        fin_frot = -0.5 * par_frot[1] / par_frot[0] / u.d
        fin_Pi0 = -0.5 * par_Pi0[1] / par_Pi0[0] * u.s
        
        try:
            highres_frot = np.linspace(frotmin, frotmax, len(unique_frot)*100)
            cutoff_chi2_frot = (par_frot[0]*fin_frot.value**2. + par_frot[1]*fin_frot.value + par_frot[2])*chi2.ppf(0.6827,degree_freedom)/degree_freedom    # calculating 1-sigma uncertainties
            fin_e_frot = np.amax(np.abs(highres_frot[np.r_[np.polyval(par_frot, highres_frot) <= cutoff_chi2_frot]] - fin_frot.value)) / u.d
    
            highres_Pi0 = np.linspace(Pi0min, Pi0max, len(unique_Pi0)*100)
            cutoff_chi2_Pi0 = (par_Pi0[0]*fin_Pi0.value**2. + par_Pi0[1]*fin_Pi0.value + par_Pi0[2])*chi2.ppf(0.6827,degree_freedom)/degree_freedom    # calculating 1-sigma uncertainties
            fin_e_Pi0 = np.amax(np.abs(highres_Pi0[np.r_[np.polyval(par_Pi0, highres_Pi0) <= cutoff_chi2_Pi0]] - fin_Pi0.value)) * u.s
        
        except:
            fin_e_frot = -1. / u.d
            fin_e_Pi0 = -1. * u.s

        return fin_frot, fin_e_frot, fin_Pi0, fin_e_Pi0, sample, chi2_vals, degree_freedom
        

    elif(optimisation_method == 'lmfit'):
        param = Parameters()
        param.add(name='f_rot',value=0.5*(frotmin+frotmax), min=frotmin, max=frotmax,vary=True)
        param.add(name='Pi0',value=0.5*(Pi0min+Pi0max), min=Pi0min, max=Pi0max,vary=True)
        param.add(name='alpha_g',value=0.5, min=0., max=1.,vary=True)
        
        mini = Minimizer(lmfit_residual_patterns, param, fcn_args=(obsstar, modes),nan_policy='omit')
        out = mini.minimize(method='leastsq')
        
        fin_frot = out.params['f_rot'].value / u.d
        fin_e_frot = out.params['f_rot'].stderr / u.d
        fin_Pi0 = out.params['Pi0'].value * u.s
        fin_e_Pi0 = out.params['Pi0'].stderr * u.s
    
        return fin_frot, fin_e_frot, fin_Pi0, fin_e_Pi0
        


def interp_withunits(newx, oldx, oldy):
    """
        An auxiliary function to help with the interpolation of unsorted astropy quantities 
        (based on the usual application of np.interp)

        Parameters:
            newx:   numpy array of astropy quantity
                    unsorted array of values onto which we want to interpolate the function
            oldx:   numpy array of astropy quantity
                    unsorted array of values for which we know the values of the function
            oldy:   numpy array of astropy quantity
                    function values for oldx
        
        Returns:
            newy:   numpy array of astropy quantity
                    function values for newx
    """

    if(type(oldx) == type(newx) == astropy.units.quantity.Quantity):
        interp_x = newx.to(oldx.unit)
        indsort_newx = np.argsort(interp_x.value)
        indsort_oldx = np.argsort(oldx.value)
        invsort_newx = np.argsort(indsort_newx)
        
        if(type(oldy) == astropy.units.quantity.Quantity):
            interp_y = np.interp(interp_x.value[indsort_newx], oldx.value[indsort_oldx], oldy.value[indsort_oldx]) * oldy.unit
        else:
            interp_y = np.interp(interp_x.value[indsort_newx], oldx.value[indsort_oldx], oldy[indsort_oldx])
        newy = interp_y[invsort_newx]

    elif((type(oldx) == type(newx)) & (type(oldx) != astropy.units.quantity.Quantity)):
        interp_x = newx
        indsort_newx = np.argsort(interp_x)
        indsort_oldx = np.argsort(oldx)
        invsort_newx = np.argsort(indsort_newx)

        if(type(oldy) == astropy.units.quantity.Quantity):
            interp_y = np.interp(interp_x[indsort_newx], oldx[indsort_oldx], oldy.value[indsort_oldx]) * oldy.unit
        else:
            interp_y = np.interp(interp_x[indsort_newx], oldx[indsort_oldx], oldy[indsort_oldx])
        newy = interp_y[invsort_newx]
    
    return newy



def residual_sum_squares(obsstar, modes,f_rot, Pi0, alpha_g=0.5):
    """
        Calculating the total residual sum of squares between the observed and theoretical
        period-spacing patterns. Routine used in the grid-based evaluation of the parameter space.

        Parameters:
            obsstar:    amigo pulsations object
                        contains the observed period-spacing patterns
            modes:      list of amigo asymptotic objects
                        each asymptotic object describes the possible theoretical (asymptotic)
                        period-spacing patterns for a given mode identification (k,m), and 
                        corresponds to one of the observed period-sapcing patterns
            f_rot:      float
                        near-core rotation rate (in cycles per day) for which we want to calculate 
                        the theoretical patterns
            Pi0:        float
                        buoyancy travel time value (in seconds) for which we want to calculate 
                        the theoretical patterns
            alpha_g:    float; optional
                        phase term dependent on the boundaries of the g-mode cavity
        
        Returns:
            rss:        float
                        the total residual sum of squares between the observed and theoretical
                        period-spacing patterns
    """
    
    res = residual_patterns(obsstar, modes, f_rot/u.d, Pi0*u.s, alpha_g=alpha_g) 
    rss = np.nansum(res**2.)
    return rss


    
def lmfit_residual_patterns(param, obsstar, modes):
    """
        Wrapper of the residual_patterns() routine, used in the lmfit-based fitting of the observed
        period-spacing patterns

        Parameters:
            param:      lmfit Parameters object
                        contains the variable near-core rotation rate and buoyancy travel time
            obsstar:    amigo pulsations object
                        contains the observed period-spacing patterns
            modes:      list of amigo asymptotic objects
                        each asymptotic object describes the possible theoretical (asymptotic)
                        period-spacing patterns for a given mode identification (k,m), and 
                        corresponds to one of the observed period-sapcing patterns
            
        Returns:
            res:        numpy array of astropy quantity
                        residuals of the fit between the observed and theoretical period-spacing patterns
    """

    res = residual_patterns(obsstar, modes, param['f_rot']/u.d, param['Pi0']*u.s, alpha_g=param['alpha_g'])
    return res



def residual_patterns(obsstar, modes, frot, Pi0, alpha_g=0.5):
    """
        Calculate the residuals of the fit between the observed and theoretical period-spacing patterns, for
        a given near-core rotation rate, buoyancy travel time and phase term.

        Parameters:
            obsstar:    amigo pulsations object
                        contains the observed period-spacing patterns
            modes:      list of amigo asymptotic objects
                        each asymptotic object describes the possible theoretical (asymptotic)
                        period-spacing patterns for a given mode identification (k,m), and 
                        corresponds to one of the observed period-sapcing patterns
            frot:       astropy quantity
                        evaluated value of the near-core rotation rate
            Pi0:        astropy quantity
                        evaluated value of the buoyancy travel time
            alpha_g:    float; optional
                        evaluated value of the phase term (dependent on the mode cavity boundaries; default value = 0.5)
            
        Returns:
            res:        numpy array of astropy quantity
                        residuals of the fit between the observed and theoretical period-spacing patterns
    """

    all_res = []
    
    obspers = obsstar.period(split=True)
    e_obspers = obsstar.e_period(split=True)
    seqids = obsstar.seqid(split=True)
    
    for mode, obsper, e_obsper, seqid in zip(modes, obspers, e_obspers, seqids):
            
            pattern = mode.uniform_pattern(frot, Pi0, alpha_g=alpha_g)
            
            # dealing with r-modes...
            if(mode.kval < 0):
                pattern = -pattern[::-1]
            
            # Evaluate if the derived radial orders make sense - if not, abort! Setting the residuals to a ridiculous high value...
            if((np.amin(obsper) < np.amin(pattern)) | (np.amax(obsper) > np.amax(pattern))):
                res = np.ones(obsper.shape) * 10.**30.
            else:
                no_gaps = np.r_[np.diff(seqid) == 1]
                res = res = np.array( np.sqrt( (np.diff(obsper)[no_gaps] - interp_withunits(obsper[:-1][no_gaps],pattern[:-1],np.diff(pattern)))**2./(e_obsper[1:][no_gaps]**2. + e_obsper[:-1][no_gaps]**2.) ), dtype=float)
 
            all_res = all_res + list(res)
        
    all_res = np.array(all_res)
    
    return all_res



def plot_results(obsstar, modes, fin_frot, fin_Pi0, optimisation_method='grid', sample=None, chi2_vals=None, degree_freedom=None):
    """
        Plotting the observed period-spacing patterns (period spacing as a function of pulsation period) with the best-fitting models.
        When the 'grid' optimisation method is used, the chi^2 values of the evaluated grid are also shown with 1-, 2-, and 3-sigma contours.

        Parameters:
            obsstar:              amigo pulsations object
                                  contains the observed period-spacing patterns
            modes:                list of amigo asymptotic objects
                                  each asymptotic object describes the possible theoretical (asymptotic)
                                  period-spacing patterns for a given mode identification (k,m), and 
                                  corresponds to one of the observed period-sapcing patterns
            fin_frot:             astropy quantity
                                  near-core rotation rate value of the best-fitting asymptotic model
            fin_Pi0:              astropy quantity
                                  buoyancy travel time value of the best-fitting asymptotic model
            optimisation_method:  string; optional
                                  method used to sample and evaluate the parameter space (grid (default) or lmfit):
                                  grid:  the parameter space is sampled on a grid according to the given values
                                  lmfit: the optimisation is done using the Levenberg-Marquardt algorithm, as 
                                         implemented in lmfit in the 'leastsq' routine
            sample:               (structured) numpy array; optional
                                  the evaluated samples in the parameter space (in the grid-based evaluation).
                                  Required input when optimisation_method='grid'.
            chi2_vals:            numpy array; optional
                                  chi2 values for the samples in the parameter space (in the grid-based evaluation)
                                  Required input when optimisation_method='grid'.
            degree_freedom:       integer; optional
                                  number of degrees of freedom of the fitted period-spacing patterns
                                  Required input when optimisation_method='grid'.
        
        Returns:
            N/A
    """
    
    obspers = obsstar.period(split=True)
    e_obspers = obsstar.e_period(split=True)
    seqids = obsstar.seqid(split=True)
    
    plt.figure(1)
    ax = plt.subplot(111)
    plt.xlabel(r'period ($\sf d$)')
    plt.ylabel(r'$\Delta$P ($\sf s$)')

    for mode, obsper, e_obsper, seqid in zip(modes, obspers, e_obspers, seqids):
        
        pattern = mode.uniform_pattern(fin_frot, fin_Pi0)
        
        # dealing with r-modes...
        if(mode.kval < 0):
            pattern = -pattern[::-1]

        solid = []
        dashed = []

        no_gaps = np.r_[np.diff(seqid) == 1]
        per_nogaps = obsper[:-1][no_gaps].value
        e_per_nogaps = e_obsper[:-1][no_gaps].value
        sp_nogaps = np.diff(obsper)[no_gaps].to(u.s).value 
        e_sp_nogaps = np.sqrt(e_obsper[1:][no_gaps].to(u.s).value **2. + e_obsper[:-1][no_gaps].to(u.s).value **2.)
        
        for ii in np.arange(1,len(per_nogaps)):
            if(abs(per_nogaps[ii]-per_nogaps[ii-1]-sp_nogaps[ii-1]) <= 0.01*sp_nogaps[ii-1]):
                solid.append([(per_nogaps[ii-1],sp_nogaps[ii-1]),(per_nogaps[ii],sp_nogaps[ii])])
            else:
                dashed.append([(per_nogaps[ii-1],sp_nogaps[ii-1]),(per_nogaps[ii],sp_nogaps[ii])])
        
        plot_solid = LineCollection(solid,linewidths=1.,linestyles='solid',colors='k')
        plot_dashed = LineCollection(dashed,linewidths=1.,linestyles='dotted',colors='k')

        plt.plot(pattern[:-1], np.diff(pattern.to(u.s)), 'r-')
            
        ax.add_collection(plot_solid)
        ax.add_collection(plot_dashed)

        plt.errorbar(per_nogaps,sp_nogaps,xerr=e_per_nogaps,yerr=e_sp_nogaps,fmt='k.',mfc='k',ecolor='k',elinewidth=1.)
        
        plt.tight_layout()

    if(optimisation_method == 'grid'):
        len_frot = len(np.unique(sample['f_rot']))
        len_Pi0 = len(np.unique(sample['Pi0']))

        chi2grid = chi2_vals.reshape((len_Pi0,len_frot))
        chi2_sel = np.r_[chi2_vals <= np.amin(chi2_vals)*chi2.ppf(0.999,degree_freedom)/degree_freedom]
        
        unique_frot = np.unique(sample['f_rot'])
        unique_Pi0 = np.unique(sample['Pi0'])
        
        chi2_frot = np.array([np.nanmin(chi2_vals[np.r_[sample['f_rot'] == ifrot]]) for ifrot in unique_frot])
        chi2_Pi0 = np.array([np.nanmin(chi2_vals[np.r_[sample['Pi0'] == iPi0]]) for iPi0 in unique_Pi0])
        sel_chi2_frot = np.r_[chi2_frot <= np.amin(chi2_frot)*chi2.ppf(0.999,degree_freedom)/degree_freedom]
        sel_chi2_Pi0 = np.r_[chi2_Pi0 <= np.amin(chi2_Pi0)*chi2.ppf(0.999,degree_freedom)/degree_freedom]
        par_frot = np.polyfit(unique_frot[sel_chi2_frot], chi2_frot[sel_chi2_frot], deg=2)
        par_Pi0 = np.polyfit(unique_Pi0[sel_chi2_Pi0], chi2_Pi0[sel_chi2_Pi0], deg=2)
        highres_frot = np.linspace(np.amin(unique_frot[sel_chi2_frot]), np.amax(unique_frot[sel_chi2_frot]), len(unique_frot)*100)
        highres_Pi0 = np.linspace(np.amin(unique_Pi0[sel_chi2_Pi0]), np.amax(unique_Pi0[sel_chi2_Pi0]), len(unique_Pi0)*100)

        plt.figure(2)
        plt.subplot(311)
        plt.semilogy(sample['f_rot'][chi2_sel],chi2_vals[chi2_sel],'k.')
        plt.semilogy(highres_frot, np.polyval(par_frot, highres_frot), 'r-', lw=1.5)
        plt.xlabel(r'$\sf f_{\sf rot}$ ($\sf d^{-1}$)')
        plt.ylabel(r'$\chi^2$')
        plt.subplot(312)
        plt.semilogy(sample['Pi0'][chi2_sel],chi2_vals[chi2_sel],'k.')
        plt.semilogy(highres_Pi0, np.polyval(par_Pi0, highres_Pi0), 'r-', lw=1.5)
        plt.xlabel(r'$\sf \Pi_{\sf 0}$ (s)')
        plt.ylabel(r'$\chi^2$')
        
        ax3 = plt.subplot(313)
        ax3.set_facecolor("white")
        im = plt.imshow(np.log10(chi2grid), extent=(np.amin(sample['f_rot']),np.amax(sample['f_rot']),np.amin(sample['Pi0']),np.amax(sample['Pi0'])), interpolation='bicubic', origin='lower', aspect="auto",cmap=plt.get_cmap('gray'))
        levels = np.array([np.amin(chi2_vals)*chi2.ppf(0.99,degree_freedom)/degree_freedom])
        CS = plt.contour(np.log10(chi2grid), np.log10(levels),colors='r',linewidths=2,linestyles='dotted',  extent=(np.amin(sample['f_rot']),np.amax(sample['f_rot']),np.amin(sample['Pi0']),np.amax(sample['Pi0'])))
        levels = np.array([np.amin(chi2_vals)*chi2.ppf(0.95,degree_freedom)/degree_freedom])
        CS = plt.contour(np.log10(chi2grid), np.log10(levels),colors='r',linewidths=2,linestyles='dashed',  extent=(np.amin(sample['f_rot']),np.amax(sample['f_rot']),np.amin(sample['Pi0']),np.amax(sample['Pi0'])))
        levels = np.array([np.amin(chi2_vals)*chi2.ppf(0.6827,degree_freedom)/degree_freedom])
        CS = plt.contour(np.log10(chi2grid), np.log10(levels),colors='r',linewidths=2, extent=(np.amin(sample['f_rot']),np.amax(sample['f_rot']),np.amin(sample['Pi0']),np.amax(sample['Pi0'])))
        plt.plot([fin_frot.value], [fin_Pi0.value], 'wx')

        cbar = plt.colorbar(im)
        cbar.set_label(r'log $\sf\chi^2$')
        plt.xlabel(r'$\sf f_{\sf rot}$ ($\sf d^{-1}$)')
        plt.ylabel(r'$\sf \Pi_{\sf 0}$ (s)')

        plt.tight_layout()
    
    return



if __name__ == "__main__":
    
    # read in the chosen parameters and relevant filenames in the inlist
    gyre_dir, star, observationfile, optimisation_method, nthreads, frot_bnd, Pi0_bnd, kvals, mvals = read_inlist()
    
    Pi0min  = Pi0_bnd[0]
    Pi0max  = Pi0_bnd[1]
    Pi0step = Pi0_bnd[2]
    
    frotmin  = frot_bnd[0]
    frotmax  = frot_bnd[1]
    frotstep = frot_bnd[2]
    
    # reading in the observations
    obsstar = obspuls(star,observationfile)
    
    # preparing the asymptotic g-mode models
    modes = [asymp_gmodes(gyre_dir,kval=kval, mval=mval, nmin=1, nmax=150) for kval,mval in zip(kvals,mvals)]
    
    # Calculating the asymptotic fits and plotting the results
    if(optimisation_method == 'grid'):
        fin_frot, fin_e_frot, fin_Pi0, fin_e_Pi0, sample, chi2_vals, degree_freedom = asymptotic_fit(obsstar, modes, optimisation_method=optimisation_method, Pi0min=Pi0min, Pi0max=Pi0max, Pi0step=Pi0step, frotmin=frotmin, frotmax=frotmax, frotstep=frotstep,nthreads=nthreads)
        plot_results(obsstar, modes, fin_frot, fin_Pi0, optimisation_method=optimisation_method, sample=sample, chi2_vals=chi2_vals, degree_freedom=degree_freedom)
    else:
        fin_frot, fin_e_frot, fin_Pi0, fin_e_Pi0 = asymptotic_fit(obsstar, modes, optimisation_method=optimisation_method, Pi0min=Pi0min, Pi0max=Pi0max, frotmin=frotmin, frotmax=frotmax, nthreads=nthreads)
        plot_results(obsstar, modes, fin_frot, fin_Pi0, optimisation_method=optimisation_method)

    # Printing the final results
    print(f"f_rot: {fin_frot.value} +/- {fin_e_frot.value} c/d")
    print(f"Pi0: {fin_Pi0.value} +/- {fin_e_Pi0.value} s")
    
    plt.show()
