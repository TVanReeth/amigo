#!/usr/bin/env python3
#
# File: fit.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: <TO BE ADDED>
 
import numpy as np
from multiprocessing import Pool
from functools import partial

from scipy.stats import chi2 as chi2dist
from collections import namedtuple

from lmfit import Parameters, Minimizer
import astropy
import astropy.units as u

import grid



def fit_with_lmfit(obsstar, modes, diagnostic='spacings', Pi0min=2300., 
                   Pi0max=5600., frotmin=0.001, frotmax=2.5, alpha_in=None, 
                   use_sequence=True, nthreads=2):
    """
        Calculate the near-core rotation rate and buoyancy travel time for the 
        best-fitting asymptotic models to a (set of) observed g-mode 
        period-spacing pattern(s), assuming a provided (set of) mode 
        identification(s).
        
        In this subroutine, the model parameter values are optimised using  
        routines form the lmfit python package.

        Parameters:
            obsstar:      amigo pulsations object
                          contains the observed period-spacing patterns
            modes:        list of amigo gravity_modes objects
                          each gravity_modes object describes the possible
                          theoretical (asymptotic) period-spacing patterns for
                          a given mode identification (k,m) and corresponds to
                          one of the observed period-spacing patterns
            diagnostic:   string; optional
                          indicate how the observed period-spacing patterns are
                          evaluated:
                          'spacings':  the period-spacings are fitted as a 
                                       function of pulsation period. 
                          'frequency': the pulsation frequencies are fitted as a
                                       function of radial order n.
            Pi0min:       float; optional
                          the minimum considered value of the buoyancy radius 
                          (in seconds; default value = 2300s)
            Pi0max:       float; optional
                          the maximum considered value of the buoyancy radius 
                          (in seconds; default value = 5600s)
            frotmin:      float; optional
                          the minimum considered value of the near-core rotation
                          rate (in cycles per day; default value = 0 c/d)
            frotmax:      float; optional
                          the maximum considered value of the near-core rotation
                          rate (in cycles per day; default value = 2.5 c/d)
            alpha_in:     float; optional.
                          Input value for the phase term alpha_g. If given, it 
                          is fixed. If it is 'None' (default), the best value 
                          (matching the found frot and Pi0 values) are 
                          calculated.
            use_sequence: boolean; optional.
                          Indicate whether the radial order sequence of the 
                          observed pulsation frequencies has to be  blindly 
                          estimated (False) or whether any gaps that are 
                          (potentially) present in the pattern are sufficiently 
                          small to count the number of missing modes (True).
            nthreads:     integer; optional
                          the number of CPU threads used in the calculations 
                          (default value = 2).
        
        Returns:
            frot:         astropy quantity
                          near-core rotation rate of the best-fitting models
            e_frot:       astropy quantity
                          1-sigma error margin on the near-core rotation rate of
                          the best-fitting models
            Pi0:          astropy quantity
                          buoyancy travel time of the best-fitting models
            e_Pi0:        astropy quantity
                          1-sigma error margin on the buoyancy travel time of 
                          the best-fitting models
            alpha_out:    numpy array (float)
                          the phase terms alpha_g of the best-fitting asymptotic
                          patterns. One value is given for each observed 
                          pattern.
            e_alpha:      1-sigma error margins on the phase terms alpha_g of 
                          the best-fitting asymptotic patterns. One value is 
                          given for each observed pattern. If the used 
                          diagnostic is 'spacings' (and alpha was calculated, 
                          but not fitted) or if the fit did not converge, the 
                          returned value is -1.
            nvals:        numpy array
                          (estimated) radial orders of the observed pulsations,
                          for the rotation frequency frot and buoyancy radius 
                          Pi0 values calculated in this subroutine.
            deg_freedom:  integer
                          number of degrees of freedom of the fitted 
                          period-spacing patterns
    """
    
    ### fitting the models
    frotavg = 0.5*(frotmin+frotmax)
    Pi0avg  = 0.5*(Pi0min+Pi0max)
    
    param = Parameters()
    param.add(name='f_rot',value=frotavg,min=frotmin,max=frotmax,vary=True)
    param.add(name='Pi0',value=Pi0avg, min=Pi0min, max=Pi0max,vary=True)
    
    if(diagnostic == 'spacings'):
        param.add(name='alpha_g',value=alpha_in, min=0., max=1.,vary=False)
        deg_freedom = np.nansum(np.r_[np.diff(obsstar.seqid()) == 1]) - 2
    elif(alpha_in is None):
        param.add(name='alpha_g',value=0.5, min=0., max=1.,vary=True)
        deg_freedom = len(obsstar.frequency()) - 3
    else:
        param.add(name='alpha_g',value=alpha_in, min=0., max=1.,vary=False)
        deg_freedom = len(obsstar.frequency()) - 3
        
    mini = Minimizer(residuals_lmfit, param, 
                          fcn_args = (obsstar, modes, diagnostic, use_sequence),
                          nan_policy = 'omit')
    out = mini.minimize(method='leastsq')
        
    frot = out.params['f_rot'].value / u.d
    Pi0 = out.params['Pi0'].value * u.s
    min_chi2red = out.redchi
    
    if(out.params['f_rot'].stderr is None):
        e_frot = -1. / u.d
        e_Pi0  = -1. * u.s
    else:
        e_frot = out.params['f_rot'].stderr / u.d
        e_Pi0 = out.params['Pi0'].stderr * u.s
    
    if(diagnostic == 'spacings'):
        nvals, alpha_out = estimate_radn(obsstar, modes, frot, Pi0, 
                                     alpha_g=alpha_in, use_sequence=use_sequence)
        e_alpha = np.array([-1.] * len(modes))
    elif(out.params['f_rot'].stderr is None):
        nvals, alpha_out = estimate_radn(obsstar, modes, frot, Pi0, 
                                    alpha_g=alpha_in, use_sequence=use_sequence)
        e_alpha = np.array([-1.] * len(modes))
    else:
        alpha_g = out.params['alpha_g'].value
        nvals, alpha_out = estimate_radn(obsstar, modes, frot, Pi0, 
                                     alpha_g=alpha_g, use_sequence=use_sequence)
        e_alpha = np.array([out.params['alpha_g'].stderr] * len(modes))
        
    
    return frot, e_frot, Pi0, e_Pi0, alpha_out, e_alpha, min_chi2red, nvals, \
                                                                     deg_freedom



def fit_with_grid(obsstar, modes, diagnostic='spacings', Pi0min=2300., 
                  Pi0max=5600., dPi0=10., frotmin=0.001, frotmax=2.5, 
                  dfrot=0.001, alpha_in=None, use_sequence=True, nthreads=2):
    """
        Calculate the near-core rotation rate and buoyancy travel time for the 
        best-fitting asymptotic models to a (set of) observed g-mode 
        period-spacing pattern(s), assuming a provided (set of) mode 
        identification(s).
        
        In this subroutine, the model parameter values are sampled in a regular
        grid.

        Parameters:
            obsstar:      amigo pulsations object
                          contains the observed period-spacing patterns
            modes:        list of amigo gravity_modes objects
                          each gravity_modes object describes the possible
                          theoretical (asymptotic) period-spacing patterns for
                          a given mode identification (k,m) and corresponds to
                          one of the observed period-spacing patterns
            diagnostic:   string; optional
                          indicate how the observed period-spacing patterns are
                          evaluated:
                          'spacings':  the period-spacings are fitted as a 
                                       function of pulsation period. 
                          'frequency': the pulsation frequencies are fitted as a
                                       function of radial order n.
            Pi0min:       float; optional
                          the minimum considered value of the buoyancy radius 
                          (in seconds; default value = 2300s)
            Pi0max:       float; optional
                          the maximum considered value of the buoyancy radius 
                          (in seconds; default value = 5600s)
            dPi0:         float; optional
                          the step size of the buoyancy travel time (in 
                          seconds). default value = 10s.
            frotmin:      float; optional
                          the minimum considered value of the near-core 
                          rotation rate (in cycles per day; 
                          default value = 0 c/d)
            frotmax:      float; optional
                          the maximum considered value of the near-core 
                          rotation rate (in cycles per day; 
                          default value = 2.5 c/d)
            dfrot:        float; optional
                          the step size of the near-core rotation rate (in 
                          cycles per day). default value = 0.001 c/d.
            alpha_in:     float; optional.
                          Input value for the phase term alpha_g. If given, it 
                          is fixed. If it is 'None' (default), the best value 
                          (matching the found frot and Pi0 values) are 
                          calculated.
            nthreads:     integer; optional
                          the number of CPU threads used in the calculations 
                          (default value = 2).
        
        Returns:
            frot:         astropy quantity
                          near-core rotation rate of the best-fitting models
            e_frot:       astropy quantity
                          1-sigma error margin on the near-core rotation rate 
                          of the best-fitting models
            Pi0:          astropy quantity
                          buoyancy travel time of the best-fitting models
            e_Pi0:        astropy quantity
                          1-sigma error margin on the buoyancy travel time of 
                          the best-fitting models
            sample:       (structured) numpy array
                          the evaluated samples in the parameter space (in the
                          grid-based evaluation)
            chi2:         numpy array
                          chi2 values for the samples in the parameter space 
                          (in the grid-based evaluation)
            deg_freedom:  integer
                          number of degrees of freedom of the fitted 
                          period-spacing patterns
    """
    
    if(diagnostic == 'spacings'):
        deg_freedom = np.nansum(np.r_[np.diff(obsstar.seqid()) == 1]) - 2
    else:
        deg_freedom = len(obsstar.frequency()) - 3
        
    ### fitting the models
    # setting the sampling
    parameters = ['Pi0','f_rot']
    minbound = np.array([Pi0min, frotmin]) # min. values of Pi0 and f_rot
    maxbound = np.array([Pi0max,frotmax]) # max. values of Pi0 and f_rot
    regular_step = np.array([dPi0,dfrot]) 
    method = 'regular'
    
    sample = grid.sampling(parameters,minbound,maxbound,method=method,
                                                      regular_step=regular_step)
        
    # Parallellised calculations
    ppool = Pool(nthreads)
    chi2 = ppool.starmap(partial(rss, obsstar, modes, diagnostic,
                                   alpha_g=alpha_in, use_sequence=use_sequence),
                                            zip(sample['f_rot'], sample['Pi0']))
    chi2 = np.array(chi2) / deg_freedom
    invalid = np.r_[chi2 > 10.**59.]
    valid = ~invalid
    if(invalid.any() & valid.any()):
        chi2[invalid] = np.nanmax(chi2[~invalid])
        
    unique_frot = np.unique(sample['f_rot'])
    unique_Pi0 = np.unique(sample['Pi0'])
        
    chi2_frot = [np.nanmin(chi2[np.r_[sample['f_rot'] == ifrot]])
                                                       for ifrot in unique_frot]
    chi2_frot = np.array(chi2_frot)
    chi2_Pi0 = [np.nanmin(chi2[np.r_[sample['Pi0'] == iPi0]]) 
                                                         for iPi0 in unique_Pi0]
    chi2_Pi0 = np.array(chi2_Pi0)
    
    frot = unique_frot[np.argmin(chi2_frot)] / u.d
    Pi0 = unique_Pi0[np.argmin(chi2_Pi0)] * u.s
        
    chi2_1sigma = chi2dist.ppf(0.6827,deg_freedom)/deg_freedom
    chi2_3sigma = chi2dist.ppf(0.9973,deg_freedom)/deg_freedom
    sel_frot = unique_frot[np.r_[chi2_frot <= np.amin(chi2_frot)*chi2_3sigma]]
    sel_Pi0 = unique_Pi0[np.r_[chi2_Pi0 <= np.amin(chi2_Pi0)*chi2_3sigma]]
    
    try:
        e_frot = np.amax(np.abs(sel_frot - frot.value)) / 3. / u.d
        e_Pi0 = np.amax(np.abs(sel_Pi0 - Pi0.value)) / 3. * u.s
    except:
        e_frot = -1. / u.d
        e_Pi0  = -1. * u.s
    
    min_chi2red = np.nanmin(chi2)
    
    if(diagnostic == 'spacings'):
        nvals, alpha_out = estimate_radn(obsstar, modes, frot, Pi0, 
                                    alpha_g=alpha_in, use_sequence=use_sequence)
        e_alpha = np.array([-1.] * len(modes))
    elif(e_frot.value == -1.):
        nvals, alpha_out = estimate_radn(obsstar, modes, frot, Pi0, 
                                    alpha_g=alpha_in, use_sequence=use_sequence)
        e_alpha = np.array([-1.] * len(modes))
    else:
        nvals, alpha_out = estimate_radn(obsstar, modes, frot, Pi0, 
                                    alpha_g=alpha_in, use_sequence=use_sequence)
        
        sel_alphas = [estimate_radn(obsstar, modes, ifrot/u.d, iPi0*u.s, 
          use_sequence=use_sequence)[1] for ifrot,iPi0 in zip(sel_frot,sel_Pi0)]
        delta_alphas = np.array([np.array(sel_alpha)-alpha_out
                                                   for sel_alpha in sel_alphas])
        e_alpha = np.nanmax(delta_alphas, axis=0)

    return frot, e_frot, Pi0, e_Pi0, alpha_out, e_alpha, min_chi2red, nvals, \
           sample, chi2, deg_freedom
        


def interpolate(newx, oldx, oldy):
    """
        An auxiliary function to help with the interpolation of unsorted astropy
        quantities (based on the usual application of np.interp)

        Parameters:
            newx:   numpy array or astropy quantity
                    unsorted array of values onto which we want to interpolate 
                    the function
            oldx:   numpy array or astropy quantity
                    unsorted array of values for which we know the values of 
                    the function
            oldy:   numpy array or astropy quantity
                    function values for oldx
        
        Returns:
            newy:   numpy array of astropy quantity
                    function values for newx
    """
    
    assert type(oldx) == type(newx),\
           "amigo.fit.interpolate: error: please ensure that the data type of \
            the old and new independent input variables are the same."
    
    astro_qtype = astropy.units.quantity.Quantity
    
    if(type(oldx) == type(newx) == astro_qtype):
        interp_x = newx.to(oldx.unit)
        indsort_newx = np.argsort(interp_x.value)
        indsort_oldx = np.argsort(oldx.value)
        invsort_newx = np.argsort(indsort_newx)
        
        newx_sqval = interp_x.value[indsort_newx]
        oldx_sqval = oldx.value[indsort_oldx]
        
        if(type(oldy) == astro_qtype):
            oldy_sqval = oldy.value[indsort_oldx]
            interp_y = np.interp(newx_sqval, oldx_sqval, oldy_sqval) * oldy.unit
        else:
            oldy_sqval = oldy[indsort_oldx]
            interp_y = np.interp(newx_sqval, oldx_sqval, oldy_sqval)
        newy = interp_y[invsort_newx]

    elif((type(oldx) == type(newx)) & (type(oldx) != astro_qtype)):
        interp_x = newx
        indsort_newx = np.argsort(interp_x)
        indsort_oldx = np.argsort(oldx)
        invsort_newx = np.argsort(indsort_newx)
        
        newx_sqval = interp_x[indsort_newx]
        oldx_sqval = oldx[indsort_oldx]

        if(type(oldy) == astro_qtype):
            oldy_sqval = oldy.value[indsort_oldx]
            interp_y = np.interp(newx_sqval, oldx_sqval, oldy_sqval) * oldy.unit
        else:
            oldy_sqval = oldy[indsort_oldx]
            interp_y = np.interp(newx_sqval, oldx_sqval, oldy_sqval)
        newy = interp_y[invsort_newx]
    
    return newy



def rss(obsstar, modes, diagnostic, f_rot, Pi0, alpha_g=None,use_sequence=True):
    """
        Calculating the total residual sum of squares between the observed and 
        theoretical period-spacing patterns. Routine used in the grid-based 
        evaluation of the parameter space.

        Parameters:
            obsstar:      amigo pulsations object
                          contains the observed period-spacing patterns
            modes:        list of amigo gravity_modes objects
                          each gravity_modes object describes the possible 
                          theoretical (asymptotic) period-spacing patterns for a
                          given mode identification (k,m), and corresponds to  
                          one of the observed period-spacing patterns
            diagnostic:   string
                          indicate how the observed period-spacing patterns are
                          evaluated:
                          'spacings':  the period-spacings are fitted as a 
                                       function of pulsation period. 
                          'frequency': the pulsation frequencies are fitted as a
                                       function of radial order n.
            f_rot:        float
                          near-core rotation rate (in cycles per day) for which
                          we want to calculate the theoretical patterns
            Pi0:          float
                          buoyancy travel time value (in seconds) for which we 
                          want to calculate the theoretical patterns
            alpha_g:      float; optional
                          phase term dependent on the boundaries of the g-mode 
                          cavity
            use_sequence: boolean; optional. 
                          Only used when diagnostic = 'frequency'!
                          Indicate whether the radial order sequence of the 
                          observed pulsation frequencies has to be  blindly 
                          estimated (False) or whether any gaps that are 
                          (potentially) present in the pattern are sufficiently 
                          small to count the number of missing modes (True).
        
        Returns:
            res_sumsq:    float
                          the total residual sum of squares between the observed
                          and theoretical period-spacing patterns
    """
    
    res = residuals(obsstar, modes, diagnostic, f_rot/u.d, Pi0*u.s, 
                                     alpha_g=alpha_g, use_sequence=use_sequence)
    res_sumsq = np.nansum(res**2.)
    
    return res_sumsq


    
def residuals_lmfit(param, obsstar, modes, diagnostic, use_sequence=True):
    """
        Wrapper of the residuals() routine, used in the lmfit-based fitting of
        the observed period-spacing patterns

        Parameters:
            param:        lmfit Parameters object
                          contains the variable near-core rotation rate and 
                          buoyancy radius
            obsstar:      amigo pulsations object
                          contains the observed period-spacing patterns
            modes:        list of amigo gravity_modes objects
                          each gravity_modes object describes the possible 
                          theoretical (asymptotic) period-spacing patterns for a
                          given mode identification (k,m), and corresponds to
                          one of the observed period-sapcing patterns
            diagnostic:   string
                          indicate how the observed period-spacing patterns are
                          evaluated:
                          'spacings':  the period-spacings are fitted as a 
                                       function of pulsation period. 
                          'frequency': the pulsation frequencies are fitted as a
                                       function of radial order n.
            use_sequence: boolean; optional. 
                          Only used when diagnostic = 'frequency'!
                          Indicate whether the radial order sequence of the 
                          observed pulsation frequencies has to be  blindly 
                          estimated (False) or whether any gaps that are 
                          (potentially) present in the pattern are sufficiently 
                          small to count the number of missing modes (True).
            
        Returns:
            res:          numpy array of astropy quantity
                          residuals of the fit between the observed and 
                          theoretical period-spacing patterns
    """
    
    res = residuals(obsstar, modes, diagnostic, param['f_rot']/u.d,
                            param['Pi0']*u.s, alpha_g=float(param['alpha_g']), 
                                                     use_sequence=use_sequence)
    return res



def residuals(obsstar,modes,diagnostic,frot,Pi0,alpha_g=None,use_sequence=True):
    """
        Calculate the residuals of the fit between the observed and theoretical 
        period-spacing patterns, for a given near-core rotation rate, buoyancy 
        radius and phase term.

        Parameters:
            obsstar:      amigo pulsations object
                          contains the gravity_modes period-spacing patterns
            modes:        list of amigo asymptotic objects
                          each gravity_modes object describes the possible 
                          theoretical (asymptotic) period-spacing patterns for a
                          given mode identification (k,m), and corresponds to 
                          one of the observed period-spacing patterns
            diagnostic:   string
                          indicate how the observed period-spacing patterns are
                          evaluated:
                          'spacings':  the period-spacings are fitted as a 
                                       function of pulsation period. 
                          'frequency': the pulsation frequencies are fitted as a
                                       function of radial order n.
            frot:         astropy quantity
                          evaluated value of the near-core rotation rate
            Pi0:          astropy quantity
                          evaluated value of the buoyancy travel time
            alpha_g:      float; optional
                          evaluated value of the phase term (dependent on the
                          mode cavity boundaries). If alpha_g is None, its value
                          is set to 0.5.
            use_sequence: boolean; optional. 
                          Only used when diagnostic = 'frequency'!
                          Indicate whether the radial order sequence of the 
                          observed pulsation frequencies has to be  blindly 
                          estimated (False) or whether any gaps that are 
                          (potentially) present in the pattern are sufficiently 
                          small to count the number of missing modes (True).
            
        Returns:
            res:          numpy array of astropy quantity
                          residuals of the fit between the observed and 
                          theoretical period-spacing patterns
    """
    
    all_res = []
    
    if(diagnostic == 'spacings'):
        
        if(alpha_g is None):
            alpha_g = 0.5
            
        obsx, e_x, obsy, e_y = obsstar.patterns()
        
        patterns = []
        for mode in modes:
            if(mode.kval < 0):
                patterns.append(-mode.uniform_pattern(frot, Pi0, \
                                             alpha_g=alpha_g,unit='days')[::-1])
            else:
                patterns.append(mode.uniform_pattern(frot, Pi0, \
                                             alpha_g=alpha_g,unit='days'))
        
        modx = [pattern[:-1] for pattern in patterns]
        mody = [np.diff(pattern) for pattern in patterns]
    
    elif(diagnostic == 'frequency'):
        obsy = obsstar.frequency(split=True)
        e_y = obsstar.e_frequency(split=True)
        obsx, alpha_mod = estimate_radn(obsstar, modes, frot, Pi0, 
                                     alpha_g=alpha_g, use_sequence=use_sequence)
        
        modx = []
        mody = []
        for mode,ialpha in zip(modes,alpha_mod):
            if(mode.kval < 0):
                modx.append(mode.nvals)
                mody.append(-mode.uniform_pattern(frot, Pi0, alpha_g=ialpha))
            else:
                modx.append(mode.nvals)
                mody.append(mode.uniform_pattern(frot, Pi0, alpha_g=ialpha))
    
    for mode,iobsx,iobsy,ie_y,imodx,imody in zip(modes,obsx,obsy,e_y,modx,mody):
        min_mod = np.amin(imodx)
        max_mod = np.amax(imodx)
        
        # Evaluate if the range of the calculated model makes sense
        # if not, setting the residuals to a ridiculous high value...
        min_obs = np.amin(iobsx)
        max_obs = np.amax(iobsx)
        min_delta_obs = np.amin(np.abs(np.diff(iobsx)))
        
        if((min_obs >= min_mod) & (max_obs <= max_mod) & (min_delta_obs > 0)):
            mod_interp = interpolate(iobsx, imodx, imody)
            res = np.abs( (iobsy - mod_interp) / ie_y )
            
        else:
            res = np.ones(iobsx.shape) * 10.**30.
 
        all_res = all_res + list(res)
        
    all_res = np.array(all_res, dtype=float)
    
    return all_res



def estimate_radn(obsstar, modes, frot, Pi0, alpha_g=None, use_sequence=True):
    """
        Estimate or determine the radial orders of observed pulsation 
        frequencies, as well as the corresponding phase term alpha_g (if it is 
        not given in the input), given a selected mode identification (k,m) 
        listed in the 'mode' variable, and rotation frequency frot and buoyancy 
        radius Pi0 values. The phase term alpha_g is assumed to have a value 
        between 0 and 1, to avoid degeneracies with the estimated radial orders. 
        
        The quality of the estimate depends on the available information (i.e. 
        whether the pattern is sparse or near-continuous, where  the sequence of
        the observed modes as a function of radial order is known). There are 
        three separate options:
            A. the pattern is sparse and the sequence of radial orders is 
               (almost) completely unknown (use_sequence = False): in this case,
               the sequence of radial orders is estimated based on the input 
               parameter values for the model.
            B. there are no gaps in the pattern, so the radial orders of 
               consecutive pulsation modes can be counted (use_sequence = True).
            C. there are gaps in the pattern, but we can (or want to try to) 
               count the number of missing radial orders based on the observed 
               pulsation period spacings (use_sequence = True).
        
        Parameters:
            obsstar:      Amigo pulsations object
                          observed pulsation parameters, including the 
                          frequencies
            modes:        list of amigo gravity_modes objects
                          the g-mode pulsation geometries for which we fit the 
                          stellar model
            frot:         astropy quantity
                          the near-core rotation frequency of the fitted 
                          stellar model
            Pi0:          astropy quantity
                          the near-core rotation frequency of the fitted 
                          stellar model
            alpha_g:      float; optional
                          phase term of the asymptotic g-mode pattern. If it is
                          not given (default: alpha_g=None) its value is 
                          estimated in this subroutine.
            use_sequence: boolean; optional
                          indicate whether the radial order sequence of the 
                          observed pulsation frequencies has to be  blindly 
                          estimated (False) or whether any gaps that are 
                          (potentially) present in the pattern are sufficiently 
                          small to count the number of missing modes (True).
        
        Returns:
            radn_est:     numpy array (integers)
                          the (estimated) radial orders of the observed 
                          pulsation frequencies
            alpha_g_est:  float
                          the best-fitting value for the phase term alpha_g, 
                          given the input rotation frequency frot and buoyancy 
                          radius Pi0, and the estimated radial orders radn_est.
    """
    
    obsspin = obsstar.spin(modes, frot, split=True)
    obslam = [interpolate(ispin, mode.spin, mode.lam) 
                                         for ispin, mode in zip(obsspin, modes)]

    # If we have no good idea about any of the radial orders
    radn_ag = [np.sqrt(ilam)*ispin/(2.*frot.to(1./u.d).value*Pi0.to(u.d).value)
                                          for ispin,ilam in zip(obsspin,obslam)]
    
    # A. The case where we have no idea about the radial orders
    if(not use_sequence):
        est_seq = [np.array(np.linspace(0, len(ispin)-1, len(ispin)),dtype=int)
                                                           for ispin in obsspin]
       ##### ianchors = [np.argsort(iseq)[len(iseq)//2] for iseq in est_seq]
        ianchors = [int(len(iseq)//2) for iseq in est_seq]
        
        offsets = [iradn_alpha[ianchor] 
                               for iradn_alpha,ianchor in zip(radn_ag,ianchors)]
        anchor_radn = np.array(np.floor(np.array(offsets)), dtype=int)

        if(alpha_g is None):
            alpha_g_est = np.array(offsets) - anchor_radn
        else:
            alpha_g_est = alpha_g * np.ones(len(obsspin))
        
       #####  if(np.r_[alpha_g_est < 0.].any()):
       #####      alpha_below_zero = np.r_[alpha_g_est < 0.]
       #####      alpha_g_est[alpha_below_zero] += 1
       #####      anchor_radn[alpha_below_zero] -= 1

        # Counting the differences between the estimated radial orders
        # (trying to) make sure there are no duplicates!
        radn_est = [np.rint(iradn_alpha - ialpha_g_est) 
                     for iradn_alpha, ialpha_g_est in zip(radn_ag, alpha_g_est)]
        
        for iseq,ianc,iradn,ialpha in zip(est_seq,ianchors,radn_est,radn_ag):
            
            if(ialpha[-1] > ialpha[0]):
                alpha_incr = True
            else:
                alpha_incr = False
            
            for seq_ind in iseq[np.r_[iseq > ianc]]:
                if((iradn[seq_ind] == iradn[seq_ind-1]) & alpha_incr):
                    iradn[seq_ind:] += 1
                elif((iradn[seq_ind] == iradn[seq_ind-1]) & (not alpha_incr)):
                    iradn[seq_ind:] -= 1
    
            for seq_ind in iseq[np.r_[iseq < ianc]][::-1]:
                if((iradn[seq_ind] == iradn[seq_ind+1]) & alpha_incr):
                    iradn[seq_ind:] -= 1
                elif((iradn[seq_ind] == iradn[seq_ind+1]) & (not alpha_incr)):
                    iradn[seq_ind:] += 1
        
        # Using the estimated radn, optimise the alpha_g_est value
        if(alpha_g is None):
            alpha_g_est = [np.nanmean(iradn_alpha - iradn) 
                                for iradn_alpha, iradn in zip(radn_ag,radn_est)]
        
    else:
        
        sequence = obsstar.seqid(split=True)
        radn_est = []
        alpha_g_est = []
        
        for iseq, iradn_alpha in zip(sequence, radn_ag):
        
            
            # B. The case where we have an idea about the radial orders 
            #    and no gaps in the pattern
            if(np.r_[np.diff(iseq) == 1].all()):
                
                if((iradn_alpha[-1] - iradn_alpha[0])*(iseq[-1] - iseq[0]) > 0):
                    radn_offset = np.nanmean(iradn_alpha - iseq)
                    anchor_radn = np.rint(radn_offset)
    
                    if(alpha_g is None):
                        alpha_g_est.append( radn_offset - anchor_radn )
                    else:
                        alpha_g_est.append(alpha_g)
    
                    if(alpha_g_est[-1] < 0.):
                        alpha_g_est[-1] += 1
                        anchor_radn -= 1
                    radn_est.append(np.array(iseq + anchor_radn, dtype=int))
                    
                else:
                    radn_offset = np.nanmean(iradn_alpha - iseq[::-1])
                    anchor_radn = np.rint(radn_offset)
                    alpha_g_est.append( radn_offset - anchor_radn )
                    if(alpha_g_est[-1] < 0.):
                        alpha_g_est[-1] += 1
                        anchor_radn -= 1
                    radn_est.append(np.array(iseq[::-1]+anchor_radn, dtype=int))


            # C. The case where we have an idea about the radial orders 
            #    but also gaps in the pattern
            else:
                ianchor = np.argsort(iseq)[len(iseq)//2]
                anchor_radn = np.rint(iradn_alpha[ianchor])
   
                if(alpha_g is None):
                    alpha_g_est.append(iradn_alpha[ianchor] - anchor_radn)
                else:
                    alpha_g_est.append(alpha_g)
    
                if(alpha_g_est[-1] < 0.):
                    alpha_g_est[-1] += 1
                    anchor_radn -= 1
    
                # Counting the differences between the estimated radial orders
                # make sure that the gaps are treated correctly!
                iradn_est = np.rint(iradn_alpha - alpha_g_est[-1])
    
                seq_inds = np.array(np.linspace(0,len(iseq)-1,len(iseq)),
                                                                      dtype=int)
                
                for seq_ind in seq_inds[np.r_[seq_inds > ianchor]]:
                    if(np.abs(iseq[seq_ind] - iseq[seq_ind-1]) == 1):
                        if((iradn_est[-1]-iradn_est[0])*(iseq[-1]-iseq[0]) > 0):
                            iradn_est[seq_ind] = \
                                 iradn_est[seq_ind-1] + np.diff(iseq)[seq_ind-1]
                        else:
                            iradn_est[seq_ind] = \
                                 iradn_est[seq_ind-1] - np.diff(iseq)[seq_ind-1]
                    else:
                        if((iradn_est[seq_ind] <= iradn_est[seq_ind-1]) 
                                          & (iradn_alpha[-1] > iradn_alpha[0])):
                            iradn_est[seq_ind] = iradn_est[seq_ind-1] + 1
                        elif((iradn_est[seq_ind] >= iradn_est[seq_ind-1]) 
                                          & (iradn_alpha[-1] < iradn_alpha[0])):
                            iradn_est[seq_ind] = iradn_est[seq_ind-1] - 1
    
                for seq_ind in seq_inds[np.r_[seq_inds < ianchor]][::-1]:
                    if(np.abs(iseq[seq_ind] - iseq[seq_ind+1]) == 1):
                        if((iradn_est[-1]-iradn_est[0])*(iseq[-1]-iseq[0]) > 0):
                            iradn_est[seq_ind] = \
                                   iradn_est[seq_ind+1] - np.diff(iseq)[seq_ind]
                        else:
                            iradn_est[seq_ind] = \
                                   iradn_est[seq_ind+1] + np.diff(iseq)[seq_ind]
                    else:
                        if((iradn_est[seq_ind] >= iradn_est[seq_ind+1]) 
                                          & (iradn_alpha[-1] > iradn_alpha[0])):
                            iradn_est[seq_ind] = iradn_est[seq_ind+1] - 1
                        elif((iradn_est[seq_ind] <= iradn_est[seq_ind-1]) 
                                          & (iradn_alpha[-1] < iradn_alpha[0])):
                            iradn_est[seq_ind] = iradn_est[seq_ind+1] + 1
    
                # Using the estimated radn, optimise the alpha_g_est value
                if(alpha_g is None):
                    alpha_g_est[-1] = np.nanmean(iradn_alpha - iradn_est)
        
                radn_est.append(np.array(iradn_est, dtype=int))
    
    alpha_g_est = np.array(alpha_g_est, dtype=float)
    
    return radn_est, alpha_g_est



def stop_iteration(e_par, delta_par, sigma_sampling):
    """
        Evaluate if the parameter space around the found solution is 
        sufficiently densely sampled to stop the iterative evaluation.
        
        Parameters:
            e_par:          float or numpy array (dtype=float)
                            standard deviation of the parameter to evaluate
            delta_par:      float or numpy array (dtype=float)
                            current step size in the grid evaluation of
                            parameter "par".     
            sigma_sampling: float or integer
                            the number of grid data points required within
                            a parameter range equal to e_par.
        
        Returns:
            stop:           boolean
                            variable indicating if the iterative fitting of the
                            evaluated parameter can be stopped, i.e., if the 
                            current parameter sampling rate is sufficiently fine
                            to provide accurate results.
    """
    
    stop = e_par > sigma_sampling * delta_par
    return stop



def fit_iterative(obsstar, modes, diagnostic='spacings', Pi0min=2300., 
                  Pi0max=5600., dPi0=10., frotmin=0.001, frotmax=2.5, 
                  dfrot=0.001, alpha_in=None, use_sequence=True, nthreads=2, 
                  sigma_sampling=10, grid_scaling=5, cvg_rate=1.5, 
                  output_log=None):
    """
        Calculate the near-core rotation rate and buoyancy travel time for the 
        best-fitting asymptotic models to a (set of) observed g-mode 
        period-spacing pattern(s), assuming a provided (set of) mode 
        identification(s).
        
        In this subroutine, the fit_with_grid routine is applied to a coarse
        parameter grid, and subsequently the grid size and sampling are 
        iteratively optimised, until the optimal solution is found and the 
        confidence intervals are well-sampled.

        Parameters:
            obsstar:        amigo pulsations object
                            contains the observed period-spacing patterns
            modes:          list of amigo gravity_modes objects
                            each gravity_modes object describes the possible
                            theoretical (asymptotic) period-spacing patterns for
                            a given mode identification (k,m) and corresponds to
                            one of the observed period-spacing patterns
            diagnostic:     string; optional
                            indicate how the observed period-spacing patterns 
                            are evaluated:
                            'spacings':  the period-spacings are fitted as a 
                                         function of pulsation period. 
                            'frequency': the pulsation frequencies are fitted as
                                         a function of radial order n.
            Pi0min:         float; optional
                            the minimum considered value of the buoyancy radius 
                            (in seconds; default value = 2300s)
            Pi0max:         float; optional
                            the maximum considered value of the buoyancy radius 
                            (in seconds; default value = 5600s)
            dPi0:           float; optional
                            the step size of the buoyancy travel time (in 
                            seconds). default value = 10s.
            frotmin:        float; optional
                            the minimum considered value of the near-core 
                            rotation rate (in cycles per day; 
                            default value = 0 c/d)
            frotmax:        float; optional
                            the maximum considered value of the near-core 
                            rotation rate (in cycles per day; 
                            default value = 2.5 c/d)
            dfrot:          float; optional
                            the step size of the near-core rotation rate (in 
                            cycles per day). default value = 0.001 c/d.
            alpha_in:       float; optional.
                            Input value for the phase term alpha_g. If given, it
                            is fixed. If it is 'None' (default), the best value
                            (matching the found frot and Pi0 values) are
                            calculated.
            use_sequence:   boolean; optional
                            indicate whether the radial order sequence of the 
                            observed pulsation frequencies has to be  blindly 
                            estimated (False) or whether any gaps that are 
                            (potentially) present in the pattern are 
                            sufficiently small to count the number of missing 
                            modes (True).
            nthreads:       integer; optional
                            the number of CPU threads used in the calculations 
                            (default value = 2).
            sigma_sampling: integer; optional
                            the minimum number of projected data points that has
                            to be calculated per 1-sigma error margin for each 
                            parameter, before the iterative model optimisation 
                            is stopped (default value = 10).
            grid_scaling:   integer; optional
                            scaling factor for the required grid size at each 
                            iteration. At each step, the grid is centered around
                            the current best solution, with parameter X ranges 
                            given by: 2 * sigma_sampling * grid_scaling * dX,
                            where dX is the current step size for the parameter 
                            X in the grid. (default value = 5)
            cvg_rate:       float; optional(default value = 1.5).
                            convergence rate of the grid during the iteration.
                            At each iteration, the parameter step size dX in the
                            grid is decreased by this factor.
            output_log:     file object; optional
                            log file to which the (intermediate) model fitting 
                            results are written. When this variable is not given
                            (default = None), the results are written to the 
                            terminal.
                            
        Returns:
            frot:           astropy quantity
                            near-core rotation rate of the best-fitting models
            e_frot:         astropy quantity
                            1-sigma error margin on the near-core rotation rate 
                            of the best-fitting models
            Pi0:            astropy quantity
                            buoyancy travel time of the best-fitting models
            e_Pi0:          astropy quantity
                            1-sigma error margin on the buoyancy travel time of 
                            the best-fitting models
            sample:         (structured) numpy array
                            the evaluated samples in the parameter space (in the
                            grid-based evaluation)
            chi2:           numpy array
                            chi2 values for the samples in the parameter space 
                            (in the grid-based evaluation)
            deg_freedom:    integer
                            number of degrees of freedom of the fitted 
                            period-spacing patterns
    """
    

    # initialising cutoff values for the iterative calculations
    # these are automatically given new appropriate values during the run
    e_frot = 0. / u.d
    e_Pi0 = 0. * u.s
    
    stop = stop_iteration(e_frot.value, dfrot, sigma_sampling) & \
           stop_iteration(e_Pi0.value, dPi0, sigma_sampling)
    
    # setting the iteration counter for the iterative calculations
    iter_cntr = 0
    
    while not stop:
        if(iter_cntr > 0):
            dPi0  /= cvg_rate
            dfrot /= cvg_rate
            Pi0min = Pi0.value - sigma_sampling*grid_scaling*dPi0
            Pi0max = Pi0.value + sigma_sampling*grid_scaling*dPi0
            frotmin = frot.value - sigma_sampling*grid_scaling*dfrot
            frotmax = frot.value + sigma_sampling*grid_scaling*dfrot
        
        (frot, e_frot, Pi0, e_Pi0, alpha_out, e_alpha, min_chi2red, nvals, 
        sample, chi2_vals, degree_freedom) \
                = fit_with_grid(obsstar, modes, diagnostic=diagnostic,
                                Pi0min=Pi0min, Pi0max=Pi0max, dPi0=dPi0, 
                                frotmin=frotmin, frotmax=frotmax, dfrot=dfrot, 
                                alpha_in=alpha_in, use_sequence=use_sequence, 
                                nthreads=nthreads)
        
        if(output_log is None):
            
            print(f'\n    Iteration {iter_cntr+1}')
            
            # printing the intermediate results for each iteration
            print(f'        frot = {frot.value} +/- {e_frot.value}; '
                  f'Pi0 = {Pi0.value} +/- {e_Pi0.value}\n')
            
            # print the min, max, and range of radial orders for each fitted pattern
            for ipattern,nval in enumerate(nvals):
                
                print(f'        pattern {int(ipattern+1)}:    '
                      f'min n = {int(np.amin(nval))}; '
                      f'max n = {int(np.amax(nval))}; '
                      f'range: {int(np.amax(nval)-np.amin(nval))}\n')
        
        else:
            
            output_log.write(f'\nIteration {iter_cntr+1}')
            
            # printing the intermediate results for each iteration
            output_log.write(f'    frot = {frot.value} +/- {e_frot.value}; '
                  f'Pi0 = {Pi0.value} +/- {e_Pi0.value}\n\n')
            
            # print the min, max, and range of radial orders for each fitted 
            # pattern
            for ipattern,nval in enumerate(nvals):
                
                output_log.write(f'    pattern {int(ipattern+1)}:    '
                                 f'min n = {int(np.amin(nval))}; '
                                 f'max n = {int(np.amax(nval))}; '
                                 f'range: {int(np.amax(nval)-np.amin(nval))}\n\n')
        
        iter_cntr += 1
    
        stop = stop_iteration(e_frot.value, dfrot, sigma_sampling) & \
               stop_iteration(e_Pi0.value, dPi0, sigma_sampling)
    
    return frot, e_frot, Pi0, e_Pi0, alpha_out, e_alpha, min_chi2red, nvals, \
           sample, chi2_vals, degree_freedom

    
    
    
