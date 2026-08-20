#!/usr/bin/env python3
#
# File: compute_rotation.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Module to fit asymptotic g-mode patterns to observed period
#              spacings and derive the near-core rotation rate and the 
#              buoyancy travel time.

#              The algorithms in this module are described in:
#                  Van Reeth et al. 2016, A&A 593, A120
#                  Van Reeth et al. 2018, A&A 618, A24
#                  Mathis & Prat 2019, A&A 631, A26
#                  Henneco et al. 2021, A&A 648, A97
#                  Van Reeth et al. 2022, A&A 662, A58

import operator as op
import os
import subprocess as sp
import sys
import warnings

import numpy as np
from itertools import product
from scipy.stats import chi2 as chi2dist

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

mpl.rc('font',size=14)

import astropy
import astropy.units as u


import fit
from asymptotic_theory import gravity_modes
from observations import pulsations as obspuls


def read_inlist(inlist_filename):
    """
        Read in the parameter values of the user-defined inlist, as well as the
        configuration file and the default numerical settings.
        
        Parameters:
            inlist_filename: string
                             path to the user-defined inlist
        
        Returns:
            gyre_dir:        string
                             path to the GYRE installation directory
            starname:        string
                             name of the analysed star
            obsfile:         string
                             path to the file with the observed period-spacing 
                             patterns
            method:          string
                             method used to sample and evaluate the parameter 
                             space. This can be:
                             'grid':      the parameter space is sampled at 
                                          regular points, defined using the 
                                          frot_bnd and Pi0_bnd values.
                             'iterative': the parameter space is sampled at  
                                          coarsely spaced, regular grid points,
                                          which are iteratively sampled more 
                                          finely. 
                             'lmfit':     the parameter space is sampled using 
                                          python package lmfit.
            nthreads:        integer
                             number of threads to be used in the parallellised 
                             parts of the code
            frot_bnd:        numpy array (dtype = float)
                             minimum, maximum and step of the evaluated rotation
                             rates (in cycle per day)
            Pi0_bnd:         numpy array (dtype = float)
                             minimum, maximum and step of the evaluated buoyancy
                             travel times (in seconds)
            all_kvals:       list (of integers)
                             meridional degrees k of the mode identifications 
                             (k,m) that will be used to model the observed 
                             period spacings
            all_mvals:       list (of integers)
                             azimuthal orders m of the mode identifications 
                             (k,m) that will be used to model the observed 
                             period spacings
            diagnostic:      string
                             the quantity that will be evaluated to fit the 
                             g-mode patterns. This can be:
                             'spacings':  the routine will fit the spacings
                                          between consecutive pulsation periods
                                          as a function of the period.
                             'frequency': the routine will fit the pulsation
                                          frequencies as a function of the
                                          radial orders.
            use_sequence:    boolean
                             indicate if the sequence of observed g-modes is 
                             sufficiently complete to directly calculate the
                             number of missing radial orders in the gaps, if
                             there are any (value = True) or not (False). This
                             variable is only used when 
                             diagnostic = 'frequency'.
            sigma_sampling:  integer
                             the minimum number of projected data points that has
                             to be calculated per 1-sigma error margin for each 
                             parameter, before the iterative model optimisation 
                             is stopped 
            grid_scaling:    integer
                             scaling factor for the required grid size at each 
                             iteration. At each step, the grid is centered around
                             the current best solution, with parameter X ranges 
                             given by: 2 * sigma_sampling * grid_scaling * dX,
                             where dX is the current step size for the parameter 
                             X in the grid.
            cvg_rate:        float
                             convergence rate of the grid during the iteration.
                             At each iteration, the parameter step size dX in the
                             grid is decreased by this factor.
            interactive:     boolean
                             indicate if the output has to be shown as a 
                             Matplotlib figure rather than just saving it
            output_dir:      string
                             path to the directory where the results have to be
                             saved
    """
    
    ### default inlist files that are always read in
    config_file = f'{os.path.dirname(__file__)}/../defaults/config.dat'
    def_num = f'{os.path.dirname(__file__)}/../defaults/default_numerical.dat'
    
    ### the main read routine (is called three times within this function)
    def read_routine(filename, params):
        """
            Base routine used to read in the inlist, called multiple times 
            within this function.
        """
        
        # Loading the data from the inlist into the dictionaries
        file = open(filename,'r')
        lines = file.readlines()
        file.close()
        
        section = ''
        add_params = False
        
        for line in lines:
            
            if(len(line.strip()) == 0):
                pass
                
            elif(line.strip()[0] == '#'):
                pass
            
            elif(line.strip()[0] == '&'):
                section = line.strip()[1:].strip().split()[0]
                add_params = True
                
                if(section == 'modes'):
                    modes_id = 1 + len([key for key in params.keys() \
                                                             if 'modes' in key])
                    section = f'modes{modes_id}'
                    params[section] = {}
                    
            elif(line.strip()[0] == '/'):
                add_params = False
                
            elif(add_params):
                param_name = line.split('=')[0].strip().split()[0]
                str_vals = line.split('=')[1].split('#')[0].strip()
                
                # string type parameters
                if("'" in str_vals):
                    params[section][param_name] = str_vals.strip("'")
                elif('"' in str_vals):
                    params[section][param_name] = str_vals.strip('"')
                elif('True' in str_vals):
                    params[section][param_name] = True
                elif('False' in str_vals):
                    params[section][param_name] = False
                else:
                    # numerical type parameters
                    if('.' in str_vals):
                        dtype = float
                    else:
                        dtype = int
                    
                    if((len(str_vals.split()) == 1) & ('modes' not in section)):
                        params[section][param_name] = dtype(str_vals.strip())
                    else:
                        params[section][param_name] \
                               = np.array(str_vals.strip().split(), dtype=dtype)
            
        return params
    
    def check_parameters(key, value, good):
        """
            Subroutine to check whether the user setting is among the
            excepted values.
        """
        if value not in good:
            good_str = ', '.join(good)
            err1 = f'The chosen {key} is not among the excepted values.'
            err2 = f'Must be in {good_str}. Your setting is {value}'
            raise ValueError(f'{err1} {err2}')
    
    
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
        
    ### A dictionary of all the possible sections in the inlist, except 'modes'
    inlist_params = {'observations':{},'numerical':{},'star':{}, 'rotation':{},\
                                                                    'output':{}}
    
    ### Reading in the default AMiGO numerical settings
    inlist_params = read_routine(def_num, inlist_params)
    
    ### Reading in the custom AMiGO inlist
    inlist_params = read_routine(inlist_filename, inlist_params)
    
    ### Converting the data from the dictionaries into what is needed for 
    ### the AMiGO program
    output_dir = inlist_params['output']['output_dir']
    interactive = bool(inlist_params['output']['interactive'])
    
    starname = inlist_params['observations']['starname']
    obsfile = inlist_params['observations']['patterns']
    method = inlist_params['numerical']['optimisation_method']
    check_parameters('optimisation_method', method, ['grid', 'iterative', 'lmfit'])
    diagnostic = inlist_params['numerical']['diagnostic']
    check_parameters('diagnostic', diagnostic, ['spacings', 'frequency'])
    use_sequence = bool(inlist_params['numerical']['use_sequence'])
    sig_sampling = float(inlist_params['numerical']['sigma_sampling'])
    grid_scaling = float(inlist_params['numerical']['grid_scaling'])
    cvg_rate = float(inlist_params['numerical']['cvg_rate'])
    
    Pi0_bnd = inlist_params['star']['Pi0']
    frot_bnd = inlist_params['rotation']['frot']
    
    all_modes = [key for key in inlist_params.keys() if 'modes' in key]
    all_kvals = [inlist_params[mode]['k'] for mode in all_modes]
    all_mvals = [inlist_params[mode]['m'] for mode in all_modes]
    all_kvals = [np.array(k) for k in product(*all_kvals)]
    all_mvals = [np.array(m) for m in product(*all_mvals)]
    
    return gyre_dir, starname, obsfile, method, nthreads, frot_bnd, Pi0_bnd, \
           all_kvals, all_mvals, diagnostic, use_sequence, sig_sampling, \
           grid_scaling, cvg_rate, interactive, output_dir




def plot_results(obsstar, modes, fin_frot, fin_Pi0, nvals, alpha, method='grid', 
                 sample=None, chi2=None, deg_freedom=None,diagnostic='spacings',
                                                                 output_dir=''):
    """
        Plotting the observed period-spacing patterns (period spacing as a 
        function of pulsation period) with the best-fitting models. When the 
        'grid' optimisation method is used, the chi^2 values of the evaluated 
        grid are also shown with 1-, 2-, and 3-sigma contours.

        Parameters:
            obsstar:     amigo pulsations object
                         contains the observed period-spacing patterns
            modes:       list of amigo asymptotic objects
                         each asymptotic object describes the possible 
                         theoretical (asymptotic) period-spacing patterns for a
                         given mode identification (k,m), and corresponds to one
                         of the observed period-sapcing patterns
            fin_frot:    astropy quantity
                         near-core rotation rate value of the best-fitting 
                         asymptotic model
            fin_Pi0:     astropy quantity
                         buoyancy travel time value of the best-fitting 
                         asymptotic model
            nvals:       
            alpha:
            method:      string; optional
                         method used to sample and evaluate the parameter space
                         (grid (default) or lmfit):
                             grid:  the parameter space is sampled on a grid 
                                    according to the given values
                             lmfit: the optimisation is done using the 
                                    Levenberg-Marquardt algorithm, as 
                                    implemented in lmfit in the 'leastsq' 
                                    routine
            sample:      (structured) numpy array; optional
                         the evaluated samples in the parameter space (in the
                         grid-based evaluation). Required input when 
                         method = 'grid'.
            chi2:        numpy array; optional
                         chi2 values for the samples in the parameter space 
                         (in the grid-based evaluation) Required input when 
                         method = 'grid'.
            deg_freedom: integer; optional
                         number of degrees of freedom of the fitted 
                         period-spacing patterns. Required input when 
                         method = 'grid'.
            diagnostics: 
            output_dir:  string; optional
                         the output directory in which the figures can be saved
        
        Returns:
            N/A
    """
    col = ['red', 'tab:blue', 'tab:green', 'tab:purple']
    while len(modes) > len(col):
        # In case of more than 4 patterns add just red.
        # That many patterns are very rare
        col.append('r')
        
    if(diagnostic == 'spacings'):
        
        ### plotting the patterns
        split_periods = obsstar.period(split=True)
        split_ampls = obsstar.amplitude(split=True)
        perran = obsstar.period().ptp()
        permin = np.amin(obsstar.period()) - 0.05*perran
        permax = np.amax(obsstar.period()) + 0.05*perran
        
        pers, e_pers, dps, e_dps, solid, dashed = obsstar.patterns(to_plot=True)
        
        plt.figure()
        ax_patt = plt.subplot(211)
        plt.xlabel(r'period ($\sf d$)')
        plt.ylabel(r'amplitude')
        ax_sp = plt.subplot(212, sharex=ax_patt)
        plt.xlabel(r'period ($\sf d$)')
        plt.ylabel(r'$\Delta$P ($\sf s$)')
        
        # plotting the asymptotic models
        i = 0
        for ialpha,mode in zip(alpha,modes):
            
            pattern = mode.uniform_pattern(fin_frot, fin_Pi0, alpha_g=ialpha, unit='days')
            # dealing with r-modes...
            if(mode.kval < 0):
                pattern = -pattern[::-1]
                
            ax_patt.vlines(pattern, np.zeros(len(pattern)), 
                  np.ones(len(pattern))*np.amax(obsstar.amplitude())*1.1, colors=col[i])
            ax_sp.plot(pattern[:-1], np.diff(pattern.to(u.s)), '-', c=col[i])
            i += 1
    
        # plotting the observations
        for per,ampl in zip(split_periods,split_ampls):
            ax_patt.vlines(per, np.zeros(len(per)), ampl, colors='k',lw=1.5)
            
        spmin = 10000000.*u.s
        spmax = 0.*u.s
        
        for per,eper,dp,edp,line,dash in zip(pers,e_pers,dps,e_dps,solid,dashed):
        
            plot_line = LineCollection(line,linewidths=1.,linestyles='-',colors='k')
            plot_dash = LineCollection(dash,linewidths=1.,linestyles=':',colors='k')
            ax_sp.add_collection(plot_line)
            ax_sp.add_collection(plot_dash)

            ax_sp.errorbar(per,dp.to(u.s),xerr=eper,yerr=edp.to(u.s),
                                      fmt='k.',mfc='k',ecolor='k',elinewidth=1.)
            if(spmin > np.amin(dp.to(u.s)-edp.to(u.s))):
                spmin = np.amin(dp.to(u.s)-edp.to(u.s))
            if(spmax < np.amax(dp.to(u.s)+edp.to(u.s))):
                spmax = np.amax(dp.to(u.s)+edp.to(u.s))
        
        spran = spmax - spmin
        ax_sp.set_ylim(max(0.,spmin.value-0.05*spran.value), spmax.value+0.05*spran.value)
        ax_patt.set_ylim(0., np.amax(obsstar.amplitude())*1.1)
        plt.xlim(permin.value, permax.value)
    
    else:
        
        ### plotting the patterns
        split_freqs = obsstar.frequency(split=True)
        split_efreqs = obsstar.e_frequency(split=True)
        split_ampls = obsstar.amplitude(split=True)
        freqran = obsstar.frequency().ptp()
        freqmin = np.amin(obsstar.frequency()) - 0.05*freqran
        freqmax = np.amax(obsstar.frequency()) + 0.05*freqran
        
        plt.figure()
        ax_patt = plt.subplot(211)
        plt.xlabel(r'frequency ($\sf d^{-1}$)')
        plt.ylabel(r'amplitude')
        ax_diff = plt.subplot(212, sharex=ax_patt)
        plt.xlabel(r'frequency ($\sf d^{-1}$)')
        plt.ylabel(r'$\delta$f ($\sf d^{-1}$)')
        
        i = 0
        for mode,freq,efreq,ampl,radn, ialpha in zip(modes,split_freqs,split_efreqs,split_ampls,nvals, alpha):
            # plotting the asymptotic models
            pattern = mode.uniform_pattern(fin_frot, fin_Pi0, alpha_g=ialpha)
            # dealing with r-modes...
            if(mode.kval < 0):
                pattern = -pattern

            ax_patt.vlines(pattern, np.zeros(len(pattern)), 
                  np.ones(len(pattern))*np.amax(obsstar.amplitude())*1.1, colors=col[i])

            ax_diff.plot((freqmin.value,freqmax.value), (0., 0.), '--', c=col[i])
        
            # plotting the observations
            ax_patt.vlines(freq, np.zeros(len(freq)), ampl, colors='k',lw=1.5)
            ax_diff.errorbar(freq, freq-np.interp(np.array(radn),mode.nvals,pattern),
                             xerr=efreq, yerr=efreq, fmt='k.', mfc='k', ecolor='k', elinewidth=1.)
            i += 1

        ax_patt.set_ylim(0., np.amax(obsstar.amplitude())*1.1)
        plt.xlim(freqmin.value, freqmax.value)
        
    plt.tight_layout()
    
    
    if(os.path.exists(output_dir)):
        mode_id_str = ''
        for mode in modes:
            mode_id_str = mode_id_str + f'k{int(mode.kval)}m{int(mode.mval)}'
        filename = f'{obsstar.starname}_pattern-fit_{mode_id_str}'
        plt.savefig(f'{output_dir}/{obsstar.starname}/{filename}.png')
        
    
    ### Part 2: plotting the results as a function of the grid (if used)
    if((method == 'grid') | (method == 'iterative')):
        
        redchi2_cutoff = chi2dist.ppf(0.999,deg_freedom)/deg_freedom
        
        N_frot = len(np.unique(sample['f_rot']))
        N_Pi0 = len(np.unique(sample['Pi0']))
        loggrid = np.log10(chi2.reshape((N_Pi0,N_frot)))
        
        uni_frot = np.unique(sample['f_rot'])
        uni_Pi0 = np.unique(sample['Pi0'])
        
        chi2_plot = np.r_[chi2 <= np.amin(chi2)*redchi2_cutoff]
        
        chi2_frot = np.array([np.nanmin(chi2[np.r_[sample['f_rot'] == ifrot]]) 
                                                         for ifrot in uni_frot])
        chi2_Pi0 = np.array([np.nanmin(chi2[np.r_[sample['Pi0'] == iPi0]]) 
                                                           for iPi0 in uni_Pi0])
        extent_plot = (np.amin(uni_frot),np.amax(uni_frot),
                                              np.amin(uni_Pi0),np.amax(uni_Pi0))

        plt.figure()
        plt.subplot(311)
        plt.semilogy(sample['f_rot'][chi2_plot],chi2[chi2_plot],'k.')
        plt.xlabel(r'$\sf f_{\sf rot}$ ($\sf d^{-1}$)')
        plt.ylabel(r'$\chi^2$')
        plt.subplot(312)
        plt.semilogy(sample['Pi0'][chi2_plot],chi2[chi2_plot],'k.')
        plt.xlabel(r'$\sf \Pi_{\sf 0}$ (s)')
        plt.ylabel(r'$\chi^2$')
        
        ax3 = plt.subplot(313)
        ax3.set_facecolor("white")
        
        im = plt.imshow(loggrid, extent=extent_plot, interpolation='bicubic', 
                       origin='lower', aspect="auto", cmap=plt.get_cmap('gray'))
        levels = [np.amin(chi2)*chi2dist.ppf(0.99,deg_freedom)/deg_freedom]
        CS = plt.contour(loggrid, np.log10(np.array(levels)), colors='r', 
                               linewidths=2, linestyles=':', extent=extent_plot)
        levels = [np.amin(chi2)*chi2dist.ppf(0.95,deg_freedom)/deg_freedom]
        CS = plt.contour(loggrid, np.log10(np.array(levels)), colors='r', 
                              linewidths=2, linestyles='--', extent=extent_plot)
        levels = [np.amin(chi2)*chi2dist.ppf(0.6827,deg_freedom)/deg_freedom]
        CS = plt.contour(loggrid, np.log10(np.array(levels)), colors='r', 
                                               linewidths=2, extent=extent_plot)
        plt.plot([fin_frot.value], [fin_Pi0.value], 'wx')
        
        plt.xlim(np.amin(sample['f_rot'][chi2_plot]),np.amax(sample['f_rot'][chi2_plot]))
        plt.ylim(np.amin(sample['Pi0'][chi2_plot]),np.amax(sample['Pi0'][chi2_plot]))
        
        cbar = plt.colorbar(im)
        cbar.set_label(r'log $\sf\chi^2$')
        plt.xlabel(r'$\sf f_{\sf rot}$ ($\sf d^{-1}$)')
        plt.ylabel(r'$\sf \Pi_{\sf 0}$ (s)')

        plt.tight_layout()

        if(os.path.exists(output_dir)):
            mode_id_str = ''
            for mode in modes:
                mode_id_str = mode_id_str + f'k{int(mode.kval)}m{int(mode.mval)}'
            filename = f'{obsstar.starname}_grid-eval_{mode_id_str}'
            plt.savefig(f'{output_dir}/{obsstar.starname}/{filename}.png')
    
    return



if __name__ == "__main__":
    
    # avoiding unnecessary warnings
    warnings.simplefilter('ignore')
    
    # read in the chosen parameters and relevant filenames in the inlist
    inlist_filename = sys.argv[1]
    
    gyre_dir, star, obsfile, method, nthreads, frot_bnd, Pi0_bnd, all_kvals, \
    all_mvals, diagnostic, use_sequence, sig_sampling, grid_scaling, cvg_rate, \
                          interactive, output_dir = read_inlist(inlist_filename)
    
    Pi0min  = Pi0_bnd[0]
    Pi0max  = Pi0_bnd[1]
    dPi0 = Pi0_bnd[2]
    
    alpha_in = None
    
    frotmin  = frot_bnd[0]
    frotmax  = frot_bnd[1]
    dfrot = frot_bnd[2]
    
    ### sigma_sampling=10, grid_scaling=5, cvg_rate=1.5
    
    # reading in the observations
    obsstar = obspuls(star,obsfile)
    
    # preparing the output directory and lo file if the path in the inlist 
    # exists
    if(os.path.exists(output_dir)):
        if not os.path.exists(f'{output_dir}/{obsstar.starname}'):
            os.makedirs(f'{output_dir}/{obsstar.starname}')
        output_log = open(f'{output_dir}/{obsstar.starname}/' + \
                          f'{obsstar.starname}_modelling.log','w')
    else:
        output_log = None
    
    # Looping over all given (k,m) mode identifications, to find the best one.
    for kvals,mvals in zip(all_kvals,all_mvals):
        
        modes_str = ""
        
        imode = 0
        while(imode+1 < len(kvals)):
            modes_str = modes_str + f"({kvals[imode]},{mvals[imode]}), "
            imode += 1
        if(len(kvals) > 1):
            modes_str = modes_str[:-2] + " and "
        modes_str = modes_str + f"({kvals[imode]},{mvals[imode]})"
        
        log_string = f"\nfitting (k,m) = {modes_str} to the observed g-modes "+\
                     f"of {obsstar.starname}:"
        
        print(log_string)
        if(output_log is not None):
            output_log.write(f"{log_string}\n")
        
        # preparing the asymptotic g-mode models
        modes = [gravity_modes(gyre_dir,kval=k, mval=m) for k,m in zip(kvals,mvals)]
        
        # Calculating the asymptotic fits and plotting the results
        if(method == 'grid'):
            fin_frot, fin_e_frot, fin_Pi0, fin_e_Pi0, alpha_out, e_alpha, \
            min_chi2red, nvals, sample, chi2, deg_freedom = \
                 fit.fit_with_grid(obsstar, modes, diagnostic=diagnostic, 
                                   Pi0min=Pi0min,Pi0max=Pi0max,dPi0=dPi0,
                                   frotmin=frotmin,frotmax=frotmax,dfrot=dfrot, 
                                   alpha_in=alpha_in, use_sequence=use_sequence,
                                   nthreads=nthreads)

            plot_results(obsstar, modes, fin_frot, fin_Pi0, nvals, alpha_out,
                         method=method, sample=sample, chi2=chi2, 
                         deg_freedom=deg_freedom, diagnostic=diagnostic, 
                         output_dir=output_dir)
    
        elif(method == 'iterative'):
            fin_frot, fin_e_frot, fin_Pi0, fin_e_Pi0, alpha_out, e_alpha, \
            min_chi2red, nvals, sample, chi2, deg_freedom = \
                fit.fit_iterative(obsstar, modes, diagnostic=diagnostic, 
                                  Pi0min=Pi0min,Pi0max=Pi0max,dPi0=dPi0,
                                  frotmin=frotmin,frotmax=frotmax,dfrot=dfrot, 
                                  alpha_in=alpha_in, use_sequence=use_sequence,
                                  nthreads=nthreads, sigma_sampling=sig_sampling,
                                  grid_scaling=grid_scaling, cvg_rate=cvg_rate, 
                                  output_log=output_log)

            plot_results(obsstar, modes, fin_frot, fin_Pi0, nvals, alpha_out, 
                         method=method, sample=sample, chi2=chi2,
                         deg_freedom=deg_freedom, diagnostic=diagnostic, 
                         output_dir=output_dir)
    
        else:
            fin_frot, fin_e_frot, fin_Pi0, fin_e_Pi0, alpha_out, e_alpha, \
            min_chi2red, nvals, deg_freedom = \
                fit.fit_with_lmfit(obsstar, modes, diagnostic=diagnostic, 
                                   Pi0min=Pi0min, Pi0max=Pi0max, 
                                   frotmin=frotmin, frotmax=frotmax, 
                                   alpha_in=alpha_in, use_sequence=use_sequence,
                                   nthreads=nthreads)
            
            plot_results(obsstar, modes, fin_frot, fin_Pi0, nvals, alpha_out, 
                         method=method, diagnostic=diagnostic, 
                         output_dir=output_dir)
    
        # Printing the final results
        print(f"    chi2_red = {min_chi2red}")
        print(f"    f_rot: {fin_frot.value} +/- {fin_e_frot.value} c/d")
        print(f"    Pi0: {fin_Pi0.value} +/- {fin_e_Pi0.value} s")
        
        if(diagnostic == 'frequency'):
            for ii, ialpha, e_ialpha in zip(np.arange(1,1+len(alpha_out)), alpha_out, e_alpha):
                print(f"    alpha{int(ii)}: {ialpha} +/- {e_ialpha}")
        else:
            for ii, ialpha in zip(np.arange(1,1+len(alpha_out)), alpha_out):
                print(f"    estimated alpha{int(ii)}: {ialpha}")
        print("\n")
        
        if(output_log is not None):
            output_log.write(f"chi2_red = {min_chi2red}\n")
            output_log.write(f"f_rot: {fin_frot.value} +/- {fin_e_frot.value} c/d\n")
            output_log.write(f"Pi0: {fin_Pi0.value} +/- {fin_e_Pi0.value} s\n")
        
            if(diagnostic == 'frequency'):
                for ii, ialpha, e_ialpha in zip(np.arange(1,1+len(alpha_out)), alpha_out, e_alpha):
                    output_log.write(f"    alpha{int(ii)}: {ialpha} +/- {e_ialpha}\n")
            else:
                for ii, ialpha in zip(np.arange(1,1+len(alpha_out)), alpha_out):
                    output_log.write(f"    estimated alpha{int(ii)}: {ialpha}\n")
            output_log.write("\n")
    
    if(os.path.exists(output_dir)):
        output_log.close()
        
    if(interactive | (not os.path.exists(output_dir))):
        plt.show()
        
