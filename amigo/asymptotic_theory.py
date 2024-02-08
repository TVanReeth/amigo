#!/usr/bin/env python3
#
# File: asymptotic_theory.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: a python class with subroutines to calculate asymptotic g-mode
#              period-spacing patterns. This can be done in the framework of the
#              traditional approximation of rotation, assuming:
#                  (1) uniform rotation and spherical symmetry
#                  (2) radially differential rotation and spherical symmetry
#                  (3) uniform rotation and weak centrifugal acceleration

import numpy as np
import astropy.units as u
import sys

from itertools import product
from multiprocessing import Pool

from amigo.centrifugal_TAR import hough

class gravity_modes(object):
    """ 
        A python class to calculate period spacing patterns in the asymptotic 
        regime using the TAR. Possible options include:
        - the case of uniform rotation and ignoring the centrifugal acceleration
          (i.e., following Van Reeth et al. 2016, A&A 593, A120)
        - the case of radially differential rotation and ignoring the centrifu- 
          gal acceleration (Van Reeth et al. 2018, A&A 618, A24).  
        - the case of uniform rotation with an approximation of the centrifugal
          acceleration (Henneco et al. 2021, A&A 648, A97).
    """
    
    def __init__(self,gyre_dir,kval=0,mval=1,nmin=1,nmax=150):
        """
            Setting up the environment/asymptotic object to calculate g-mode 
            period-spacing patterns.
            
            Parameters:
                gyre_dir: string
                          The GYRE 6.x (or higher) installation directory.
                kval:     integer
                          Part of the mode identification of the pulsation 
                          pattern. For gravito-inertial modes with an equivalent
                          in a non-rotating star, k = l - |m|. 
                          Default value = 0.
                mval:     integer
                          Part of the mode identification of the pulsation 
                          pattern: azimuthal order. mval > 0 for prograde modes.
                          Default value = 1.
                nmin:     integer
                          The minimum radial order that we will (attempt to) 
                          calculate. Default value = 1.
                nmax:     integer
                          The maximum radial order that we will (attempt to) 
                          calculate. Default value = 150.
        """
        
        self.kval = int(kval)
        self.mval = int(mval)
        self.lval = abs(kval) + abs(mval)
        self.nvals = np.arange(nmin,nmax+0.1,1.)
        
        self.lam_fun  = self._retrieve_laplacegrid(gyre_dir)
        self.spin, self.lam, self.spinsqlam, self.fun1, self.fun2 \
                                                    = self._sample_laplacegrid()
        
        
    
    def _retrieve_laplacegrid(self,gyre_dir):
        """
            Retrieving the function lambda(nu) given in GYRE v6.x (or higher).
            
            Parameters:
                self:     gravity_modes object
                gyre_dir: string
                          The GYRE 6.x installation directory.
            
            Returns:
                lam_fun: function
                         A function to calculate lambda, given spin parameter 
                         values as input.
        """
        
        if(self.kval >= 0):
            kstr = f'+{self.kval}'
        else:
            kstr = f'{self.kval}'
        if(self.mval >= 0):
            mstr = f'+{self.mval}'
        else:
            mstr = f'{self.mval}'
        
        infile = f'{gyre_dir}/data/tar/tar_fit.m{mstr}.k{kstr}.h5'
        
        sys.path.append(gyre_dir+'/src/tar/')
        import gyre_tar_fit
       # import gyre_cheb_fit
        
        tf = gyre_tar_fit.TarFit.load(infile)
        lam_fun = np.vectorize(tf.lam)
        
        return lam_fun
    
    
    
    def _sample_laplacegrid(self,spinmin=None,spinmax=1000.,spindensity=1.):
        """
            Sampling the function lambda(nu) that was set up in 
            _retrieve_laplacegrid().
            This subroutine includes a hard-coded custom sampling density 
            function for the spin parameter.
            
            Parameters:
                self:        gravity_modes object
                spinmin:     float; optional
                             The minimum spin parameter value for which lambda 
                             eigenvalues will be retrieved (following the 
                             Laplace tidal equation). In the default case (None)
                             the value will be:
                                 * -0.1 
                                   (for a non-zonal gravito-inertial mode, that 
                                   is, k >=0 and m != 0).
                                 * -spinmax 
                                   (for a zonal gravito-inertial mode, that is, 
                                   k >=0 and m != 0).)
                                 * (|m| + |k|) * (|m| + |k| - 1) / |m| 
                                   (for an r-mode, that is, k < 0 and m < 0).
                spinmax:     float; optional
                             The maximum spin parameter value for which lambda 
                             eigenvalues will be retrieved (following the 
                             Laplace tidal equation). (Default value = 1000).
                spindensity: float; optional
                             A scaling factor for the sampling density function.
                             The default value (= 1) results in 20000 data 
                             points for the spin parameter range [0, 100].

            Returns:
                spin:        numpy array, dtype=float
                             The spin parameter values for which lambda 
                             eigenvalues are returned
                lam:         numpy array, dtype=float
                             The lambda eigenvalues corresponding to the spin 
                             parameter values in the array 'spin'
                spinsqlam:   numpy array, dtype=float
                             spin * sqrt(lam)
                fun1:        numpy array, dtype=float
                             The first of two functions required to calculate 
                             patterns for a differentially rotating star, using 
                             Taylor expansion.
                fun2:        numpy array, dtype=float
                             The second of two functions required to calculate 
                             patterns for a differentially rotating star, using 
                             Taylor expansion.
        """
        
        if((self.kval >= 0) & (self.mval != 0)):
            
            if(spinmin is None):
                spinmin = -0.1
            
            # Relatively "ad hoc" optimal sampling (based on "experience")
            nspinmin = round(spindensity * 20000. * (np.log10(1.+abs(spinmin))
                                                         / np.log10(101.))**0.5)
            nspinmax = round(spindensity * 20000. * (np.log10(1.+abs(spinmax))
                                                         / np.log10(101.))**0.5)
            
            if((spinmin < 0.) & (spinmax <= 0.)):
                spin = -10.**(np.linspace(np.log10(1.+abs(spinmin))**0.5,
                   np.log10(1.+abs(spinmax))**0.5,int(nspinmin-nspinmax))**2.) \
                                                                            + 1.
            elif((spinmin >= 0.) & (spinmax > 0.)):
                spin = 10.**(np.linspace(np.log10(1.+spinmin)**0.5, 
                        np.log10(1.+spinmax)**0.5,int(nspinmax-nspinmin))**2.) \
                                                                            - 1.
            else:
                spinneg = -10.**(np.linspace(np.log10(1.+abs(spinmin))**0.5,0.,
                                                        int(nspinmin))**2.) + 1.
                spinpos = 10.**(np.linspace(0.,np.log10(1.+spinmax)**0.5,
                                                        int(nspinmax))**2.) - 1.
                spin = np.unique(np.hstack((spinneg,spinpos)))
        
        
        elif((self.kval >= 0) & (self.mval == 0)):
            
            if(spinmin is None):
                spinmin = -spinmax
                
            # Relatively "ad hoc" optimal sampling (based on "experience")
            nspinmin = round(spindensity * 20000. * (np.log10(1.+abs(spinmin))
                                                         / np.log10(101.))**0.5)
            nspinmax = round(spindensity * 20000. * (np.log10(1.+abs(spinmax))
                                                         / np.log10(101.))**0.5)
            
            if((spinmin < 0.) & (spinmax <= 0.)):
                spin = -10.**(np.linspace(np.log10(1.+abs(spinmin))**0.5,
                   np.log10(1.+abs(spinmax))**0.5,int(nspinmin-nspinmax))**2.) \
                                                                            + 1.
            elif((spinmin >= 0.) & (spinmax > 0.)):
                spin = 10.**(np.linspace(np.log10(1.+spinmin)**0.5,
                        np.log10(1.+spinmax)**0.5,int(nspinmax-nspinmin))**2.) \
                                                                            - 1.
            else:
                spinneg = -10.**(np.linspace(np.log10(1.+abs(spinmin))**0.5,0.,
                                                        int(nspinmin))**2.) + 1.
                spinpos = 10.**(np.linspace(0.,np.log10(1.+spinmax)**0.5,
                                                        int(nspinmax))**2.) - 1.
                spin = np.unique(np.hstack((spinneg,spinpos)))
        
        
        else:
            if(spinmin is None):
                spinmin = float((abs(self.mval) + abs(self.kval)) *
                         (abs(self.mval) + abs(self.kval) - 1)) / abs(self.mval)
            
            # Relatively "ad hoc" optimal sampling (based on "experience")
            nspinmax = round(spindensity * 20000. * (np.log10(1.+abs(spinmax))
                                                         / np.log10(101.))**0.5)
            
            spinpos = 10.**(np.linspace(np.log10(1.)**0.5,
                                        np.log10(1.+spinmax)**0.5,
                                        int(nspinmax))[1:]**2.) - 1. + spinmin
            spinneg = np.linspace(0.,spinmin,100)
            spin = np.unique(np.hstack((spinneg,spinpos)))
            
        
        # Obtaining the eigenvalues lambda
        lam = self.lam_fun(spin)
        
        # Limiting ourselves to the part of the arrays that exists...
        lam_exists = np.isfinite(lam) & np.r_[lam != 0.]
        spin = spin[lam_exists]
        lam = lam[lam_exists]
        
        # A required array to determine the near-core rotation rate
        spinsqlam = spin*np.sqrt(lam)

        # Calculating the derivatives
        dlamdnu = np.gradient(lam)/np.gradient(spin)
        d2lamdnu2 = np.gradient(dlamdnu)/np.gradient(spin)
        
        # Array required to account for 1st-order effects of differential 
        # near-core rotation
        fun1 = 0.5 * spin * ( (dlamdnu*(1.+(self.mval*spin/2.))/lam)+self.mval )

        # Array required to account for 2nd-order effects of differential 
        # near-core rotation
        fun2 = 0.25 * spin**2. * ( ( ( (2.*d2lamdnu2/lam) - (dlamdnu/lam)**2. )\
                          * (1.+(self.mval*spin/2.))**2.) \
                       + (3.*self.mval*dlamdnu*(1. + (self.mval*spin/2.))/lam) \
                       + (2.*self.mval**2.) )

        return spin, lam, spinsqlam, fun1, fun2

    
    
    
    def update_laplacegrid(self,spinmin=None,spinmax=1000.,spindensity=1.):
        """
            Recalculating the spin aparameter range and the corresponding lambda
            eigenvalues included within the asymptotic object.
            
            Parameters:
                self:        gravity_modes object
                spinmin:     float; optional
                             The minimum spin parameter value for which lambda 
                             eigenvalues will be retrieved (following the 
                             Laplace tidal equation). In the default case (None)
                             the value will be:
                                 * -0.1 
                                   (for a non-zonal gravito-inertial mode, that 
                                   is, k >=0 and m != 0).
                                 * -spinmax 
                                   (for a zonal gravito-inertial mode, that is, 
                                   k >=0 and m != 0).)
                                 * (|m| + |k|) * (|m| + |k| - 1) / |m| 
                                   (for an r-mode, that is, k < 0 and m < 0).
                spinmax:     float; optional
                             The maximum spin parameter value for which lambda 
                             eigenvalues will be retrieved (following the 
                             Laplace tidal equation). (default value = 1000).
                spindensity: float; optional
                             A scaling factor for the sampling density function.
                             The default value (=1) results in 20000 data points
                             for the spin parameter range [0, 100].
            
            Returns:
                N/A
        """
        
        self.spin, self.lam, self.spinsqlam, self.fun1, self.fun2 = \
                       self._sample_laplacegrid(spinmin=spinmin,spinmax=spinmax,
                                                        spindensity=spindensity)
    
    
     
    
    def uniform_pattern(self,frot,Pi0,alpha_g=0.5,unit='cycle_per_day'):
        """
            Calculate the asymptotic period spacing pattern for a spherically 
            symmetric star with uniform rotation, following Eq.(4) in 
            Van Reeth et al. 2016, A&A, 593, A120.
            
            Parameters:
                self:     gravity_modes object
                frot:     astropy quantity (type: frequency)
                          the rotation frequency of the star
                Pi0:      astropy quantity (type: time)
                          the buoyancy radius of the star
                alpha_g:  float; optional
                          phase term, dependent on the boundary conditions of 
                          the g-mode cavity. Typically, alpha_g is 0.5 for a 
                          star with a convective core and a radiative envelope.
                          (default value = 0.5)
                unit:     string; optional
                          The preferred (astropy) unit of the calculated g-mode 
                          pattern. The options are 'days', 'seconds', 
                          'cycle_per_day', 'muHz', 'nHz', 'Hz'. 
                          (default = 'cycle_per_day').
            
            Returns:
                pattern:  astropy quantity array
                          theoretical asymptotic g-mode pattern (matching the 
                          preferred 'unit')
        """
        
        ### Verifying the input
        
        # Make sure that the rotation frequency and the buoyancy radius have 
        # appropriate units
        assert frot.unit.physical_type == 'frequency', \
               "The provided rotation frequency does not have a frequency unit \
                (as provided in astropy.units)."
        assert Pi0.unit.physical_type == 'time', \
               "The provided buoyancy radius does not have a time unit \
                (as provided in astropy.units)."
        
        # Can we handle the preferred output unit?
        allowed_out_units = {'days':u.day, 'seconds':u.s,
                                'cycle_per_day':u.day**-1., 'muHz':u.uHz,
                                'nHz':u.nHz, 'Hz':u.Hz}
        
        assert unit in allowed_out_units.keys(), \
               f"Please ensure that the requested output unit is one of: \
               {', '.join(str(x_unit) for x_unit in allowed_out_units.keys())}."
        astrounit = allowed_out_units[unit]
        
        ### Alright, start the calculation!
        
        # Safety check: the following routine only works if frot != 0
        if(frot != 0.*frot.unit):
            # Calculating the pattern -- in spin parameter
            basefun_lhs = 2.*float(frot*Pi0)*(self.nvals+alpha_g)
            basefun_rhs = self.spinsqlam
            selected_spin = np.interp(basefun_lhs, basefun_rhs, self.spin)
            
            # Setting the values which lie outside of the computable range to 
            # "nan"
            out_of_range = ~np.r_[(basefun_lhs >= np.amin(basefun_rhs)) &
                                  (basefun_lhs <= np.amax(basefun_rhs))]
            selected_spin[out_of_range] = np.nan
            
            # Converting to frequencies -- in the unit of frot
            puls_freq = (2.*frot/selected_spin) + (self.mval*frot)
        
        else:
            if(np.amin(self.spin) < 0.):
                # Pulsation frequencies -- in the unit of 1/Pi0
                puls_freq = np.sqrt(np.interp(0.,self.spin,self.lam)) \
                                                    / (Pi0*(self.nvals+alpha_g))
            else:
                # These are likely r-modes; not present in a non-rotating star!
                puls_freq = self.nvals*np.nan / Pi0.unit
            
        # Converting to the preferred output unit
        if(astrounit.physical_type == 'frequency'):
            pattern = puls_freq.to(astrounit)
        else:
            pattern = (1./puls_freq).to(astrounit)
        
        return pattern
        



    def centrifugal_pattern(self, frot, star, alpha_g=0.5, unit='cycle_per_day', 
                                                          npts=100, nthreads=8):
        """
            Calculate an asymptotic period spacing pattern using the generalised 
            TAR framework described by Mathis & Prat (2019) and Henneco et al. 
            (2021), i.e., including weak centrifugal acceleration.
            
            Parameters:
                self:     gravity_modes object
                frot:     astropy quantity (type: frequency)
                          the rotation frequency of the star
                star:     stellar_model object
                          the (non-rotating) stellar model
                alpha_g:  float; optional
                          phase term, dependent on the boundary conditions of 
                          the g-mode cavity. Typically, alpha_g is 0.5 for a 
                          star with a convective core and a radiative envelope.
                          (default value = 0.5)
                unit:     string; optional
                          The preferred (astropy) unit of the calculated g-mode 
                          pattern. The options are 'days', 'seconds', 
                          'cycle_per_day', 'muHz', 'nHz', 'Hz'. 
                          (default = 'cycle_per_day').
                npts:     integer; optional
                          the number of equidistantly spaced sampling points for
                          the colatitude theta (with values ranging from 0 to 
                          pi), at which the Brunt-Vaisala frequency profile and
                          centrifugal deformation are evaluated to calculate the
                          asymptotic period-spacing pattern. 
                          (default value = 100).
                nthreads: integer; optional
                          the number of threads used in the parallellised parts
                          of the calculation.
                          
            Returns:
                pattern:  astropy quantity array
                          theoretical asymptotic g-mode pattern (matching the 
                          preferred 'unit')
        """
            
        # Make sure that the rotation frequency has appropriate units
        assert frot.unit.physical_type == 'frequency', \
               "The provided rotation frequency does not have a frequency unit \
                (as provided in astropy.units)."
        
        # Can we handle the preferred output unit?
        allowed_out_units = {'days':u.day, 'seconds':u.s,
                                'cycle_per_day':u.day**-1., 'muHz':u.uHz,
                                'nHz':u.nHz, 'Hz':u.Hz}
        assert unit in allowed_out_units.keys(), \
               f"Please ensure that the requested output unit is one of: \
               {', '.join(str(x_unit) for x_unit in allowed_out_units.keys())}."
        astrounit = allowed_out_units[unit]
        
        ### Alright, start the calculation!
            
        # determine the angular rotation frequency
        if(frot.unit == u.rad/u.s):
            omrot = frot
        else:
            omrot = 2.*np.pi * frot.to(1./u.s) * u.rad
        
        # calculate the corresponding centrifugal deformation of the given 
        # stellar model 'star'
        star.calculate_centrifugal_deformation(omrot, nthreads=nthreads)
        
        if(omrot.value == 0.):
            Pi0 = star.Pi0()
            pattern = self.uniform_pattern(frot,Pi0, alpha_g=alpha_g, unit=unit)
            
        else:
            # Step 1: Calculating the period spacing pattern without the 
            #         centrifugal acceleration, and determine the corresponding 
            #         spin parameters and eigenvalues lambda (oversampled by a 
            #         factor 2 and with a buffer)
            Pi0 = star.Pi0()
            uni_patt = self.uniform_pattern(frot,Pi0, alpha_g=alpha_g)
            
            if(frot.unit == u.rad/u.s):
                resc_patt = uni_patt.to(1./u.s) * 2.*np.pi*u.rad
            else:
                resc_patt = uni_patt.to(frot.unit)
            
            patt_spin = 2. * frot / (resc_patt - (self.mval*frot))
            
            nmin = max(1, int(np.amin(self.nvals)) - 2)
            nmax = int(np.amax(self.nvals)) + 2
            Nsampling = 2
            uni_spin = np.interp(np.linspace(nmin,nmax,Nsampling*(nmax-nmin)+1),
                                                    self.nvals, patt_spin.value)
            uni_lmbda = np.interp(uni_spin, self.spin, self.lam)
            
            # Step 2: Calculate the Hough function(s) without the centrifugal 
            #         acceleration, to then calculate the weighted average 
            #         N/r-profile (with the centrifugal acceleration)
            hr_fun = [hough(nu, self.lval,self.mval, lmbd_est=lmbd,npts=npts)[2]
                                         for nu,lmbd in zip(uni_spin,uni_lmbda)]
            
            npts = np.array(hr_fun).shape[1]
            mu = np.cos(np.pi/npts * (np.arange(2*npts//2)+0.5))
            sinth = np.sqrt(1.-mu**2.)
            hough_weights = [np.abs(hr)*sinth/np.nansum(np.abs(hr)*sinth) 
                                                              for hr in hr_fun]
            
            N2_mu = np.array([star.centrifugal_N2profile(imu) for imu in mu])
            N2_mu[~np.isfinite(N2_mu) | (N2_mu < 0)] = 0.
            N_mu = np.sqrt(N2_mu)
            rad_mu = np.array([star.centrifugal_radius(imu) for imu in mu])
            Nr_per_spin = np.array(hough_weights) @ ( N_mu / rad_mu ) \
                                                         * star.radius[-1].value
            
            # Step 3: Calculate the centrifugal Lambda eigenvalues at sparse, 
            #         discrete radii
            rel_a_sparse = np.linspace(0.,1.,11)
            eps0_vals = star.eps_l0[-1] * rel_a_sparse**3.
            eps2_vals = star.eps_l2[-1] * rel_a_sparse**3.
            
            lmbd_sparse = []
            for ieps0,ieps2 in zip(eps0_vals,eps2_vals):
                lmbd_sparse.append([hough(nu,self.lval,self.mval,lmbd_est=lmbd,
                                          npts=npts, epsl0=ieps0, epsl2=ieps2, 
                                          only_lambda=True)
                                        for nu,lmbd in zip(uni_spin,uni_lmbda)])
            lmbd_sparse = np.array(lmbd_sparse).T
                
            # Step 4: dense interpolation, integration with N/r, and 
            # multiplication with spin
            Nr_times_spin = []
            for i_nu,i_lmbd,i_Nr in zip(uni_spin,lmbd_sparse,Nr_per_spin):
                par = np.polyfit(rel_a_sparse, i_lmbd, deg=3)
                lmbd_dense = np.polyval(par,star.radius/star.radius[-1])
                Nr_times_spin.append(i_nu*np.trapz(np.sqrt(lmbd_dense)*i_Nr,
                                                 x=star.radius/star.radius[-1]))
            # # # print(2.*omrot*np.pi*(self.nvals+alpha_g), Nr_times_spin)
            # Step 5: calculate the period-spacing pattern
            puls_spin = np.interp(2.*omrot*np.pi*(self.nvals+alpha_g), 
                                    np.array(Nr_times_spin)*u.rad/u.s, uni_spin)
            
            puls_freq = ( self.mval + 2./puls_spin ) * frot
            
            # Converting to the preferred output unit
            if(astrounit.physical_type == 'frequency'):
                pattern = puls_freq.to(astrounit)
            else:
                pattern = (1./puls_freq).to(astrounit)
               
        return pattern
        
        
        
    
    
    def differential_pattern_taylor(self, frot, Pi0, coeffs=[], alpha_g=0.5, 
                                                          unit='cycle_per_day'):
        """
            Calculates the asymptotic period spacing pattern for a star with 
            differential rotation in a Taylor expansion, following a corrected
            Eq.9 of Van Reeth et al. 2018, A&A, 618, A24.
            
            Parameters:
                self:     gravity_modes object
                frot:     astropy quantity (type: frequency)
                          the rotation frequency of the star
                Pi0:      astropy quantity (type: time)
                          the buoyancy radius of the star
                coeffs:   list
                          dimensionless coefficients for the Taylor expansion, 
                          for the arrays self.fun1 and self.fun2. 
                alpha_g:  float
                          phase term, dependent on the boundary conditions of 
                          the g-mode cavity. Typically, alpha_g = 1/2.
                unit:     string
                          The preferred (astropy) unit of the calculated g-mode 
                          pattern. The options are 'days', 'seconds', 
                          'cycle_per_day', 'muHz', 'nHz', 'Hz'. 
                          Default = 'cycle_per_day'.
            
            Returns:
                pattern:  astropy quantity arraymatching the preferred
                          'unit'.
        """
        
        ### Verifying the input
        
        # Make sure that the rotation frequency and the buoyancy radius have 
        # appropriate units
        assert frot.unit.physical_type == 'frequency', \
               "The provided rotation frequency does not have a frequency unit \
                (as provided in astropy.units)."
        assert Pi0.unit.physical_type == 'time', \
               "The provided buoyancy radius does not have a time unit \
                (as provided in astropy.units)."
        
        # Make sure that there are not too many input coefficients, and that 
        # they have suitable units
        assert len(coeffs) <= 2, \
               "The Taylor expansion for differential rotation has been \
                calculated up to the 2nd-order only."
        
        # Give everything the same base unit (=frot.unit), for convenience
        if(len(coeffs) < 1): 
            c1 = 0.
        else:
            c1 = coeffs[0]
        if(len(coeffs) < 2): 
            c2 = 0.
        else:
            c2 = coeffs[1]
        
        # Can we handle the preferred output unit?
        allowed_units = {'days':u.day,'seconds':u.s, 
                         'cycle_per_day':u.day**-1., 'muHz':u.uHz, 'nHz':u.nHz,
                         'Hz':u.Hz}
        assert unit in allowed_units.keys(), \
               f"Please ensure that the requested output unit is one of: \
                 {', '.join(str(x_unit) for x_unit in allowed_units.keys())}."
        astrounit = allowed_units[unit]
        
        
        ### Alright, start the calculation!
        
        # Safety check: the following routine only works if frot != 0
        if(frot != 0.*frot.unit):
            # Calculating the pattern -- in spin parameter
            base_lhs = 2. * float( frot * Pi0 ) * ( self.nvals + alpha_g )
            base_rhs = self.spinsqlam \
                       * ( 1. + c1 * self.spin * self.fun1 \
                           + c2 * self.spin**2 * self.fun2 )
            selected_spin = np.interp(base_lhs, base_rhs, self.spin)
            
            # Setting the values which lie out of range to "nan"
            out_of_range = ~np.r_[(base_lhs >= np.amin(base_rhs))
                                  & (base_lhs <= np.amax(base_rhs))]
            selected_spin[out_of_range] = np.nan
            
            # Converting to frequencies -- in the unit of frot
            puls_freq = (2.*frot/selected_spin) + (self.mval*frot)
        
        # if frot == 0, the Taylor expansion cannot be used to model 
        # differential rotation: everything reverts to the case of non-rotation.
        else:
            if(np.amin(self.spin) < 0.):
                # Pulsation frequencies -- in the unit of 1/Pi0
                puls_freq = np.sqrt(np.interp(0.,self.spin,self.lam)) \
                                                    / (Pi0*(self.nvals+alpha_g))
                
            else:
                # These are likely r-modes -- not present in a non-rotating star!
                puls_freq = self.nvals*np.nan
            
        ### Converting to the preferred output unit
        
        if(astrounit.physical_type == 'frequency'):
            pattern = puls_freq.to(astrounit)
            
        else:
            pattern = (1./puls_freq).to(astrounit)
        
        return pattern
    
 
        
    def differential_pattern_integral(self, star, frot_profile, alpha_g=0.5,
                                          unit='cycle_per_day', df_step = 0.01):
        """
            Calculates the asymptotic period spacing pattern for a star with 
            differential rotation given the rotation profile, following Eq.3 in
            Van Reeth et al. 2018, A&A, 618, A24.
            
            This calculation is performed iteratively, whereby the maximal 
            fraction of differential rotation (relative to the average rotation
            rate) is increased by a step df_step at every iteration, until the 
            input rotation profile frot_profile is reached.
            
            Parameters:
                self:         gravity_modes object
                star: 
                              
                frot_profile: array, dtype=astropy quantity (type: frequency)
                              the rotation frequency profile of the star
                alpha_g:      float; optional
                              phase term, dependent on the boundary conditions
                              of the g-mode cavity. (default value = 0.5).
                unit:         string; optional
                              The preferred (astropy) unit of the calculated 
                              g-mode pattern. The options are 'days', 'seconds', 
                              'cycle_per_day', 'muHz', 'nHz', 'Hz'. 
                              (default = 'cycle_per_day')
                df_step:      float; optional
                              step with which the relative differential rotation
                              rate of the rotation profile is increased at every
                              iteration (default value = 0.01)
            
            Returns:
                pattern:      array, dtype = astropy quantity
                              The calculated asymptotic period-spacing pattern.
        """
                
        # Verifying the quantities and converting to the easiest units
        assert frot_profile.unit.physical_type in ['angular speed','frequency'],\
               "Please ensure that the rotation frequency array has an angular \
                speed unit or a frequency unit (as given in astropy.units)."
        
        if(frot_profile.unit.physical_type == 'angular speed'):
            frot_prof = (frot_profile / (2.*np.pi*u.rad)).to(u.day**-1.)
        else:
            frot_prof = frot_profile.to(u.day**-1.)
        
        # Can we handle the preferred output unit?
        output_units = {'days':u.day, 'seconds':u.s, 'muHz':u.uHz, \
                             'cycle_per_day':u.day**-1., 'nHz':u.nHz, 'Hz':u.Hz}
        assert unit in output_units.keys(), \
               f"Please ensure that the requested output unit is one of: \
                 {', '.join(str(x_unit) for x_unit in output_units.keys())}."
        astrounit = output_units[unit]
        
        
        xrad = star.radius.value / star.radius.value[-1]
        rs = np.trapz(star.N, x=star.radius) / \
                                     np.trapz(star.N/star.radius, x=star.radius)
        frot_s = np.interp(rs, star.radius, frot_profile)
        
        df_max = ( np.amax(frot_profile) - frot_s ) / frot_s
        df_s = np.unique(np.hstack((np.arange(df_step, df_max, df_step), df_max)))
        puls_freq = self.uniform_pattern(frot_s, star.Pi0())
        out_of_range = np.array(np.zeros(len(puls_freq)), dtype=bool)
        
        for i_df in df_s:
            i_frot = frot_s + i_df * (frot_profile - frot_s) / df_max
            spin_profs = [2.*i_frot/(fr-self.mval*i_frot) for fr in puls_freq]
            
            nusql_profs = [np.interp(nu, self.spin, self.spinsqlam) 
                                                          for nu in spin_profs]
            
            shifted_radn = [float(np.trapz( nusql * star.N/(4.*np.pi**2.*u.rad)
                                            / (xrad*i_frot) ,x=xrad)) 
                                                      for nusql in nusql_profs]
            
            radn = np.array(shifted_radn) - alpha_g
            
            # Make sure that the results make sense...
            radn_valid_bool = np.isfinite(radn)
            ind_radial_sort = np.argsort(radn[radn_valid_bool])
            valid_freqs = puls_freq[radn_valid_bool][ind_radial_sort]
            valid_radn = radn[radn_valid_bool][ind_radial_sort]   
            
            if(len(valid_radn) >= 2):
                # Mapping the results onto the input radial orders self.nvals
                puls_freq = np.interp(self.nvals, valid_radn, valid_freqs)
                
                new_spin = [2.*i_frot/(fr-self.mval*i_frot) for fr in puls_freq]
            
                # Setting the values outside of the computable range to "nan"
                out_of_range1 = np.array([ (np.amax(i_spin.value) > self.spin[-2]) 
                                           | (np.amin(i_spin.value) < self.spin[1])  
                                                         for i_spin in spin_profs])
                out_of_range2 = np.array([ (np.amax(i_spin.value) > self.spin[-2]) 
                                           | (np.amin(i_spin.value) < self.spin[1])  
                                                           for i_spin in new_spin])
                out_of_range = np.r_[out_of_range | out_of_range1 | out_of_range2]
                puls_freq[out_of_range] = np.nan
                
            else:
                if(astrounit.physical_type == 'frequency'):
                    puls_freq = np.array([np.nan] * len(puls_freq)) * astrounit
                else:
                    puls_freq = np.array([np.nan] * len(puls_freq)) / astrounit
                
        # Converting to the preferred output unit
        if(astrounit.physical_type == 'frequency'):
            pattern = puls_freq.to(astrounit)
        else:
            pattern = (1./puls_freq).to(astrounit)
            
        return pattern
        
        
        
        
        
