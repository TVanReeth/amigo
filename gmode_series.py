import numpy as np
import astropy.units as u
import sys

import matplotlib.pyplot as plt

class asymptotic(object):
    """ 
        A python class to calculate period spacing patterns in the asymptotic regime using the TAR.
        Currently, it is possible to calculate g-mode patterns for uniform rotation and 
        radial differential rotation (using both a Taylor expansion and a given rotation profile.)
         
        author: Timothy Van Reeth (KU Leuven)
                timothy.vanreeth@kuleuven.be
    """
    
    def __init__(self,gyre_dir,kval=0,mval=1,nmin=1,nmax=150,allow_partial_modecavity=False):
        """
            Setting up the environment/asymptotic object to calculate g-mode period spacing patterns.
            
            Parameters:
                gyre_dir: string
                          The GYRE 5.x installation directory.
                kval:     integer
                          Part of the mode identification of the pulsation pattern. For gravito-inertial 
                          modes with an equivalent in a non-rotating star, k = l - |m|. Default value = 0.
                mval:     integer
                          Part of the mode identification of the pulsation pattern: azimuthal order. 
                          mval > 0 for prograde modes. Default value = 1.
                nmin:     integer
                          The minimum radial order that we will (attempt to) calulcate. Default value = 1.
                nmax:     integer
                          The maximum radial order that we will (attempt to) calulcate. Default value = 150.
                allow_partial_modecavity: boolean
                          Part of an experimental set-up for the radial differential rotation patterns
                          (with a rotation profile as input). If this is 'True', calculated r-modes no longer
                          have to propagate within the entire g-mode cavity. Not sure if this is what happens
                          in real stars or not. Default value = False (the conservative option).
        """
        
        self.kval = int(kval)
        self.mval = int(mval)
        self.nvals = np.arange(nmin,nmax+0.1,1.)
        
        self.lam_fun  = self._retrieve_laplacegrid(gyre_dir)
        self.spin, self.lam, self.spinsqlam, self.fun1, self.fun2 = self._sample_laplacegrid(allow_partial_modecavity=allow_partial_modecavity)
        
    
    def _retrieve_laplacegrid(self,gyre_dir):
        """
            Retrieving the function lambda(nu) given in GYRE v5.x.
            
            Parameters:
                self:     asymptotic object
                gyre_dir: string
                          The GYRE 5.x installation directory.
            
            Returns:
                lam_fun: function
                         A function to calculate lambda, given spin parameter values as input.
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
        import gyre_cheb_fit
        
        tf = gyre_tar_fit.TarFit.load(infile)
        lam_fun = np.vectorize(tf.lam)
        
        return lam_fun
    
    
    
    def _sample_laplacegrid(self,spinmax=1000.,spindensity=1.,allow_partial_modecavity=False):
        """
            Sampling the function lambda(nu) that was set up in _retrieve_laplacegrid().
            This subroutine includes a har-coded custom sampling density function for the spin parameter.
            
            Parameters:
                self:     asymptotic object
                spinmax:  float
                          The maximum spin parameter value for which lambda eigenvalues will be retrieved
                          (following the Laplace tidal equation). Default value = 1000.
                spindensity: float
                          A scaling factor for the sampling density function. The default value (=1) results
                          in 20000 data points for the spin parameter range [0, 100].
                allow_partial_modecavity: boolean
                          Part of an experimental set-up for the radial differential rotation patterns
                          (with a rotation profile as input). If this is 'True', calculated r-modes no longer
                          have to propagate within the entire g-mode cavity. Not sure if this is what happens
                          in real stars or not. If True, the returned spin values in the range
                          where r-modes do not exists, are set to 0 rather than Nan (for numerical reasons). 
                          Default value = False (the conservative option).

            Returns:
                spin:     numpy array, dtype=float
                          The spin parameter values for which lambda eigenvalues are returned
                lam:      numpy array, dtype=float
                          The lambda eigenvalues corresponding to the spin parameter values in the array 'spin'
                spinsqlam: numpy array, dtype=float
                          = spin * sqrt(lam)
                fun1      numpy array, dtype=float
                          The first of two functions required to calculate patterns for a differentially rotating star, using Taylor expansion.
                          Following the corrected versions of Eq.9 and Appendix B in the corrigendum of Van Reeth et al. 2018, A&A 618, A24.
                fun2      numpy array, dtype=float
                          The second of two functions required to calculate patterns for a differentially rotating star, using Taylor expansion.
                          Following the corrected versions of Eq.9 and Appendix B in the corrigendum of Van Reeth et al. 2018, A&A 618, A24.
        """
        
        if((self.kval >= 0) & (self.mval != 0)):
            spinmin = -0.1
            # Relatively "ad hoc" optimal sampling (based on "experience")
            nspinmin = round(spindensity * 20000. * (np.log10(1.+abs(spinmin))/np.log10(101.))**0.5)
            nspinmax = round(spindensity * 20000. * (np.log10(1.+abs(spinmax))/np.log10(101.))**0.5)
            if((spinmin < 0.) & (spinmax <= 0.)):
                spin = -10.**(np.linspace(np.log10(1.+abs(spinmin))**0.5,np.log10(1.+abs(spinmax))**0.5,int(nspinmin-nspinmax))**2.) + 1.
            elif((spinmin >= 0.) & (spinmax > 0.)):
                spin = 10.**(np.linspace(np.log10(1.+spinmin)**0.5,np.log10(1.+spinmax)**0.5,int(nspinmax-nspinmin))**2.) - 1.
            else:
                spinneg = -10.**(np.linspace(np.log10(1.+abs(spinmin))**0.5,0.,int(nspinmin))**2.) + 1.
                spinpos = 10.**(np.linspace(0.,np.log10(1.+spinmax)**0.5,int(nspinmax))**2.) - 1.
                spin = np.unique(np.hstack((spinneg,spinpos)))
        
        elif((self.kval >= 0) & (self.mval == 0)):
            spinmin = -spinmax
            # Relatively "ad hoc" optimal sampling (based on "experience")
            nspinmin = round(spindensity * 20000. * (np.log10(1.+abs(spinmin))/np.log10(101.))**0.5)
            nspinmax = round(spindensity * 20000. * (np.log10(1.+abs(spinmax))/np.log10(101.))**0.5)
            if((spinmin < 0.) & (spinmax <= 0.)):
                spin = -10.**(np.linspace(np.log10(1.+abs(spinmin))**0.5,np.log10(1.+abs(spinmax))**0.5,int(nspinmin-nspinmax))**2.) + 1.
            elif((spinmin >= 0.) & (spinmax > 0.)):
                spin = 10.**(np.linspace(np.log10(1.+spinmin)**0.5,np.log10(1.+spinmax)**0.5,int(nspinmax-nspinmin))**2.) - 1.
            else:
                spinneg = -10.**(np.linspace(np.log10(1.+abs(spinmin))**0.5,0.,int(nspinmin))**2.) + 1.
                spinpos = 10.**(np.linspace(0.,np.log10(1.+spinmax)**0.5,int(nspinmax))**2.) - 1.
                spin = np.unique(np.hstack((spinneg,spinpos)))
        else:
            spinmin = float((abs(self.mval) + abs(self.kval)) * (abs(self.mval) + abs(self.kval) - 1)) / abs(self.mval)
            # Relatively "ad hoc" optimal sampling (based on "experience")
            nspinmax = round(spindensity * 20000. * (np.log10(1.+abs(spinmax))/np.log10(101.))**0.5)
            
            spinpos = 10.**(np.linspace(np.log10(1.)**0.5,np.log10(1.+spinmax)**0.5,int(nspinmax))[1:]**2.) - 1. + spinmin
            spinneg = np.linspace(0.,spinmin,100)
            spin = np.unique(np.hstack((spinneg,spinpos)))
            
        # Obtaining the eigenvalues lambda
        lam = self.lam_fun(spin)
        
        # A small experimental setup to deal with stronger differential rotation... Turned off by default (allow_partial_modecavity = False).
        # If the differential rotation is so strong that the spin parameter values for certain inertial modes are out of the validity range,
        # it is assumed here that the inertial mode no longer propagates in that part of the star, but still exists in part of the mode cavity.
        # The buoyancy is the dominant force that radially couples the different shells where the mode propagates. BUT: if the Coriolis force in a certain shell
        # cannot support an inertial oscillation at a given frequency, the pulsation should not propagate there (at least, not at that frequency). At the moment, I
        # assume that such pulsations do NOT propagate in such layers at all. Not sure if this is correct...
        if(allow_partial_modecavity & (self.kval < 0)):
            lam_exists = np.isfinite(lam)
            spinstart = spin[lam_exists][0]
            lowspin = np.r_[spin < spinstart]
            lam[lowspin] = 0.
        
        # Limiting ourselves to the part of the arrays that exists...
        lam_exists = np.isfinite(lam)
        spin = spin[lam_exists]
        lam = lam[lam_exists]
        
        # Calculating the derivatives
        dlamdnu = np.gradient(lam)/np.gradient(spin)
        d2lamdnu2 = np.gradient(dlamdnu)/np.gradient(spin)
        
        # A required array to determine the near-core rotation rate
        spinsqlam = spin*np.sqrt(lam)
        
        # Array required to account for the first-order effects of differential near-core rotation
        fun1 = (dlamdnu*(1. + (self.mval*spin/2.))/lam) + self.mval
        
        # Array required to account for the second-order effects of differential near-core rotation
        fun2 = 0.5 * ((( (2.*d2lamdnu2/lam) - (dlamdnu/lam)**2. )*(1.+(self.mval*spin/2.))**2.) + (3.*self.mval*dlamdnu*(1. + (self.mval*spin/2.))/lam) + (2.*self.mval**2.) )
        
        return spin, lam, spinsqlam, fun1, fun2
    
    
    
    
    def update_laplacegrid(self,spinmax=10000.,spindensity=1.,allow_partial_modecavity=False):
        """
            Recalculating the spin aparameter range and the corresponding lambda eigenvalues included within the asymptotic object.
            
            Parameters:
                self:     asymptotic object
                spinmax:  float
                          The maximum spin parameter value for which lambda eigenvalues will be retrieved
                          (following the Laplace tidal equation). Default value = 1000.
                spindensity: float
                          A scaling factor for the sampling density function. The default value (=1) results
                          in 20000 data points for the spin parameter range [0, 100].
                allow_partial_modecavity: boolean
                          Part of an experimental set-up for the radial differential rotation patterns
                          (with a rotation profile as input). If this is 'True', calculated r-modes no longer
                          have to propagate within the entire g-mode cavity. Not sure if this is what happens
                          in real stars or not. If True, the returned spin values in the range
                          where r-modes do not exists, are set to 0 rather than Nan (for numerical reasons). 
                          Default value = False (the conservative option).
        """
        
        self.spin, self.lam, self.spinsqlam, self.fun1, self.fun2 = self._sample_laplacegrid(spinmax=spinmax,spindensity=spindensity,allow_partial_modecavity=allow_partial_modecavity)
    
    
     
    
    def uniform_pattern(self,frot,Pi0,alpha_g=0.5,unit='days'):
        """
            Calculates the asymptotic period spacing pattern for a star with uniform rotation, following Eq.4 in Van Reeth et al. 2016, A&A, 593, A120.
            
            Parameters:
                self:     asymptotic object
                frot:     astropy quantity (type: frequency)
                          the rotation frequency of the star
                Pi0:      astropy quantity (type: time)
                          the buoyancy radius of the star
                alpha_g:  float
                          phase term, dependent on the boundary conditions of the g-mode cavity. Typically, alpha_g = 1/2.
                unit:     string
                          The preferred (astropy) unit of the calculated g-mode pattern. The options are 'days', 'seconds', 
                          'cycle_per_day', 'muHz', 'nHz', 'Hz'. Default = 'days'.
            
            Output:
                pattern:  array, dtype = astropy quantity matching the preferred 'unit'.
        """
        
        ### Verifying the input
        
        # Make sure that the rotation frequency and the buoyancy radius have appropriate units
        assert frot.unit.physical_type == 'frequency', "The provided rotation frequency does not have a frequency unit (as provided in astropy.units)."
        assert Pi0.unit.physical_type == 'time', "The provided buoyancy radius does not have a time unit (as provided in astropy.units)."
        
        # Can we handle the preferred output unit?
        allowed_output_units = {'days':u.day,'seconds':u.s,'cycle_per_day':u.day**-1.,'muHz':u.uHz,'nHz':u.nHz,'Hz':u.Hz}
        assert unit in allowed_output_units.keys(), f"Please ensure that the requested output unit is one of: {', '.join(str(x_unit) for x_unit in allowed_output_units.keys())}."
        astrounit = allowed_output_units[unit]
        
        ### Alright, start the calculation!
        
        # Safety check: the following routine only works if frot != 0
        if(frot != 0.*frot.unit):
            # Calculating the pattern -- in spin parameter
            basefun_lhs = 2.*float(frot*Pi0)*(self.nvals+alpha_g)
            basefun_rhs = self.spinsqlam
            selected_spin = np.interp(basefun_lhs, basefun_rhs, self.spin)
            
            # Setting the values which lie outside of the computable range to "nan"
            out_of_range = ~np.r_[(basefun_lhs >= np.amin(basefun_rhs)) & (basefun_lhs <= np.amax(basefun_rhs))]
            selected_spin[out_of_range] = np.nan
            
            # Converting to frequencies -- in the unit of frot
            puls_freq = (2.*frot/selected_spin) + (self.mval*frot)
        
        # if frot == 0, the Taylor expansion cannot be used to model differential rotation: everything reverts to the case of non-rotation.
        else:
            if(np.amin(self.spin) < 0.):
                # Pulsation frequencies -- in the unit of 1/Pi0
                puls_freq = np.sqrt(np.interp(0.,self.spin,self.lam)) / (Pi0*(self.nvals+alpha_g))
            else:
                # These are likely r-modes -- not present in a non-rotating star!
                puls_freq = self.nvals*np.nan
            
        # Converting to the preferred output unit
        if(astrounit.physical_type == 'frequency'):
            pattern = puls_freq.to(astrounit)
        else:
            pattern = (1./puls_freq).to(astrounit)
        
        return pattern
    
    
    
    
    def differential_pattern_taylor(self,frot,Pi0,diffrot_coeffs=[],alpha_g=0.5,unit='days'):
        """
            Calculates the asymptotic period spacing pattern for a star with differential rotation in a Taylor expansion, 
            following the corrected Eq.9 and Appendix B in the corrigendum of Van Reeth et al. 2018, A&A, 618, A24.
            
            Parameters:
                self:     asymptotic object
                frot:     astropy quantity (type: frequency)
                          the rotation frequency of the star
                Pi0:      astropy quantity (type: time)
                          the buoyancy radius of the star
                diffrot_coeffs: list
                          coefficients for the Taylor expansion, for the arrays self.fun1 and self.fun2.
                alpha_g:  float
                          phase term, dependent on the boundary conditions of the g-mode cavity. Typically, alpha_g = 1/2.
                unit:     string
                          The preferred (astropy) unit of the calculated g-mode pattern. The options are 'days', 'seconds', 
                          'cycle_per_day', 'muHz', 'nHz', 'Hz'. Default = 'days'.
            
            Output:
                pattern:  array, dtype = astropy quantity matching the preferred 'unit'.
        """
        
      ### Verifying the input
        
        # Make sure that the rotation frequency and the buoyancy radius have appropriate units
        assert frot.unit.physical_type == 'frequency', "The provided rotation frequency does not have a frequency unit (as provided in astropy.units)."
        assert Pi0.unit.physical_type == 'time', "The provided buoyancy radius does not have a time unit (as provided in astropy.units)."
        
        # Make sure that there are not too many input coefficients, and that they have suitable units
        assert len(diffrot_coeffs) <= 2, "The Taylor expansion for differential rotation has been calculated up to the 2nd-order only."
        if(len(diffrot_coeffs) > 0):
            assert diffrot_coeffs[0].unit.physical_type == 'frequency', "The first provided diffrot_coeffs value does not have a frequency unit (as provided in astropy.units)."
        if(len(diffrot_coeffs) > 1):
            assert np.sqrt(diffrot_coeffs[1]).unit.physical_type == 'frequency', "The second provided diffrot_coeffs value does not have a frequency unit squared (as provided in astropy.units)."
        
        # Give everything the same base unit (=frot.unit), for convenience
        coeffs = diffrot_coeffs.copy()
        if(len(coeffs) < 1): 
            coeffs.append(0.*frot.unit)
        if(len(coeffs) < 2): 
            coeffs.append(0.*frot.unit**2.)
        coeffs[0].to(frot.unit)
        coeffs[1].to(frot.unit**2.)
        
        # Can we handle the preferred output unit?
        allowed_output_units = {'days':u.day,'seconds':u.s,'cycle_per_day':u.day**-1.,'muHz':u.uHz,'nHz':u.nHz,'Hz':u.Hz}
        assert unit in allowed_output_units.keys(), f"Please ensure that the requested output unit is one of: {', '.join(str(x_unit) for x_unit in allowed_output_units.keys())}."
        astrounit = allowed_output_units[unit]
        
      ### Alright, start the calculation!
        
        # Safety check: the following routine only works if frot != 0
        if(frot != 0.*frot.unit):
            # Calculating the pattern -- in spin parameter
            basefun_lhs = 2.*float(frot*Pi0)*(self.nvals+alpha_g)
            basefun_rhs = self.spinsqlam * (1. + float(coeffs[0]/(2.*frot))*self.spin*self.fun1 + float(coeffs[1]/(4.*frot**2.))*(self.spin**2.)*self.fun2)
            selected_spin = np.interp(basefun_lhs, basefun_rhs, self.spin)
            
            # Setting the values which lie outside of the computable range to "nan"
            out_of_range = ~np.r_[(basefun_lhs >= np.amin(basefun_rhs)) & (basefun_lhs <= np.amax(basefun_rhs))]
            selected_spin[out_of_range] = np.nan
            
            # Converting to frequencies -- in the unit of frot
            puls_freq = (2.*frot/selected_spin) + (self.mval*frot)
        
        # if frot == 0, the Taylor expansion cannot be used to model differential rotation: everything reverts to the case of non-rotation.
        else:
            if(np.amin(self.spin) < 0.):
                # Pulsation frequencies -- in the unit of 1/Pi0
                puls_freq = np.sqrt(np.interp(0.,self.spin,self.lam)) / (Pi0*(self.nvals+alpha_g))
            else:
                # These are likely r-modes -- not present in a non-rotating star!
                puls_freq = self.nvals*np.nan
            
        # Converting to the preferred output unit
        if(astrounit.physical_type == 'frequency'):
            pattern = puls_freq.to(astrounit)
        else:
            pattern = (1./puls_freq).to(astrounit)
        
        return pattern
    
    
    
    def differential_pattern_integral(self,rad_profile,frot_profile,brunt2_profile,Pi0,alpha_g=0.5,unit='days',computation_path='old'):
        """
            Calculates the asymptotic period spacing pattern for a star with differential rotation given the rotation profile, 
            following Eq.3 in Van Reeth et al. 2018, A&A, 618, A24.
            
            Parameters:
                self:     asymptotic object
                rad_profile: array, dtype=astropy quantity (type: length)
                          The (local) stellar radius
                frot_profile: array, dtype=astropy quantity (type: frequency)
                          the rotation frequency profile of the star
                brunt2_profile: array, dtype=astropy quantity (unit: rad2/s2)
                          the squared Brunt-Vaisala frequency (N^2) profile of the star
                Pi0:      astropy quantity (type: time)
                          the buoyancy radius of the star, matching the given brunt2_profile
                alpha_g:  float
                          phase term, dependent on the boundary conditions of the g-mode cavity. Typically, alpha_g = 1/2.
                unit:     string
                          The preferred (astropy) unit of the calculated g-mode pattern. The options are 'days', 'seconds', 
                          'cycle_per_day', 'muHz', 'nHz', 'Hz'. Default = 'days'.
                computation_path: string
                          can be 'old' or 'new'. This specifies which spin parameter range has to be considered in the computations.
                          When 'new' this range is wider, which can allow more modes to be computed, but is also slower. The 
                          difference is clearest in the case of moderately to strongly differential rotation. Default = 'old'.
            
            Output:
                pattern:  array, dtype = astropy quantity matching the preferred 'unit'.
        """
                
        # Verifying the quantities and converting to the easiest units
        assert frot_profile.unit.physical_type in ['angular speed','frequency'], "Please ensure that the rotation frequency array has an angular speed unit or a frequency unit (as given in astropy.units)."
        assert brunt2_profile.unit == "rad2 / s2", "Please ensure that the N^2 frequency array has 'rad2 / s2' as unit (as given in astropy.units)."
        assert rad_profile.unit.physical_type == 'length', "Please ensure that the stellar radius array has a length unit (as given in astropy.units)."
        assert computation_path in ['old','new'], "Please choose either the 'old' or 'new' computation_path. The old one is likely a bit faster, but more restrictive: this is better if the computation has to be repeated for a large grid of models."
        
        if(frot_profile.unit.physical_type == 'angular speed'):
            frot_prof = (frot_profile / (2.*n.pi*u.radian)).to(u.day**-1.)
        else:
            frot_prof = frot_profile.to(u.day**-1.)
        
        # Can we handle the preferred output unit?
        allowed_output_units = {'days':u.day,'seconds':u.s,'cycle_per_day':u.day**-1.,'muHz':u.uHz,'nHz':u.nHz,'Hz':u.Hz}
        assert unit in allowed_output_units.keys(), f"Please ensure that the requested output unit is one of: {', '.join(str(x_unit) for x_unit in allowed_output_units.keys())}."
        astrounit = allowed_output_units[unit]
        
        # Make sure that the radius array is monotonically increasing
        ind_sort = np.argsort(rad_profile)
        rad_sort = rad_profile[ind_sort]
        brunt2_sort = brunt2_profile[ind_sort]
        frot_sort = frot_prof[ind_sort]
        
        # Make sure that we only consider the radiative stellar regions in this subroutine...
        brunt2_sort[brunt2_sort < 0.] = 0.
        brunt_sort = np.sqrt(brunt2_sort)
        
        # ...and that we don't run into numerical issues
        domain = rad_sort > 0.
        
        if(computation_path == 'old'):

            # Computing the reference frequencies in the co-rotating frame at a coordinate rs
            if(self.mval > 0):
                evaluated = np.r_[brunt_sort > 0.]
                ind_max_rot = np.argmax(frot_sort[evaluated])
                rs = 0.5*(np.trapz(brunt_sort[domain], x=rad_sort[domain]) / np.trapz(brunt_sort[domain]/rad_sort[domain], x=rad_sort[domain]) + rad_sort[evaluated][ind_max_rot])
            elif(self.mval == 0):
                rs = np.trapz(brunt_sort[domain], x=rad_sort[domain]) / np.trapz(brunt_sort[domain]/rad_sort[domain], x=rad_sort[domain])
            else:
                evaluated = np.r_[brunt_sort > 0.]
                ind_min_rot = np.argmin(frot_sort[evaluated])
                rs = rad_sort[evaluated][ind_min_rot]
            
            frots = np.interp(rs, rad_sort, frot_sort.value) * frot_sort.unit
            freqs = self.uniform_pattern(frots,Pi0, unit='cycle_per_day')
        
        else:
            freqs = (2.*frot_sort[0] / self.spin) + self.mval*frot_sort[0]
        
        # Estimate the appropriate boundary values for the spin parameter within the differentially rotating star
        spin_min_bnd = np.array(2.*np.amin(frot_sort)/(freqs - self.mval*np.amin(frot_sort)), dtype=float)
        spin_max_bnd = np.array(2.*np.amax(frot_sort)/(freqs - self.mval*np.amax(frot_sort)), dtype=float)
        
        computable = np.r_[(spin_min_bnd >= self.spin[0]) & (spin_max_bnd >= self.spin[0]) & (spin_min_bnd <= self.spin[-1]) & (spin_max_bnd <= self.spin[-1])]
        
        if(computable.any()):
            computable_freqs = freqs[computable]
            
            # Compute the radial orders from the RHS of Eq.(3) -- this will be slower if 'computation_path = 'new'.
            shifted_radn = np.array([float(np.trapz(np.interp(2.*frot_sort[domain]/(freq-self.mval*frot_sort[domain]), self.spin, self.spinsqlam) * brunt_sort[domain]/(rad_sort[domain]*2.*frot_sort[domain]*2.*np.pi**2.*u.radian),x=rad_sort[domain])) for freq in computable_freqs])
            radial_n = shifted_radn - alpha_g
            
            # Make sure that the results make sense...
            valid_radial_n = np.isfinite(radial_n)
            ind_radial_sort = np.argsort(radial_n[valid_radial_n])
            freqs_valid_sort = computable_freqs[valid_radial_n][ind_radial_sort]
            radial_valid_sort = radial_n[valid_radial_n][ind_radial_sort]   
            
            # Mapping the results onto the input radial orders self.nvals
            puls_freq = np.interp(self.nvals,radial_valid_sort,freqs_valid_sort.value) * freqs_valid_sort.unit
            
            # Setting the values which lie outside of the computable range to "nan"
            out_of_range = ~np.r_[(self.nvals >= np.amin(radial_valid_sort)) & (self.nvals <= np.amax(radial_valid_sort))]
            puls_freq[out_of_range] = np.nan
        else:
            puls_freq = self.nvals*np.nan * freqs.unit
            
        # Converting to the preferred output unit
        if(astrounit.physical_type == 'frequency'):
            pattern = puls_freq.to(astrounit)
        else:
            pattern = (1./puls_freq).to(astrounit)
            
        return pattern
        

    
