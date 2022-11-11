import numpy as np
import astropy.units as u
import sys


class asymptotic(object):
    """ 
        A python class to calculate period spacing patterns in the asymptotic regime using the TAR.
        Currently, it is possible to calculate g-mode patterns for uniform rotation and 
        radial differential rotation (using both a Taylor expansion and a given rotation profile.)
         
        author: Timothy Van Reeth (KU Leuven)
                timothy.vanreeth@kuleuven.be
        
        License: GPL-3+
    """
    
    def __init__(self,gyre_dir,kval=0,mval=1,nmin=1,nmax=150):
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
        """
        
        self.kval = int(kval)
        self.mval = int(mval)
        self.nvals = np.arange(nmin,nmax+0.1,1.)
        
        self.lam_fun  = self._retrieve_laplacegrid(gyre_dir)
        self.spin, self.lam, self.spinsqlam = self._sample_laplacegrid()
        
        
    
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
       # import gyre_cheb_fit
        
        tf = gyre_tar_fit.TarFit.load(infile)
        lam_fun = np.vectorize(tf.lam)
        
        return lam_fun
    
    
    
    def _sample_laplacegrid(self,spinmax=1000.,spindensity=1.):
        """
            Sampling the function lambda(nu) that was set up in _retrieve_laplacegrid().
            This subroutine includes a har-coded custom sampling density function for the spin parameter.
            
            Parameters:
                self:        asymptotic object
                spinmax:     float
                             The maximum spin parameter value for which lambda eigenvalues will be retrieved
                             (following the Laplace tidal equation). Default value = 1000.
                spindensity: float
                             A scaling factor for the sampling density function. The default value (=1) results
                             in 20000 data points for the spin parameter range [0, 100].

            Returns:
                spin:      numpy array, dtype=float
                           The spin parameter values for which lambda eigenvalues are returned
                lam:       numpy array, dtype=float
                           The lambda eigenvalues corresponding to the spin parameter values in the array 'spin'
                spinsqlam: numpy array, dtype=float
                           = spin * sqrt(lam)
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
        
        # Limiting ourselves to the part of the arrays that exists...
        lam_exists = np.isfinite(lam)
        spin = spin[lam_exists]
        lam = lam[lam_exists]
        
        # A required array to determine the near-core rotation rate
        spinsqlam = spin*np.sqrt(lam)
        
        return spin, lam, spinsqlam
    
    
    
    def update_laplacegrid(self,spinmax=10000.,spindensity=1.):
        """
            Recalculating the spin aparameter range and the corresponding lambda eigenvalues included within the asymptotic object.
            
            Parameters:
                self:        asymptotic object
                spinmax:     float
                             The maximum spin parameter value for which lambda eigenvalues will be retrieved
                             (following the Laplace tidal equation). Default value = 1000.
                spindensity: float
                             A scaling factor for the sampling density function. The default value (=1) results
                             in 20000 data points for the spin parameter range [0, 100].
        """
        
        self.spin, self.lam, self.spinsqlam = self._sample_laplacegrid(spinmax=spinmax,spindensity=spindensity)
    
    
     
    
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
        
        else:
            if(np.amin(self.spin) < 0.):
                # Pulsation frequencies -- in the unit of 1/Pi0
                puls_freq = np.sqrt(np.interp(0.,self.spin,self.lam)) / (Pi0*(self.nvals+alpha_g))
            else:
                # These are likely r-modes -- not present in a non-rotating star!
                puls_freq = self.nvals*np.nan / Pi0.unit
            
        # Converting to the preferred output unit
        if(astrounit.physical_type == 'frequency'):
            pattern = puls_freq.to(astrounit)
        else:
            pattern = (1./puls_freq).to(astrounit)
        
        return pattern
