#!/usr/bin/env python3
#
# File: stellar_model.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: a class and subroutines to store and calculate different stellar
#              structure profiles and (observable) quantities, required for 
#              g-mode asteroseismology.

import numpy as np
import astropy.units as u
import sys

from scipy.integrate import cumtrapz
from scipy.integrate import solve_ivp
from multiprocessing import Pool
from itertools import product


class stellar_model(object):
    
    """
        A class to to store and calculate different stellar structure profiles 
        and (observable) quantities, required for g-mode asteroseismology.
        
        The algorithms in this module are described in:
            Van Reeth et al. 2016, A&A 593, A120
            Van Reeth et al. 2018, A&A 618, A24
            Mathis & Prat 2019, A&A 631, A26
            Henneco et al. 2021, A&A 648, A97
    """
    
    def __init__(self, radius, brunt2, mass=None, pressure=None, density=None, 
                                                                   Gamma1=None):
        """
            Setting up a stellar model object.
            
            Parameters:
                radius:    astropy quantity array (type=length)
                           the stellar radial coordinates
                brunt2:    astropy quantity array (with unit rad2/s2)
                           the squared Brunt-Vaisala N^2 profile
                mass:      astropy quantity array (type=mass); optional
                           the corresponding stellar mass coordinates
                           (default = None)
                pressure:  astropy quantity array (type=pressure); optional
                           the stellar pressure profile. (default = None)
                density:   astropy quantity array (type=density); optional
                           the stellar density profile. (default = None)
                Gamma1:    numpy array; optional
                           the stellar Gamma1 profile. (default = None)
        """
        
        # Verifying the quantities and converting to the easiest units
        assert brunt2.unit == "rad2 / s2", \
               "Please ensure that the N^2 frequency array has 'rad2 / s2' as \
                unit (as given in astropy.units)."
        assert radius.unit.physical_type == 'length', \
               "Please ensure that the stellar radius array has a length unit \
                (as given in astropy.units)."
        
        # Make sure that the radius array is monotonically increasing
        ind_sort = np.argsort(radius)
        rad_sort = radius[ind_sort]
        brunt2_sort = brunt2[ind_sort]
        
        # The square root of N^2
        brunt2_sort[(brunt2_sort < 0.) | ~np.isfinite(brunt2_sort)] = 0.
        brunt_sort = np.sqrt(brunt2_sort)
        
        self._radius = rad_sort
        self._brunt2 = brunt2_sort
        self._brunt  = brunt_sort
        
        self._mass   = mass
        self._pressure = pressure
        self._density = density
        self._Gamma1 = Gamma1
        
        # Avoiding unnecessary nans
        minr = 0.5*rad_sort[1]
        self._brunt2[0] = np.interp(minr, self._radius, self._brunt2)
        self._brunt[0] = np.interp(minr, self._radius, self._brunt)
        if self._mass is not None:
            self._mass[0] = np.interp(minr, self._radius, self._mass)
        if self._pressure is not None:
            self._pressure[0] = np.interp(minr, self._radius, self._pressure)
        if self._density is not None:
            self._density[0] = np.interp(minr, self._radius, self._density)
        if self._Gamma1 is not None:
            self._Gamma1[0] = np.interp(minr, self._radius, self._Gamma1)
        self._radius[0] = minr
        
        # Setting the other necessary quantities for the centrifugal deformation
        self._omrot_centrifugal = None
        self._phi_l0 = None
        self._phi_l2 = None
        self._U_l0   = None
        self._U_l2   = None
        self._eps_l0 = None
        self._eps_l2 = None
        
        self._rho_l0 = None
        self._rho_l2 = None
        
        self._P_l0   = None
        self._P_l2   = None
        
        self._mu_grid = None
        self._centrifugal_N2_grid = None
        
    
        
    ### stellar_model properties - part 1: the non-rotating model
    ### Note: the properties are listed alphabetically.
    
    @property
    def drhodr(self):
        """
            the radial density gradient profile (astropy quantity)
        """
        if(self.rho is None):
            return None
        else:
            mrad = 0.5*(self.radius[1:]+self.radius[:-1])
            mdrhodr = np.diff(self.rho)/np.diff(self.radius)
            drhodr = np.interp(self.radius,mrad,mdrhodr)
            return drhodr
    
    
    @property
    def g(self):
        """
            Gravitational acceleration g of the equilibrium (non-rotating) 
            stellar model.
            
            Parameters:
                 N/A
           
            Returns:
                grav:   astropy quantity (unit = (cm/s)^2)
                        gravitational acceleration of the non-rotating stellar
                        model            
        """
        
        G = 6.6743 * 10.**(-8.) * u.cm**3. / (u.g * u.s**2.)
        grav = G * self.mass.to(u.g) / self.radius.to(u.cm)**2.
        
        return grav
    
    
    @property
    def Gamma1(self):
        """
            the adiabatic coefficient profile (numpy array)
        """
        return self._Gamma1
    
    
    @property
    def mass(self):
        """
            the mass coordinates of the stellar profile (astropy quantity)
        """
        return self._mass

    
    @property
    def N(self):
        """
            the Brunt-Vaisala frequency profile N (astropy quantity)
        """
        return self._brunt
    
    
    @property
    def N2(self):
        """
            the squared Brunt-Vaisala frequency profile N^2 (astropy quantity)
        """
        return self._brunt2
    
    
    @property
    def omegacrit_roche(self):
        """
            Calculate the local critical rotation rate from a non-rotating 
            stellar model, given a mass and (corresponding) radial coordinate, 
            in cgs. Here, we follow the "Roche model definition", as given by 
            Bouabid et al. (2013), i.e., the stellar radius input values are
            those of the non-rotating model, scaled with a factor 3/2.
            
            Parameters:
                 N/A
           
            Returns:
                omk:    astropy quantity
                        the critical ("Roche") rotation rate (in rad/s) of the
                        given stellar model
        """
        
        G = 6.6743 * 10.**(-8.)
        mass_g = self.mass.to(u.g).value
        rad_cm = self.radius.to(u.cm).value
        omk = np.sqrt(8.*G*mass_g/(27.*np.amax(rad_cm)**3.)) * u.rad/u.s
        
        return omk
    

    @property
    def omegacrit_scale(self):
        """
            Calculate the local "critical" scaling rotation rate from a 
            non-rotating stellar model, given a mass and (corresponding) radial
            coordinate, in cgs. Here, we follow the "Keplerian definition", but
            as given by Mathis & Prat (2019), i.e., the stellar radius input 
            values are those of the non-rotating model.
            
            Parameters:
                 N/A
           
            Returns:
                omk:    astropy quantity
                        the critical ("Keplerian") rotation rate (in rad/s) of
                        the given stellar model
        """
        
        G = 6.6743 * 10.**(-8.)
        mass_g = self.mass.to(u.g).value
        rad_cm = self.radius.to(u.cm).value
        omk = np.sqrt( G*mass_g / (np.amax(rad_cm)**3.)) * u.rad/u.s

        return omk 
    
    
    @property
    def P(self):
        """
            the pressure profile (astropy quantity)
        """
        return self._pressure
    
    
    @property
    def radius(self):
        """
            the radial coordinates of the stellar profile (astropy quantity)
        """
        return self._radius


    @property
    def phi(self):
        """
            Gravitational potential phi of the equilibrium (non-rotating) 
            stellar model, i.e., the quantity phi_0 in Henneco et al. (2021).
            
            Parameters:
                 N/A
           
            Returns:
                phi0:   astropy quantity (unit = (cm/s)^2)
                        gravitational potential phi of the non-rotating stellar
                        model            
        """
        
        G = 6.6743 * 10.**(-8.) * u.cm**3. / (u.g * u.s**2.)
        phi0 = -G * self.mass.to(u.g) / self.radius.to(u.cm)
        
        return phi0
    
    
    @property
    def rho(self):
        """
            the density profile (astropy quantity)
        """
        return self._density
        
    
        
    ### stellar_model properties - part 2: the centrifugal deformation
    ### Note: the properties are listed alphabetically.
    
    @property
    def centrifugal_omrot(self):
        """
            the angular rotation frequency that was last used
            to calculate the centrifugal deformation of the stellar model 
            (astropy quantity)
        """
        return self._omrot_centrifugal
    
    
    @property
    def eps_l0(self):
        """
            the l=0 component of the centrifugal deformation of the non-rotating
            stellar model (astropy quantity)
        """
        return self._eps_l0
    
    
    @property
    def eps_l2(self):
        """
            the l=2 component of the centrifugal deformation of the non-rotating
            stellar model (astropy quantity)
        """
        return self._eps_l2
    
    
    @property
    def P_l0(self):
        """
            the l=0 component of the centrifugal perturbation of the 
            pressure of the non-rotating stellar model (astropy quantity)
        """
        return self._P_l0
    
    
    @property
    def P_l2(self):
        """
            the l=2 component of the centrifugal perturbation of the 
            density of the non-rotating stellar model (astropy quantity)
        """
        return self._P_l2
    
    
    @property
    def phi_l0(self):
        """
            the l=0 component of the centrifugal perturbation of the 
            graviational potential of the non-rotating stellar model
            (astropy quantity)
        """
        return self._phi_l0
    
    
    @property
    def phi_l2(self):
        """
            the l=2 component of the centrifugal perturbation of the 
            graviational potential of the non-rotating stellar model
            (astropy quantity)
        """
        return self._phi_l2
    
    
    @property
    def rho_l0(self):
        """
            the l=0 component of the centrifugal perturbation of the 
            density of the non-rotating stellar model (astropy quantity)
        """
        return self._rho_l0
    
    
    @property
    def rho_l2(self):
        """
            the l=2 component of the centrifugal perturbation of the 
            density of the non-rotating stellar model (astropy quantity)
        """
        return self._rho_l2
    
    
    @property
    def U_l0(self):
        """
            the l=0 component of the centrifugal potential, calculated based on 
            the non-rotating stellar model (astropy quantity)
        """
        return self._U_l0
    
    
    @property
    def U_l2(self):
        """
            the l=2 component of the centrifugal potential, calculated based on 
            the non-rotating stellar model (astropy quantity)
        """
        return self._U_l2
    
    
        
    ### stellar_model subroutines - part 1: the non-rotating model
    ### Note: the subroutines are listed alphabetically.
        
    def Pi0(self,unit='days'):
        """
            Calculates the buoyancy radius Pi0 from a N^2 profile as a function
            of stellar radius.
            
            Parameters:
                self:         a stellar model object
                unit:         string
                              the preferred unit of the (output) Pi0. Options
                              are 'days' and 'seconds'. Default = 'days'.
            
            Returns:
                buoyancy_rad: astropy quantity (unit = day or seconds)
                              the buoyancy radius    
        """
        
        # Can we handle the preferred output unit?
        allowed_output_units = {'days':u.day,'seconds':u.s}
        
        assert unit in allowed_output_units.keys(), \
               f"Please ensure that the requested output unit is one of: \
            {', '.join(str(x_unit) for x_unit in allowed_output_units.keys())}."
        
        astrounit = allowed_output_units[unit]
        
        # Computing the buoyancy radius
        integrand = self.N/self.radius
        N_integrated = np.trapz(integrand, x=self.radius)
        buoyancy_rad = 2.*np.pi**2. * u.radian / N_integrated
        buoyancy_rad = buoyancy_rad.to(astrounit)
        
        return buoyancy_rad
    
    
        
    ### stellar_model subroutines - part 2: the centrifugal deformation
    ### Note: the subroutines are listed alphabetically.

    def calculate_centrifugal_deformation(self, omrot, full_mapping=False, 
                                          ode_method='BDF', rel_tol=0.0001,
                                          maxiter=10, nsample=501, nthreads=6):
        """
            Calculate the effects of weak centrifugal acceleration on the
            stellar structure, assuming a rotation frequency onrot, starting 
            from a non-rotating stellar model.
            
            The results from this subroutine are not given as output, but stored
            in the stellar_model object.
            
            The framework that was used for these calculations, is described in:
                Mathis & Prat 2019, A&A 631, A26
                Henneco et al. 2021, A&A 648, A97
            
            Parameters:
                self:         a stellar_model object
                omrot:        astropy quantity (unit = rad/s)
                              the angular rotation frequency
                full_mapping: boolean; optional
                              indicate whether only the radial coordinates are 
                              mapped onto a 3rd-order polynomial (as described 
                              by Henneco et al. 2021; full_mapping = False), or
                              if this also has to be done for the centrifugal
                              perturbation of the gravitational potential
                              (full_mapping = True). This propagates into the
                              calculation of the centrifugally perturbed 
                              Brunt-Vaisala frequency profile. (default = False)
                ode_method:   string; optional
                              the numerical method that has to be used to solve
                              the ODE. This will be passed to the 'solve_ivp'
                              subroutine in scipy, and we refer to the 
                              documentation of the scipy python package for more
                              information. (default = 'BDF')
                rel_tol:      float; optional
                              the maximum relative error tolerance on the found 
                              solution(s) of the ODE (in the subroutine 
                              self._solve_ODE()), which is solved with an 
                              iterative shooting scheme. (default = 0.0001).
                maxiter:      integer; optional
                              the maximum number of iterations allowed to 
                              optimize the variables in the ODE (in the 
                              subroutine self._solve_ODE()). (default = 10).
                nsample:      integer; optional
                              the number considered values for the variables in
                              the ODE (default=501).
                nthreads:     integer; optional
                              the number of threads used in the parallellization
                              of the calculations, where possible (default = 6).
            
            Returns:
                N/A
        """
        
        phi_l0 = self.centrifugal_phi_l(0, omrot, ode_method=ode_method, 
                                        rel_tol=rel_tol, maxiter=maxiter, 
                                        nsample=nsample, nthreads=nthreads)
        phi_l2 = self.centrifugal_phi_l(2, omrot, ode_method=ode_method, 
                                        rel_tol=rel_tol, maxiter=maxiter, 
                                        nsample=nsample, nthreads=nthreads)
        U_l0_profile = self.centrifugal_potential_l(0, omrot)
        U_l2_profile = self.centrifugal_potential_l(2, omrot)
        
        self._omrot_centrifugal = omrot
        self._U_l0 = U_l0_profile
        self._U_l2 = U_l2_profile
        
        if(full_mapping):
            self._phi_l0 = phi_l0[-1] * ( self.radius / self.radius[-1] )**3.
            self._phi_l2 = phi_l2[-1] * ( self.radius / self.radius[-1] )**3.
            
        else:
            self._phi_l0 = phi_l0
            self._phi_l2 = phi_l2
        
        G = 6.6743 * 10.**(-8.) * u.cm**3. / (u.g * u.s**2.)
        self._eps_l0 = -(self._U_l0[-1]+self._phi_l0[-1]) * self.radius**3.\
                                            / (self.g[-1] * self.radius[-1]**4.)
        self._eps_l2 = -(self._U_l2[-1]+self._phi_l2[-1]) * self.radius**3.\
                                            / (self.g[-1] * self.radius[-1]**4.)
        
        self._rho_l0 = -self._eps_l0 * self.drhodr * self.radius
        self._rho_l2 = -self._eps_l2 * self.drhodr * self.radius
        
        self._P_l0 = -(self._U_l0 + self._phi_l0) * self.rho
        self._P_l2 = -(self._U_l2 + self._phi_l2) * self.rho
        
        self._pre_calculate_centrifugal_N2grid()
        
        return
    
    
    
    def centrifugal_N2profile(self, mu):
        """
            determine the brunt-Vaisala frequency profile N^2 in the 
            centrifugally deformed star, for a given value of mu = cos(theta),
            with theta being the co-latitude.
            If a grid of N^2-profiles has been pre-calculated (via the 
            calculate_centrifugal_deformation() subroutine), the requested 
            N^2-profile will be obtained via interpolation. Otherwise, the 
            requested profile will be custom calculated (and most likely simply 
            be the N^2-profile of the non-rotating model). The calculations are
            done as explained in Appendix A of Henneco et al. (2021).
            
            Parameters:
                self:           a stellar_model object
                mu:             float
                                the given value of cos(theta).
            
            Returns:
                centrifugal_N2: astropy quantity array
                                the N^2-profile in the centrifugally deformed 
                                star, for the given value of mu = cos(theta).
        """
        
        if(self._centrifugal_N2_grid is None):
            centrifugal_N2 = self._pre_calculate_centrifugal_N2profile(mu)
        
        else:
            len_mu = len(self._mu_grid)
            index_mu = np.interp(mu**2., self._mu_grid**2., np.arange(len_mu))
            index_min = int(np.floor(index_mu))
            index_max = int(np.ceil(index_mu))
            weight = 1. - (index_mu - float(index_min))
            
            centrifugal_N2 = weight * self._centrifugal_N2_grid[index_min] + \
                             (1 - weight) * self._centrifugal_N2_grid[index_max]
            
        return centrifugal_N2
    
    
    
    def centrifugal_phi_l(self, lval, omrot, ode_method='BDF', rel_tol=0.0001,
                         maxiter=10, nsample=501, nthreads=6, full_output=False,
                         dphidrmax0=5.*10.**5., dphidrmax2=30.):
        """
            Calculate the perturbations on the gravitational potential caused by
            a weak centrifugal acceleration, assuming a given rotation frequency
            omrot, starting from a non-rotating stellar model.
            
            This is done by rewriting the ODE in Eq.(A.17) in Henneco et al. 
            (2021) into a set of linear ODEs with inner and outer bounary 
            conditions. These are subsequently solved by trying a range of 
            different possible solutions at the stellar center in an iterative
            shooting scheme, and then solving the ODEs using the 
            scipy.integrate.solve_ivp subroutine.
            
            Parameters:
                self:        a stellar_model object
                lval:        integer
                             the degree of the Legendre polynomial component for 
                             which the centrifugal perturbation on the 
                             gravitational potential is calculated.
                omrot:       astropy quantity (unit = rad/s)
                             the angular rotation frequency of the centrifugally 
                             deformed star
                ode_method:  string; optional
                             the numerical method that has to be used to solve
                             the ODE. This will be passed to the 'solve_ivp'
                             subroutine in scipy, and we refer to the 
                             documentation of the scipy python package for more
                             information. (default = 'BDF')
                rel_tol:     float; optional
                             the maximum relative error tolerance on the found 
                             solution(s) of the ODE (in the subroutine 
                             self._solve_ODE()), which is solved with an 
                             iterative shooting scheme. (default = 0.0001).
                maxiter:     integer; optional
                             the maximum number of iterations allowed to 
                             optimize the variables in the ODE (in the 
                             subroutine self._solve_ODE()). (default = 10).
                nsample:     integer; optional
                             the number considered values for the variables in
                             the ODE (default=501).
                nthreads:    integer; optional
                             the number of threads used in the parallellization
                             of the calculations, where possible (default = 6).
                full_output: boolean; optional
                             indicate if only the best solution has to be 
                             returned (False) or the full output of the ODE,
                             with corresponding residuals (True). 
                             (default = False).
                dphidrmax0:  float; optional
                             the maximal considered value of the derivative 
                             dphi/dr in the steller center for l=0. The assumed
                             units are cm / s^2. (default = 5.*10.**5.).
                dphidrmax2:  float; optional
                             the maximal considered value of the derivative 
                             dphi/dr in the steller center for l=2. The assumed
                             units are cm / s^2. (default = 30.).
            
            Returns:
                phi_l:       astropy quantity array (unit = cm^2 / s^2)
                             the found perturbation component profile on the 
                             gravitational potential Phi, corresponding to
                             the given degree l.
                sol:         bunch object
                             the solution returned by the 
                             scipy.integrate.solve_ivp subroutine, used to solve
                             the ODE.
                dphidrs:     numpy array
                             the values of the derivative dphi/dr in the stellar
                             center that were evaluated in the iterative 
                             shooting scheme to solve the ODE.
                resid:       numpy array
                             the residuals for the outer boundary condition 
                             (i.e., at r = R), for the different evaluated 
                             values of dphi/dr.
        """
        
        assert (lval == 0) | (lval == 2), \
               "amigo.stellar_model.centrifugal_phi_l: error: please ensure \
                that the given degree l has a value of 0 or 2."
               
        assert omrot.unit == 'rad / s', \
               "amigo.stellar_model.centrifugal_phi_l: error: please ensure \
                that the angular rotation frequency has 'rad/s' as unit (as \
                given in astropy.units)."
        

        # Preparing required terms for the computations
        dphidrs = []
        resid = []
        dphidrmin = 0.             # HARD-CODED! assumed units = cm / s^2

        if(lval == 0.):
            dphidrmax = dphidrmax0   
            
        elif(lval == 2.):
            dphidrmax = dphidrmax2
        
        Niter = 1
        ppool = Pool(processes=nthreads)
        
        while((np.r_[np.array(resid) > rel_tol].all()) & (Niter <= maxiter)): 
            Niter += 1
            dphidr_guess = np.linspace(dphidrmin,dphidrmax,nsample)
            param_product = product(dphidr_guess,[lval],[omrot],[ode_method])
            resid_guess = ppool.starmap(self._solve_ode, param_product)
            dphidrs = dphidrs + list(dphidr_guess)
            resid = resid + list(resid_guess)

            if((np.argmin(resid_guess) == len(resid_guess)-1) | \
                                                 (np.argmin(resid_guess) == 0)):
                dphidr_ran = dphidrmax-dphidrmin
                
            else:
                dphidr_ran = 0.5*(dphidrmax-dphidrmin)
        
            dphidrmin = dphidr_guess[np.argmin(resid_guess)] - 0.5*dphidr_ran
            dphidrmax = dphidr_guess[np.argmin(resid_guess)] + 0.5*dphidr_ran
        
        dphidrs = np.array(dphidrs)
        resid = np.array(resid)
        dphidr = dphidrs[np.argmin(resid)]
        sol = self._solve_ode(dphidr, lval, omrot, ode_method, output_sol=True)
        
        phi_l = np.array(sol.y[0]) * u.cm**2. / u.s**2.
        
        if(full_output):
            return phi_l, sol, dphidrs, resid
            
        else:
            return phi_l
    
    

    def centrifugal_potential_l(self,lval,omrot):
        """
            Calculate the direct contribution of the centrifugal acceleration to
            the gravitational potential, expanded in terms of the Legendre 
            polynomials with l=0 and l=2, following Eqs.(A.14) and (A.15) in 
            Henneco et al. (2021).
        
            Parameters:
                self:  a stellar model object
                       the non-rotating stellar model for which we calculate the 
                       centrifugal potential.
                lval:  integer
                       the degree l of the Legendre polynomial component
                omrot: astropy quantity (angular speed)
                       the (uniform) rotation frequency of the star, with units
                       rad/s.
                       
            Returns:
                U_l:   astropy quantity (unit = (cm/s)^2)
                       centrifugal potential component U_l
        """
        
        assert lval%2 == 0, \
               "amigo.stellar_model.centrifugal_potential_l: error: \
                the value of the degree l has to be 0 or 2."
        
        if(lval == 0):
            U_l = -(omrot * self.radius)**2. / (3. * u.rad**2.)
    
        elif(lval == 2):
            U_l = (omrot * self.radius)**2. / (3. * u.rad**2.)
    
        return U_l
    
    
    
    def centrifugal_radius(self, mu, radius=None):
        """
            Calculate the centrifugally deformed stellar radius along a value
            mu = cos(theta), where theta is the colatitude. This is done 
            following Eq.(10) in Henneco et al. (2021).
            
            Parameters:
                self:   a stellar model object
                mu:     float
                        the given value of cos(theta), where theta is the 
                        colatitude
                radius: astropy quantity (single value or array); optional
                        the radial coordinates of the non-rotating model for 
                        which we want to calculate the deformed stellar radius.
                        In the default case (None) the full radial coordinate
                        profile of the non-rotating stellar model is used.
            
            Returns:
                rad_a:  astropy quantity (single value or array)
                        the centrifugally deformation of the given stellar 
                        radial coordinates and for the given value of mu. 
        """
        
        if(radius is None):
            radius = self.radius
            
        if((self._eps_l0 is None) | (self._eps_l2 is None)):
            rad_a = radius
        
        else:
            eps0_val = self._eps_l0
            eps2_val = self._eps_l2
            
        rad_a = radius * (1. + eps0_val + 0.5*eps2_val*(3.*mu**2. - 1.) )
            
        return rad_a
            


    def _pre_calculate_centrifugal_N2grid(self, mu=None):
        """
            Calculate a grid of centrifugally deformed stellar N^2 profiles,
            for a range of different values of mu = cos(theta), where theta is
            the co-latitude.
            
            The results from this subroutine are not given as output, but stored
            in the stellar_model object.
            
            Parameters:
                self:   a stellar model object
                mu:     numpy array; optional
                        the given values of cos(theta), where theta is the 
                        colatitude. If mu is None (the default), 101 
                        equidistantly spaced values between 0 and 1 are used.
            
            Returns:
                N/A
        """
        
        if(mu is None):
            self._mu_grid = np.sqrt(np.linspace(0., 1., 101))
        else:
            self._mu_grid = mu
                  
        self._centrifugal_N2_grid = []
        
        for imu in self._mu_grid:
            centrifugal_N2 = self._pre_calculate_centrifugal_N2profile(imu)
            self._centrifugal_N2_grid.append(centrifugal_N2)
        
        self._centrifugal_N2_grid = np.array(self._centrifugal_N2_grid) \
                                                                     * u.rad/u.s
            
        return 



    def _pre_calculate_centrifugal_N2profile(self, mu):
        """
            calculate the brunt-Vaisala frequency profile N^2 in the 
            centrifugally deformed star, for a given value of mu = cos(theta),
            with theta being the co-latitude. The calculations are
            done as explained in Appendix A of Henneco et al. (2021).
            
            Parameters:
                self:           a stellar_model object
                mu:             float
                                the given value of cos(theta).
            
            Returns:
                centrifugal_N2: astropy quantity array
                                the N^2-profile in the centrifugally deformed 
                                star, for the given value of mu = cos(theta).
        """
        
        if((self._eps_l0 is None) | (self._eps_l2 is None)):
            centrifugal_N2 = self._brunt2
        
        else:
            Pleg_l0 = 1.
            Pleg_l2 = 0.5 * (3.*mu**2. - 1.)
            PhiU =    (self.phi_l0 + self.U_l0) * Pleg_l0 \
                    + (self.phi_l2 + self.U_l2) * Pleg_l2
            
            eps0_val = self._eps_l0
            eps2_val = self._eps_l2
            eps = eps0_val * Pleg_l0 + eps2_val * Pleg_l2
        
            eps_fac = 1. + 4.*eps 
    
            dPdrsm = -self.rho * self.g
            drhodrsm =   ( self.rho * dPdrsm / ( self.Gamma1 * self.P ) ) \
                       - ( self.N2 * self.rho / ( self.g * u.rad**2. ) )
            Pbar =  self.P - (self.rho * PhiU)
            rhobar = self.rho + drhodrsm * PhiU / self.g
            
            mrad = 0.5*(self.radius[1:]+self.radius[:-1])
            mdrhodr = np.diff(drhodrsm) / np.diff(self.radius)
            drhodrsm2 = np.interp(self.radius, mrad, mdrhodr)
                
            mdPhiUdr = np.diff(PhiU) / np.diff(self.radius)
            dPhiUdr = np.interp(self.radius, mrad, mdPhiUdr)
            dPbardasm = -rhobar * (self.g + dPhiUdr) / eps_fac
            
            mdgdr = np.diff(self.g) / np.diff(self.radius)
            dgdr = np.interp(self.radius, mrad, mdgdr)
            drhobardasm = (drhodrsm / eps_fac) * (1. - (dgdr*PhiU/self.g**2.) \
                           + dPhiUdr/self.g) + PhiU*drhodrsm2/(eps_fac*self.g)
                           
            mgeff1 = np.diff(PhiU) / np.diff(self.radius)
            geff = self.g + np.interp(self.radius, mrad, mgeff1)
    
            Gamma1bar = self.rho * Pbar * drhobardasm * dPdrsm \
                        / (rhobar * self.P * drhodrsm * dPbardasm * self.Gamma1)
            centrifugal_N2 =   ( geff * dPbardasm * Gamma1bar / Pbar ) \
                             - ( geff * drhobardasm / rhobar )
            
        return centrifugal_N2
    
    
    
    def _solve_ode(self, dphidr, lval, omrot, ode_method, output_sol=False):
        """
            Subroutine to solve the 2nd-order ODE (Eq.(A.17) in 
            Henneco et al. 2021) required to calculate the centrifugal 
            deformation of the stellar model.
            
            Parameters:
                self:       a stellar model object
                dphidr:     float or numpy array
                            assumed or estimated value of the derivative dphi/dr
                            at the stellar center, where r=0.
                lval:       integer
                            the degree l of the (Legendre polynomial) component
                            of the centrifugal perturbation of the gravitational
                            potential that is calculated.
                omrot:      astropy quantity (unit: rad/s)
                            the angular rotation frequency
                ode_method: string
                            the numerical method that has to be used to solve
                            the ODE. This will be passed to the 'solve_ivp'
                            subroutine in scipy, and we refer to the 
                            documentation of the scipy python package for more
                            information. 
                output_sol: boolean; optional
                            indicate if the full solution returned by the 
                            scipy.integrate.solve_ivp subroutine has to be given
                            as output (default = False).
                
            Returns:
                resid:      float
                            the residuals for the outer boundary condition 
                            (i.e., at r = R), for the evaluated value of 
                            dphi/dr. This output is returned when the input 
                            variable output_sol is False (default).
                solution:   bunch object
                            the solution returned by the 
                            scipy.integrate.solve_ivp subroutine, used to solve
                            the ODE. This output is returned when the input 
                            variable output_sol is True.
        """
        
        term = 4.*np.pi*self.radius**2. * self.drhodr / self.mass
        fct1 = (lval*(lval+1.)/(self.radius**2.)) + term
        fct2 = (lval - 1.)*(omrot * self.radius)**2. * term / 3.
        
        term = term.to(1/u.cm**2.).value
        fct1 = fct1.to(1/u.cm**2.).value
        fct2 = fct2.to(u.rad**2./u.s**2.).value
        
        rad_profile = self.radius.to(u.cm).value
        minr = rad_profile[0]
        maxr = rad_profile[-1]


        def fun_rhs(rad, phil):
            """The RHS of the ODE that we need to solve."""
            
            dphildr = np.zeros(phil.shape)
            dphildr[0] = phil[1]
            dphildr[1] = -2.*phil[1]/rad \
                         + np.interp(rad,rad_profile,fct1)*phil[0] \
                         + np.interp(rad,rad_profile,fct2)
            
            return dphildr
         
        try:
            solution = solve_ivp(fun_rhs, (minr,maxr), [0.,dphidr], 
                                      method = ode_method, t_eval = rad_profile)
            diff = np.abs(solution.y[1][-1]-(lval+1.)*(solution.y[0][-1]/maxr))
            resid = diff / np.abs(solution.y[1][-1])
            
        except:
            solution = None
            resid = 10.**8.
            pass
        
        if(output_sol):
            return solution
        else:
            return resid



    ### stellar_model subroutines - part 3: differential rotation
    ### Note: the subroutines are listed alphabetically.
        
    def brunt_rot_profile(self, frot_s, delta_f, x_vals=None, frot_vals=None):
        """
            Computing the rotation profile following Eq.(10) in 
            Van Reeth et al. 2018, A&A 618, A24.
            
            Parameters:
                self:      a stellar model object
                frot_s:    astropy quantity (type= frequency or angular speed)
                           The rotation frequency at the radius 
                           rs = int(N*dr)/int(N*dr/r).
                delta_f:   float
                           the relative difference in rotation rate between the 
                           core and radius rs.
                x_vals:    array (dtype = float or astropy quantity); optional
                           the fractional radii at which rotation rates 
                           'frot_vals' are known. If given, this overrides 
                           frot_s and delta_f. (default = None)
                frot_vals: array (of astropy quantity frequency); optional
                           the rotation rates at the radii 'r_vals'. If given,
                           this overrides frot_s and delta_f. (default = None)
            
            Returns:
                frot_prof: array (astropy quantity frequency or angular speed)
                           The calculated rotation profile, which relies on the
                           Brunt-Vaisala frequency profile.
                rs:        astropy quantity
                           the radial coordinate at which the g modes are (in
                           theory and on average) most sensitive to stellar 
                           properties such as the rotation rate.
        """
        
        # Verifying the quantities and converting to the easiest units
        assert frot_s.unit.physical_type in ['angular speed','frequency'], \
               "Please ensure that the mean rotation frequency has an angular \
                speed unit or a frequency unit (as given in astropy.units)."
        
        # Normalising the brunt profile
        brunt_norm = self.N / np.trapz( self.N / self.radius, x = self.radius )
        # deriving the radius rs where the g-modes are on average most sensitive
        rs = np.trapz( brunt_norm, x = self.radius )
        
        # Computing the differential rotation
        integrand = brunt_norm**2./self.radius
        cum_integral = cumtrapz( integrand, x = self.radius, initial = 0. )
        diffrot_prof = 1. - ( cum_integral / np.amax(cum_integral) )
        
        # Rescaling the differential rotation to match the frot_vals, if given
        if((x_vals is not None) & (frot_vals is not None)):
            r_vals = np.array(x_vals)*np.amax(self.radius)
            dfrot_vals = np.interp(r_vals,self.radius,diffrot_prof)
            dfrot_norm = (diffrot_prof - dfrot_vals[0]) / np.diff(dfrot_vals)
            frot_prof = frot_vals[0] + dfrot_norm * (frot_vals[1]-frot_vals[0])
            
        else:
            frs = np.interp(rs, self.radius, diffrot_prof)
            dfrot_norm = (diffrot_prof - frs) / (diffrot_prof[0] - frs)
            frot_prof = frot_s * ( 1. + (delta_f * dfrot_norm) )
        
        return frot_prof, rs




