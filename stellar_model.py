import numpy as np
import astropy.units as u
import sys

from scipy.integrate import cumtrapz


class stellar_model(object):
    
    """
        A simple class to handle stellar models, with the option to extend it to other stellar codes than MESA.
        This will make it easier to run my other codes, without having to adapt *everything*...
        FOR NOW, this is a very simple class, with only options to read in the radius and N^2, and calculate Pi0
        and a rotation profile.
        
        author: Timothy Van Reeth (KU Leuven)
                timothy.vanreeth@kuleuven.be
    """
    
    def __init__(self,rad_profile, brunt2_profile):
        """
            Setting up a stellar model object.
            
            Parameters:
                rad_profile: array (of astropy quantities; type=length)
                             the stellar radius
                brunt2_profile: array (of astropy quantities with unit rad2/s2)
                             the squared Brunt-Vaisala N^2 profile
        """
        
        # Verifying the quantities and converting to the easiest units
        assert brunt2_profile.unit == "rad2 / s2", "Please ensure that the N^2 frequency array has 'rad2 / s2' as unit (as given in astropy.units)."
        assert rad_profile.unit.physical_type == 'length', "Please ensure that the stellar radius array has a length unit (as given in astropy.units)."
        
        # Make sure that the radius array is monotonically increasing
        ind_sort = np.argsort(rad_profile)
        rad_sort = rad_profile[ind_sort]
        brunt2_sort = brunt2_profile[ind_sort]
        
        # The square root of N^2
        brunt2_sort[(brunt2_sort < 0.) | ~np.isfinite(brunt2_sort)] = 0.
        brunt_sort = np.sqrt(brunt2_sort)
        
        # Avoiding unnecessary nans
        rad_sort[0] = 0.001*rad_sort[1]
        
        self.radius = rad_sort
        self.brunt2 = brunt2_sort
        self.brunt = brunt_sort

        
        
    def buoyancy_radius(self,unit='days'):
        """
            Calculates the buoyancy radius Pi0 from a N^2 profile as a function of stellar radius.
            
            Parameters:
                self: a stellar model object
                unit: string
                      the preferred unit of the (output) Pi0. Options are 'days' and 'seconds'. Default = 'days'.
            
            Returns:
                Pi0:  astropy quantity (unit = day or seconds)
                      the buoyancy radius    
        """
        
        # Can we handle the preferred output unit?
        allowed_output_units = {'days':u.day,'seconds':u.s}
        assert unit in allowed_output_units.keys(), f"Please ensure that the requested output unit is one of: {', '.join(str(x_unit) for x_unit in allowed_output_units.keys())}."
        astrounit = allowed_output_units[unit]
        
        # Make sure that we don't run into numerical issues
        domain = self.radius > 0.
        
        # Computing the buoyancy radius
        Pi0 = 2.*np.pi**2.*u.radian/np.trapz(self.brunt[domain]/self.radius[domain], x=self.radius[domain])
        
        Pi0_conv = Pi0.to(astrounit)
        
        return Pi0_conv
        
    
    
    
    def rotation_profile_brunt(self,frot_s, delta_f, x_known=[], known_rot=[], with_rs=False):
        """
            Computing the rotation profile following Eq.10 in Van Reeth et al. 2018, A&A 618, A24.
            
            Parameters:
                self:      a stellar model object
                frot_s:    astropy quantity (type= frequency or angular speed)
                           The rotation frequency at the radius rs = int(N*dr)/int(N*dr/r).
                delta_f:   float
                           the relative difference in rotation rate between the core and radius rs.
                x_known:   list (empty or containing floats)
                           the fractional radii at which rotation rates 'known_rot' are known. If given, this overrides frot_s and delta_f.
                known_rot: list (empty or containing astropy quantities; type = frequency or angular speed)
                           the rotation rates at the fractional radii 'x_known'. If given, this overrides frot_s and delta_f.
            
            Returns:
                frot_prof: array (dtype= astropy quantity; frequency or angular speed)
                           The calculated rotation profile, which relies on the Brunt-Vaisala frequency profile.
        """
        
        # Verifying the quantities and converting to the easiest units
        assert frot_s.unit.physical_type in ['angular speed','frequency'], "Please ensure that the mean rotation frequency has an angular speed unit or a frequency unit (as given in astropy.units)."
        
        # Make sure that we don't run into numerical issues
        radius = self.radius.copy()
        radius[0] = 0.001*radius[1]
        
        # Normalising the brunt profile
        brunt_norm = np.sqrt(self.brunt2) / np.trapz(np.sqrt(self.brunt2)/self.radius,x=self.radius)
        
        # Computing the differential rotation
        diffrot_prof = 1. - ( cumtrapz(brunt_norm**2./self.radius,x=self.radius,initial=0.) / np.trapz(brunt_norm**2./self.radius,x=self.radius) )
        
        # Rescaling the differential rotation according to known_rot, if given
        if((len(x_known) > 0) & (len(known_rot) > 0)):
            rad_0 = np.array(x_known)*np.amax(self.radius)
            rot_0 = np.array(known_rot)
          
            diffrot_0 = np.interp(rad_0,self.radius,diffrot_prof)
            frot_prof = rot_0[0] + ((diffrot_prof - diffrot_0[0]) * (rot_0[1]-rot_0[0])/(diffrot_0[1]-diffrot_0[0])) 
            
        else:
            rs = np.trapz(brunt_norm,x=self.radius)
            frs = np.interp(rs, self.radius, diffrot_prof)
            
            frot_prof = frot_s * ( 1. + (delta_f * (diffrot_prof - frs) / (diffrot_prof[0]-frs)) )
        
        if(with_rs):
            return frot_prof,rs
        else:
            return frot_prof

    
    def omegacrit_scale(self):
        """
            Calculate the local "critical" scaling rotation rate from a non-rotating stellar model, given a mass and (corresponding) radial coordinate, in cgs.
            Here, we follow the "Keplerian definition", but as given by Mathis & Prat (2019).
            
            Parameters:
                mass (float): stellar mass coordinate(s) in grams
                radius (float): corresponding radial coordinate(s) in cm
            Out:
                omk (float): the critical ("Keplerian") rotation rate (in rad s^-1)
        """
        
        G = 6.6743 * 10.**(-8.)
        omk = np.sqrt(G*self.mass/(self.radius**3.))

        return omk
    

    def omegacrit_roche(self):
        """
            Calculate the local critical rotation rate from a non-rotating stellar model, given a mass and (corresponding) radial coordinate, in cgs.
            Here, we follow the "Roche model definition", but as given by Bouabid et al. (2013).
            
            Parameters:
                mass (float): stellar mass coordinate(s) in grams
                radius (float): corresponding radial coordinate(s) in cm
            Out:
                omk (float): the critical ("Roche") rotation rate (in rad s^-1)
        """
        
        G = 6.6743 * 10.**(-8.)
        omk = np.sqrt(8.*G*self.mass/(27.*self.radius**3.))
        
        return omk