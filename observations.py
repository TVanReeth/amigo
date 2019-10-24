import numpy as np
import astropy.units as u

try:
  # Something to make life fun...
  from pratchett import HEX
except:
  import sys as HEX

"""
    A simple python class to handle observed pulsations and pulsation patterns...
    
    Author: Timothy Van Reeth
            timothy.vanreeth@kuleuven.be
"""

class pulsations(object):
    
    def __init__(self,starname,filename,filetype='spacings'):
        """
            Initialising an object to contain all the info on the observed pulsations of a given star.
            
            Parameters:
                starname: string
                          the identifier
                filename: string
                          the file containing the observations. This can be either a "<starname>_spacings.dat" file (following my old format)
                          or a file with all the observed frequencies, with an extra column (or several), containing info on 'pattern membership'.
                filetype: string
                          specifying the type of file with observations is given. The optiosn are 'spacings' or 'scargle'. The default is 'spacings'.
        """
        
        # Something to tie this to other data
        self.starname = starname
        
        if(('_spacings.dat' in filename) | (filetype == 'spacings')):
            frequency, e_frequency, period, e_period, amplitude, e_amplitude, phase, e_phase, sn, sequence = self.read_spacings_file(filename)
        
        elif(('_scargle.dat' in filename) | (filetype == 'scargle')):
            HEX.exit('pulsations.__init__: cannot read in "scargle files" yet... If you have some time to write the function? *smiles hopefully*')
        
        # The usual observables
        self.frequency = frequency
        self.e_frequency = e_frequency
        self.period = period
        self.e_period = e_period
        self.amplitude = amplitude
        self.e_amplitude = e_amplitude
        self.phase = phase
        self.e_phase = e_phase
        self.sn = sn
        
        # Attempting to reorganize my period spacing series into something more versatile...
        self.sequence = sequence
        self.nval = None
        self.kval = None
        self.mval = None
        
    
    
    def read_spacings_file(self,filename):
        """
            A routine to read in the old '*_spacings.dat' files, containing the periods, amplitudes, 
            phases and errors for pulsations forming patterns.
            
            Parameters:
                self:     a pulsations object
                filename: string
                          the file containing the observations. 
            
            Returns:
                frequency:     numpy array, dtype=float
                               the pulsation frequencies
                e_frequency:   numpy array, dtype=float
                               the errors on the pulsation frequencies
                period:        numpy array, dtype=float
                               the pulsation periods
                e_period:      numpy array, dtype=float
                               the errors on the pulsation periods
                ampl:          numpy array, dtype=float
                               the pulsation amplitudes
                e_ampl:        numpy array, dtype=float
                               the errors on the pulsation amplitudes
                phase:         numpy array, dtype=float
                               the pulsation phases, with values between -0.5 and 0.5.
                e_phase:       numpy array, dtype=float
                               the errors on the pulsation phases
                sn:            numpy array, dtype=float
                               the signal-to-noise ratios of the detected pulsations
                sequence:      numpy array, dtype=int
                               indices indicating which pulsations form patterns.
                               Multiples of 1000 indicate which pattern is formed, and added numbers indicate which pulsation in the pattern it is.
                               For example, a pulsation with sequence=2015 is the 15th pulsation in the 2nd detected pattern.
        """
        
        period = []
        e_period = []
        ampl = []
        e_ampl = []
        phase = []
        e_phase = []
        sn = []
        sequence = []
        
        file = open(filename,'r')
        lines = file.readlines()
        file.close()
        
        seqval = 1000
        head = lines[0].replace('#','')
        parameters = head.strip().split()
        
        ind_period = parameters.index('per')
        ind_e_period = parameters.index('e_per')
        ind_ampl = parameters.index('ampl')
        ind_e_ampl = parameters.index('e_ampl')
        ind_phase = parameters.index('phase')
        ind_e_phase = parameters.index('e_phase')
        ind_sn = parameters.index('stopcrit')
        
        for line in lines[1:]:
            if('--' in line):
                seqval += 1
            
            elif('***' in line):
                seqval = int(1000*np.ceil(seqval / 1000.))
            
            elif(line.isspace()):
                continue
            
            else:
                values = np.array([float(value) for value in line.strip().split()])
                period.append(values[ind_period])
                e_period.append(values[ind_e_period])
                ampl.append(values[ind_ampl])
                e_ampl.append(values[ind_e_ampl])
                phase.append(values[ind_phase])
                e_phase.append(values[ind_e_phase])
                sn.append(values[ind_sn])
                
                seqval += 1
                sequence.append(seqval)
        
        period = np.array(period)*u.day
        e_period = np.array(e_period)*u.day
        ampl = np.array(ampl)*1000.   #*u.mmag; maybe add units later...
        e_ampl = np.array(e_ampl)*1000.   #*u.mmag
        phase = np.array(phase)
        e_phase = np.array(e_phase)
        sn = np.array(sn)
        sequence = np.array(sequence)
        
        frequency = 1./period
        e_frequency = e_period / (period**2.)
        
        return frequency, e_frequency, period, e_period, ampl, e_ampl, phase, e_phase, sn, sequence







        
