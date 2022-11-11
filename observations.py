import numpy as np
import astropy.units as u


class pulsations(object):
    
    """
        A python class to read in and handle observed pulsations and pulsation patterns

        Author: Timothy Van Reeth
                timothy.vanreeth@kuleuven.be
        
        License: GPL-3+
    """
    
    def __init__(self,starname,filename):
        """
            Initialising an object to contain all the info on the observed pulsations of a given star.

            Parameters:
                starname: string
                          the identifier
                filename: string
                          the file containing the observations. This can be either a "<starname>_spacings.dat" file (following my old format)
                          or a file with all the observed frequencies, with an extra column (or several), containing info on 'pattern membership'.
        """

        # reading in the data
        self.starname = starname
        frequency, e_frequency, period, e_period, amplitude, e_amplitude, phase, e_phase, sn, sequence = self.read_amigo_file(filename)

        # The usual observables
        self.freq = frequency
        self.e_freq = e_frequency
        self.per = period
        self.e_per = e_period
        self.ampl = amplitude
        self.e_ampl = e_amplitude
        self.ph = phase
        self.e_ph = e_phase
        self.signaltonoise = sn

        # Attempting to reorganize my period spacing series into something more versatile...
        self.sequence = sequence
        self.nval = None
        self.kval = None
        self.mval = None


    
    def frequency(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the frequencies, based on specified selection parameters
        """
        
        return self.retrieve_selected_quantity(self.freq, split=split, sequence=sequence, n=n, k=k, m=m)


    def e_frequency(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the e_frequencies, based on specified selection parameters
        """
        
        return self.retrieve_selected_quantity(self.e_freq, split=split, sequence=sequence, n=n, k=k, m=m)
    

    def period(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the periods, based on specified selection parameters
        """
        
        return self.retrieve_selected_quantity(self.per, split=split, sequence=sequence, n=n, k=k, m=m)


    def e_period(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the e_periods, based on specified selection parameters
        """
        
        return self.retrieve_selected_quantity(self.e_per, split=split, sequence=sequence, n=n, k=k, m=m)


    def amplitude(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the amplitudes, based on specified selection parameters
        """
        
        return self.retrieve_selected_quantity(self.ampl, split=split, sequence=sequence, n=n, k=k, m=m)


    def e_amplitude(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the e_amplitudes, based on specified selection parameters
        """
        
        return self.retrieve_selected_quantity(self.e_ampl, split=split, sequence=sequence, n=n, k=k, m=m)
    

    def phase(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the phases, based on specified selection parameters
        """
        
        return self.retrieve_selected_quantity(self.ph, split=split, sequence=sequence, n=n, k=k, m=m)


    def e_phase(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the e_phases, based on specified selection parameters
        """
        
        return self.retrieve_selected_quantity(self.e_ph, split=split, sequence=sequence, n=n, k=k, m=m)

    
    def sn(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the signal-to-noise values, based on specified selection parameters
        """
        
        return self.retrieve_selected_quantity(self.signaltonoise, split=split, sequence=sequence, n=n, k=k, m=m)
    

    def seqid(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the sequence values, based on specified selection parameters
        """
        
        return self.retrieve_selected_quantity(self.sequence, split=split, sequence=sequence, n=n, k=k, m=m)


    
    def retrieve_selected_quantity(self, quantity, split=False, sequence=None, n=None, k=None, m=None):
        """
            generalised routine to retrieve selected quantities from a pulsations object

            Parameters:
                self:       the pulsations object
                quantity:   pulsations object attribute
                            the quantity attribute of the pulsations object that has to be retrieved
                split:      boolean; optional
                            split the quantity values into a list, according to pattern membership
                            (default value = False)
                sequence:   numpy array; optional
                            sequence index numbers; if given, return pulsation quantity values according to these
                            index numbers. (default = None)
                n:          numpy array; optional
                            radial orders; if given, return pulsation quantity values according to these
                            radial orders. (default = None)
                k:          numpy array; optional
                            meridional degrees; if given, return pulsation quantity values according to these
                            meridional degrees. (default = None)
                m:          numpy array; optional
                            azimuthal orders; if given, return pulsation quantity values according to these
                            azimuthal orders. (default = None)
            
            Returns:
                quantity[subset]:   pulsations object attribute
                                    the requested quantity attribute of the pulsations object, or (if at least one of the
                                    optional keyword arguments is not None) a subset of it.
        """
        
        if(split == True):
            if(self.sequence is None):
                return [quantity]
            else:
                seq_ids = np.rint(self.sequence / 1000)
                return [quantity[np.r_[seq_ids == seq_id]] for seq_id in np.unique(seq_ids)]

        elif(sequence is not None):
            seq_ids = np.rint(self.sequence / 1000)
            if(np.r_[seq_ids == sequence].any()):
                return quantity[np.r_[seq_ids == sequence]]
            else:
                print('WARNING: the given sequence identifier does not match any of the known values.')
                return None

        elif((n is not None) | (k is not None) | (m is not None)):
            if((n is not None) & (self.nval is not None)):
                if(type(n) == list):
                    nsel = np.array([ np.r_[n == nval].any() for nval in self.nval], dtype=bool)
                else:
                    nsel = np.r_[n == self.nval]
            else:
                nsel = np.array(np.ones(self.nval.shape), dtype=bool)

            if((k is not None) & (self.kval is not None)):
                if(type(k) == list):
                    ksel = np.array([ np.r_[k == kval].any() for kval in self.kval], dtype=bool)
                else:
                    ksel = np.r_[k == self.kval]
            else:
                ksel = np.array(np.ones(self.kval.shape), dtype=bool)

            if((m is not None) & (self.mval is not None)):
                if(type(m) == list):
                    msel = np.array([ np.r_[m == mval].any() for mval in self.mval], dtype=bool)
                else:
                    msel = np.r_[m == self.mval]
            else:
                msel = np.array(np.ones(self.mval.shape), dtype=bool)
            
            mode_id_sel = np.r_[nsel & ksel & msel]

            return quantity[mode_id_sel]
        
        else:
            return quantity        # return everything if no custom selection specifiers were given



    def read_amigo_file(self,filename):
        """
            A routine to read in the 'amigo' pattern files, containing the periods, amplitudes,
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
        ampl = np.array(ampl)      # unitless
        e_ampl = np.array(e_ampl)  # unitless
        phase = np.array(phase)
        e_phase = np.array(e_phase)
        sn = np.array(sn)
        sequence = np.array(sequence)

        frequency = 1./period
        e_frequency = e_period / (period**2.)

        return frequency, e_frequency, period, e_period, ampl, e_ampl, phase, e_phase, sn, sequence

