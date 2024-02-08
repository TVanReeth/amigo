#!/usr/bin/env python3
#
# File: observations.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: <TO BE ADDED>

import numpy as np
import astropy.units as u


class pulsations(object):
    
    """
        A python class to read in and handle observed pulsations and pulsation 
        patterns

        Author: Timothy Van Reeth
                timothy.vanreeth@kuleuven.be
        
        License: GPL-3+
    """
    
    def __init__(self,starname,filename):
        """
            Initialising an object to contain all the info on the observed 
            pulsations of a given star.

            Parameters:
                starname: string
                          the identifier
                filename: string
                          the file containing the observed period-spacing 
                          patterns. The required file format is demonstrated in
                          the demo file:
                            - There are different columns with headers, listing:
                                * the pulsation periods ('per')
                                * error margins on the periods ('e_per')
                                * the pulsation amplitudes ('ampl')
                                * error margins on the amplitudes ('e_ampl')
                                * the pulsation phases ('phase')
                                * error margins on the phases ('e_phase')
                                * the signal-to-noise ratio ('stopcrit')
                            - For now, only the columns with the pulsation 
                              periods and their error margins are required for
                              the code. The other parameter values can be 
                              replaced with dummy values. They are included to
                              allow for easy upgrades to the code.
                            - The pulsations in each pattern are listed in order
                              of increasing period.
                            - Gaps in the patterns are indicated with '--' 
                              entries.
                            - The end of each pattern is indicated by a '******'
                              line.
        """

        # reading in the data
        self.starname = starname
        freq, e_freq, per, e_per, ampl, e_ampl, ph, e_ph, sn, sequence = \
                                                  self.read_amigo_file(filename)

        # The usual observables
        self.freq = freq
        self.e_freq = e_freq
        self.per = per
        self.e_per = e_per
        self.ampl = ampl
        self.e_ampl = e_ampl
        self.ph = ph
        self.e_ph = e_ph
        self.signaltonoise = sn
        
        self.sequence = sequence
        self.nval = None
        self.kval = None
        self.mval = None


    
    def frequency(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the pulsation frequencies, based on specified 
            selection parameters.
            
            Parameters:
                self:     the pulsations object
                split:    boolean; optional
                          indicate if the returned pulsations frequencies have
                          to be split up in separate lists, based on 
                          period-spacing pattern membership. (default = False.)
                sequence: numpy array; optional
                          sequence index numbers; if given, return pulsation 
                          frequencies matching these index numbers.
                          (default = None)
                n:        numpy array; optional
                          radial orders; if given, return pulsation frequencies
                          matching these radial orders. (default = None)
                k:        numpy array; optional
                          meridional degrees; if given, return pulsation 
                          frequencies matching these meridional degrees. 
                          (default = None)
                m:        numpy array; optional
                          azimuthal orders; if given, return pulsation 
                          frequencies matching these azimuthal orders. 
                          (default = None)
            
            Returns:
                freq:     numpy array
                          pulsation frequencies, matching the provided selection
                          criteria
        """
        
        freq = self.retrieve_selected_quantity(self.freq, split=split, 
                                               sequence=sequence, n=n, k=k, m=m)
        
        return freq


    def e_frequency(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the pulsation frequency errors, based on 
            specified selection parameters.
            
            Parameters:
                self:     the pulsations object
                split:    boolean; optional
                          indicate if the returned pulsations frequency errors 
                          have to be split up in separate lists, based on 
                          period-spacing pattern membership. (default = False.)
                sequence: numpy array; optional
                          sequence index numbers; if given, return pulsation 
                          frequency errors matching these index numbers.
                          (default = None)
                n:        numpy array; optional
                          radial orders; if given, return pulsation frequency 
                          errors matching these radial orders. (default = None).
                k:        numpy array; optional
                          meridional degrees; if given, return pulsation 
                          frequency errors matching these meridional degrees. 
                          (default = None)
                m:        numpy array; optional
                          azimuthal orders; if given, return pulsation 
                          frequency errors matching these azimuthal orders. 
                          (default = None)
            
            Returns:
                e_freq:   numpy array
                          error margins on the pulsation frequencies, matching 
                          the provided selection criteria
        """
        
        e_freq = self.retrieve_selected_quantity(self.e_freq, split=split, 
                                               sequence=sequence, n=n, k=k, m=m)
        
        return e_freq
    

    def period(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the pulsation periods, based on specified 
            selection parameters.
            
            Parameters:
                self:     the pulsations object
                split:    boolean; optional
                          indicate if the returned pulsations periods have
                          to be split up in separate lists, based on 
                          period-spacing pattern membership. (default = False.)
                sequence: numpy array; optional
                          sequence index numbers; if given, return pulsation 
                          periods matching these index numbers.
                          (default = None)
                n:        numpy array; optional
                          radial orders; if given, return pulsation periods
                          matching these radial orders. (default = None)
                k:        numpy array; optional
                          meridional degrees; if given, return pulsation 
                          periods matching these meridional degrees. 
                          (default = None)
                m:        numpy array; optional
                          azimuthal orders; if given, return pulsation 
                          periods matching these azimuthal orders. 
                          (default = None)
            
            Returns:
               per:       numpy array
                          pulsation periods, matching the provided selection
                          criteria
        """
        
        per = self.retrieve_selected_quantity(self.per, split=split, 
                                               sequence=sequence, n=n, k=k, m=m)
        
        return per

    
    def e_period(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the pulsation period errors, based on 
            specified selection parameters.
            
            Parameters:
                self:     the pulsations object
                split:    boolean; optional
                          indicate if the returned pulsations period errors 
                          have to be split up in separate lists, based on 
                          period-spacing pattern membership. (default = False.)
                sequence: numpy array; optional
                          sequence index numbers; if given, return pulsation 
                          period errors matching these index numbers.
                          (default = None)
                n:        numpy array; optional
                          radial orders; if given, return pulsation period 
                          errors matching these radial orders. (default = None).
                k:        numpy array; optional
                          meridional degrees; if given, return pulsation 
                          period errors matching these meridional degrees. 
                          (default = None)
                m:        numpy array; optional
                          azimuthal orders; if given, return pulsation 
                          period errors matching these azimuthal orders. 
                          (default = None)
            
            Returns:
                e_per:    numpy array
                          error margins on the pulsation periods, matching 
                          the provided selection criteria
        """
        
        e_per = self.retrieve_selected_quantity(self.e_per, split=split, 
                                               sequence=sequence, n=n, k=k, m=m)
        
        return e_per


    def amplitude(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the pulsation amplitudes, based on specified 
            selection parameters.
            
            Parameters:
                self:     the pulsations object
                split:    boolean; optional
                          indicate if the returned pulsations amplitudes have
                          to be split up in separate lists, based on 
                          period-spacing pattern membership. (default = False.)
                sequence: numpy array; optional
                          sequence index numbers; if given, return pulsation 
                          amplitudes matching these index numbers.
                          (default = None)
                n:        numpy array; optional
                          radial orders; if given, return pulsation amplitudes
                          matching these radial orders. (default = None)
                k:        numpy array; optional
                          meridional degrees; if given, return pulsation 
                          amplitudes matching these meridional degrees. 
                          (default = None)
                m:        numpy array; optional
                          azimuthal orders; if given, return pulsation 
                          amplitudes matching these azimuthal orders. 
                          (default = None)
            
            Returns:
                ampl:     numpy array
                          pulsation amplitudes, matching the provided selection
                          criteria
        """
        
        ampl = self.retrieve_selected_quantity(self.ampl, split=split, 
                                               sequence=sequence, n=n, k=k, m=m)
        
        return ampl


    def e_amplitude(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the pulsation amplitude errors, based on 
            specified selection parameters.
            
            Parameters:
                self:     the pulsations object
                split:    boolean; optional
                          indicate if the returned pulsations amplitude errors 
                          have to be split up in separate lists, based on 
                          period-spacing pattern membership. (default = False.)
                sequence: numpy array; optional
                          sequence index numbers; if given, return pulsation 
                          amplitude errors matching these index numbers.
                          (default = None)
                n:        numpy array; optional
                          radial orders; if given, return pulsation amplitude 
                          errors matching these radial orders. (default = None).
                k:        numpy array; optional
                          meridional degrees; if given, return pulsation 
                          amplitude errors matching these meridional degrees. 
                          (default = None)
                m:        numpy array; optional
                          azimuthal orders; if given, return pulsation 
                          amplitude errors matching these azimuthal orders. 
                          (default = None)
            
            Returns:
                e_ampl:   numpy array
                          error margins on the pulsation amplitudes, matching 
                          the provided selection criteria
        """
        
        e_ampl = self.retrieve_selected_quantity(self.e_ampl, split=split, 
                                               sequence=sequence, n=n, k=k, m=m)
        
        return e_ampl
    

    def phase(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the pulsation phases, based on specified 
            selection parameters.
            
            Parameters:
                self:     the pulsations object
                split:    boolean; optional
                          indicate if the returned pulsations phases have
                          to be split up in separate lists, based on 
                          period-spacing pattern membership. (default = False.)
                sequence: numpy array; optional
                          sequence index numbers; if given, return pulsation 
                          phases matching these index numbers.
                          (default = None)
                n:        numpy array; optional
                          radial orders; if given, return pulsation phases
                          matching these radial orders. (default = None)
                k:        numpy array; optional
                          meridional degrees; if given, return pulsation 
                          phases matching these meridional degrees. 
                          (default = None)
                m:        numpy array; optional
                          azimuthal orders; if given, return pulsation 
                          phases matching these azimuthal orders. 
                          (default = None)
            
            Returns:
                ph:       numpy array
                          pulsation phases, matching the provided selection
                          criteria
        """
        
        ph = self.retrieve_selected_quantity(self.ph, split=split, 
                                               sequence=sequence, n=n, k=k, m=m)
        
        return ph


    def e_phase(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the pulsation phase errors, based on 
            specified selection parameters.
            
            Parameters:
                self:     the pulsations object
                split:    boolean; optional
                          indicate if the returned pulsations phase errors 
                          have to be split up in separate lists, based on 
                          period-spacing pattern membership. (default = False.)
                sequence: numpy array; optional
                          sequence index numbers; if given, return pulsation 
                          phase errors matching these index numbers.
                          (default = None)
                n:        numpy array; optional
                          radial orders; if given, return pulsation phase 
                          errors matching these radial orders. (default = None).
                k:        numpy array; optional
                          meridional degrees; if given, return pulsation 
                          phase errors matching these meridional degrees. 
                          (default = None)
                m:        numpy array; optional
                          azimuthal orders; if given, return pulsation 
                          phase errors matching these azimuthal orders. 
                          (default = None)
            
            Returns:
                e_ph:     numpy array
                          error margins on the pulsation phases, matching 
                          the provided selection criteria
        """
        
        e_ph = self.retrieve_selected_quantity(self.e_ph, split=split, 
                                               sequence=sequence, n=n, k=k, m=m)
        
        return e_ph

    
    def sn(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the signal-to-noise (S/N) ratios of the pulsations, 
            based on specified selection parameters.
            
            Parameters:
                self:     the pulsations object
                split:    boolean; optional
                          indicate if the returned S/N ratios have
                          to be split up in separate lists, based on 
                          period-spacing pattern membership. (default = False.)
                sequence: numpy array; optional
                          sequence index numbers; if given, return S/N ratios
                          matching these index numbers. (default = None)
                n:        numpy array; optional
                          radial orders; if given, return S/N ratios matching
                          these radial orders. (default = None)
                k:        numpy array; optional
                          meridional degrees; if given, return S/N ratios
                          matching these meridional degrees. 
                          (default = None)
                m:        numpy array; optional
                          azimuthal orders; if given, return S/N ratios 
                          matching these azimuthal orders. (default = None)
            
            Returns:
                sn:       numpy array
                          S/N ratios, matching the provided selection criteria
        """
        
        sn = self.retrieve_selected_quantity(self.signaltonoise, split=split, 
                                               sequence=sequence, n=n, k=k, m=m)
        
        return sn
    

    def seqid(self, split=False, sequence=None, n=None, k=None, m=None):
        """
            Retrieve (some?) of the pattern sequence indices of the pulsations, 
            based on specified selection parameters. Multiples of 1000 in the 
            value indicate which pattern is formed, and added numbers indicate
            which pulsation in the pattern it is. For example, a pulsation with 
            sequence=2015 is the 15th pulsation in the 2nd detected pattern.
            
            Parameters:
                self:     the pulsations object
                split:    boolean; optional
                          indicate if the returned sequence values have to be 
                          split up in separate lists, based on period-spacing 
                          pattern membership. (default = False.)
                sequence: numpy array; optional
                          sequence index numbers; if given, return sequence 
                          values matching these index numbers. 
                          (default = None)
                n:        numpy array; optional
                          radial orders; if given, return sequence values 
                          matching these radial orders. (default = None)
                k:        numpy array; optional
                          meridional degrees; if given, return sequence values
                          matching these meridional degrees. 
                          (default = None)
                m:        numpy array; optional
                          azimuthal orders; if given, return sequence values 
                          matching these azimuthal orders. (default = None)
            
            Returns:
                seq:      numpy array
                          sequence values, matching the provided selection 
                          criteria
        """
        
        seq = self.retrieve_selected_quantity(self.sequence, split=split, 
                                               sequence=sequence, n=n, k=k, m=m)
        
        return seq


    
    def retrieve_selected_quantity(self, quantity, split=False, sequence=None, 
                                                        n=None, k=None, m=None):
        """
            generalised routine to retrieve selected quantities from a
            pulsations object

            Parameters:
                self:             the pulsations object
                quantity:         pulsations object attribute
                                  the quantity attribute of the pulsations 
                                  object that has to be retrieved
                split:            boolean; optional
                                  split the quantity values into a list, 
                                  according to pattern membership (default 
                                  value = False).
                sequence:         numpy array; optional
                                  sequence index numbers; if given, return 
                                  pulsation quantity values according to these 
                                  index numbers. (default = None)
                n:                numpy array; optional
                                  radial orders; if given, return pulsation 
                                  quantity values according to these radial 
                                  orders. (default = None)
                k:                numpy array; optional
                                  meridional degrees; if given, return pulsation
                                  quantity values according to these meridional 
                                  degrees. (default = None)
                m:                numpy array; optional
                                  azimuthal orders; if given, return pulsation 
                                  quantity values according to these azimuthal 
                                  orders. (default = None)
            
            Returns:
                quantity[subset]: pulsations object attribute
                                  the requested quantity attribute of the 
                                  pulsations object, or (if at least one of the
                                  optional keyword arguments is not None) a 
                                  subset of it.
        """
        
        if(split == True):
            if(self.sequence is None):
                return [quantity]
            else:
                seq_ids = np.rint(self.sequence / 1000)
                return [quantity[np.r_[seq_ids == seq_id]] 
                                               for seq_id in np.unique(seq_ids)]

        elif(sequence is not None):
            seq_ids = np.rint(self.sequence / 1000)
            if(np.r_[seq_ids == sequence].any()):
                return quantity[np.r_[seq_ids == sequence]]
            else:
                print('WARNING: the given sequence identifier does not match \
                                                      any of the known values.')
                return None

        elif((n is not None) | (k is not None) | (m is not None)):
            if((n is not None) & (self.nval is not None)):
                if(type(n) == list):
                    nsel = np.array([np.r_[n == nval].any() 
                                             for nval in self.nval], dtype=bool)
                else:
                    nsel = np.r_[n == self.nval]
            else:
                nsel = np.array(np.ones(self.nval.shape), dtype=bool)

            if((k is not None) & (self.kval is not None)):
                if(type(k) == list):
                    ksel = np.array([ np.r_[k == kval].any() 
                                             for kval in self.kval], dtype=bool)
                else:
                    ksel = np.r_[k == self.kval]
            else:
                ksel = np.array(np.ones(self.kval.shape), dtype=bool)

            if((m is not None) & (self.mval is not None)):
                if(type(m) == list):
                    msel = np.array([ np.r_[m == mval].any() 
                                             for mval in self.mval], dtype=bool)
                else:
                    msel = np.r_[m == self.mval]
            else:
                msel = np.array(np.ones(self.mval.shape), dtype=bool)
            
            mode_id_sel = np.r_[nsel & ksel & msel]

            return quantity[mode_id_sel]
        
        else:
            # return everything if no custom selection specifiers were given
            return quantity




    def patterns(self, to_plot=False):
        """
            Provide the analysed g-mode patterns in terms of period-spacing 
            (measured between modes with consecutive radial orders n and 
            identical mode identification (k,m)) as a function of the pulsation
            periods. 
            
            Parameters:
                self:        a pulsations object
                to_plot:     boolean; optional
                             indicate if, in addition to the pulsation periods 
                             and spacings, you also require information to  
                             properly plot the period-spacing patterns, i.e.,
                             which spacings are consecutive and which ones 
                             aren't, which will be connected with full and 
                             dashed lines, respectively.
            
            Returns:
                seq_per:     list (of numpy arrays, dtype = float)
                             g-mode pulsation periods, listed per period-spacing
                             pattern, for modes that are followed by another
                             mode with a consecutive value of radial order n.
                e_seq_per:   list (of numpy arrays, dtype = float)
                             error margins on the g-mode pulsation periods 
                             listed in seq_per.
                seq_dp:      list (of numpy arrays, dtype = float)
                             spacings between the pulsation periods listed in
                             seq_per.
                e_seq_dp:    list (of numpy arrays, dtype = float)
                             error margins on the g-mode pulsation spacings
                             listed in seq_dp.
                plot_solid:  list; optional (returned if to_plot = True)
                             begin and end points of connections between 
                             consecutive period spacings (i.e., consecutive in
                             terms of radial order n), listed in seq_per and 
                             seq_dp.
                plot_dashed: list; optional (returned if to_plot = True)
                             begin and end points of connections between 
                             non-consecutive period spacings (i.e., 
                             non-consecutive in terms of radial order n), listed
                             in seq_per and seq_dp.                             
        """
        
        split_periods = self.period(split=True)
        e_split_periods = self.e_period(split=True)
        split_seq = self.seqid(split=True)
        
        seq_per = []
        e_seq_per = []
        seq_dp = []
        e_seq_dp = []
        plot_solid = []
        plot_dashed = []
        
        for per, e_per, seqid in zip(split_periods, e_split_periods, split_seq):
            no_gaps = np.r_[np.diff(seqid) == 1]
            obsp = per.to(u.d)[:-1][no_gaps]
            e_obsp = e_per.to(u.d)[:-1][no_gaps]
            obsdp = np.diff(per)[no_gaps]
            e_obsdp = np.sqrt(e_per[1:][no_gaps]**2. + e_per[:-1][no_gaps]**2.)
            
            seq_per.append(obsp)
            e_seq_per.append(e_obsp)
            seq_dp.append(obsdp)
            e_seq_dp.append(e_obsdp)
            
            if(to_plot):
                cont = np.r_[np.abs(np.diff(obsp)-obsdp[:-1]) <= 0.1*obsdp[:-1]]
                solid = [[(obsp.value[ii],obsdp.to(u.s).value[ii]),
                          (obsp.value[ii+1],obsdp.to(u.s).value[ii+1])]
                                       for ii,icont in enumerate(cont) if icont]
                dashed = [[(obsp.value[ii],obsdp.to(u.s).value[ii]),
                           (obsp.value[ii+1],obsdp.to(u.s).value[ii+1])]
                                   for ii,icont in enumerate(cont) if not icont]
                plot_solid.append(solid)
                plot_dashed.append(dashed)
                
        if(to_plot):
            return seq_per, e_seq_per, seq_dp, e_seq_dp, plot_solid, plot_dashed
            
        else:
            return seq_per, e_seq_per, seq_dp, e_seq_dp
    
    
    
    def spin(self, modes, frot, split=False):
        """
            Calculate the spin parameter values for the pulsations, given a
            rotation frequency and assumed mode identifications (k,m) for the 
            observed period-spacing patterns.
            
            Based on the most common observations, it is assumed that if we are
            dealing with gravito-inertial modes (k >= 0), the pulsation 
            frequencies in the corotating frame are larger than m times the
            rotation frequency. For r-modes (k < 0) the opposite is true.
            
            Parameters:
                self:  the pulsations object
                modes: list of gravity_modes objects
                       variables containing (assumed) mode identifications (k,m)
                       for the observed period-spacing patterns.
                frot:  astropy quantity (frequency)
                       stellar rotation frequency. We either assume uniform 
                       rotation, or take this to be the stellar rotation in the
                       near-core region of the star, where the g-modes are
                       dominant.
                split: boolean
                       indicate if the spin parameter values have to be split
                       according to period-spacing pattern membership or not.
                       (default = False).
            
            Returns:
                nu:    numpy array (dtype = float)
                       calculated spin parameter values 
        """
        
        freq_cor = []
        freq_inert = self.frequency(split=True)
        
        for freqs,mode in zip(freq_inert, modes):
            if(mode.kval != 0):
                sgn_k = mode.kval / abs(mode.kval)
            else:
                sgn_k = 1.
            freq_cor.append(sgn_k*freqs.to(1./u.d) - mode.mval*frot.to(1./u.d))
        
        nus = [2.*frot.to(1./u.d).value / freqs.value for freqs in freq_cor]
        
        if(split):
            nu = nus
        else:
            nu = []
            for inu in nus:
                nu = nu + list(inus)
            nu = np.array(nu)            
        
        return nu
    
    


    def read_amigo_file(self,filename):
        """
            A routine to read in the 'amigo' pattern files, containing the 
            periods, amplitudes, phases (with errors) for pulsations in 
            period-spacing patterns.

            Parameters:
                self:          a pulsations object
                filename:      string
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
                               the pulsation phases, with values between -0.5 
                               and 0.5.
                e_phase:       numpy array, dtype=float
                               the errors on the pulsation phases
                sn:            numpy array, dtype=float
                               the signal-to-noise ratios of the detected 
                               pulsations
                sequence:      numpy array, dtype=int
                               indices indicating which pulsations form 
                               patterns. Multiples of 1000 indicate which 
                               pattern is formed, and added numbers indicate 
                               which pulsation in the pattern it is. For 
                               example, a pulsation with sequence=2015 is the 
                               15th pulsation in the 2nd detected pattern.
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
                values = np.array([float(val) for val in line.strip().split()])
                period.append(values[ind_period])
                e_period.append(values[ind_e_period])
                ampl.append(values[ind_ampl])
                e_ampl.append(values[ind_e_ampl])
                phase.append(values[ind_phase])
                e_phase.append(values[ind_e_phase])
                sn.append(values[ind_sn])

                seqval += 1
                sequence.append(seqval)

        period = np.array(period) * u.day
        e_period = np.array(e_period) * u.day
        ampl = np.array(ampl)      # unitless
        e_ampl = np.array(e_ampl)  # unitless
        phase = np.array(phase)
        e_phase = np.array(e_phase)
        sn = np.array(sn)
        sequence = np.array(sequence)

        frequency = 1./period
        e_frequency = e_period / (period**2.)

        return frequency, e_frequency, period, e_period, ampl, e_ampl, \
               phase, e_phase, sn, sequence




