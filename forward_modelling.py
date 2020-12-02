import itertools as it
import numpy as np
import sobol_seq

from amigo.stellar_model import stellar_model
from amigo.gmode_series import asymptotic

"""
    A module with several subroutines to facilitate the forward modelling of pulsating sdtars. For now, this is
    a very simple module, only focusing on the sampling of the parameter space and providing a basic merit function
    for model evaluation.
    
    author: Timothy Van Reeth (KU Leuven)
            timothy.vanreeth@kuleuven.be
"""

def sampling(parameters,minbound,maxbound,method='regular',parameter_formats='f8',return_structured=True,include_raw=False,**kwargs):
    """ 
        A routines to sample a (bound) parameter space.
        
        Parameters:
            parameters: list (dtype = string)
                        The names of the parameters which are considered in the parameter space.
            minbound:   numpy array
                        The minimum (allowed) values of the considered parameters. NOTE: these are simple float values. 
                        Units are not considered within this routine! If the user wants the parameters to correspond to
                        astropy quantities with matching units, these have to be converted to ordinary numbers (and back)
                        when using this routine.
            maxbound:   numpy array
                        The maximum (allowed) values of the considered parameters. NOTE: these are simple float values. 
                        Units are not considered within this routine! If the user wants the parameters to correspond to
                        astropy quantities with matching units, these have to be converted to ordinary numbers (and back)
                        when using this routine.
            method:     string or list (of strings)
                        The method(s) to be used to sample the parameter space. If a single method (string) is provided, 
                        it will be applied to the entire parameter space. If a list f methods is provided, a single method 
                        per parameter has to be specified. The available options are 'regular' (= a regular grid), 'sobol' (sequence)
                        and 'random'. Default = 'regular'.
            parameter formats: string
                        The format of the different parameters. By default, it is assumed that all parameter are floats 'f8'.
            return_structured: boolean
                        If the preferred output is a structured numpy array (master_sample['i_parameter_name']) rather than a normal
                        array (master_sample[:,i_parameter]), return_structured is True (=default).
            include_raw: boolean
                        If true, include the non-rescaled Sobol sequence array in the output (in the same format as the scaled 
                        parameter list). False = default.
            **kwargs:   parameters for the different sampling methods (Nrandom, Nsobol, Nregulars: how many models are required; 
                        regular_step: step size in a regular grid. Alternative for Nregulars).
        
        Returns:
            master_sample: (structured?) numpy array
                        the generated samples in the parameter space.
    """
    
    # Which parameters do you need?
    if(type(parameter_formats) == str):
        paramtypes = [parameter_formats]*len(parameters)
    elif(type(parameter_formats) == list):
        assert len(parameter_formats) == len(parameters), "Please provide a format for each parameter."
        paramtypes = parameter_formats.copy()
    else:
        sys.exit("forward_modelling.sampling: something is wrong with the formats 'parameter_formats' given for the input parameters.")
        
    # So... How do you want to do this?
    available_methods = ['sobol','regular','random']
    if(type(method) == str):
        assert method in available_methods, f"Please make sure that the method used to sample the parameter space is one of {', '.join(x_unit for x_unit in available_methods)}."
        selected_methods = [method]*len(parameters)
    
    elif(type(method) == list):
        assert len(method) == len(parameters), "Please provide a method to sample the parameter space for each parameter."

        assert np.array([indiv in available_methods for indiv in method],dtype=bool).all(), f"Please make sure that the methods used to sample the parameter space are any of {', '.join(x_unit for x_unit in available_methods)}."

        selected_methods = method.copy()
    else:
        sys.exit("forward_modelling.sampling: (one or some of) the given method(s) to sample the parameter space are not valid.")
        
    # Making sure that the required keyword arguments for the different methods are listed in **kwargs
    if('sobol' in selected_methods):
        assert 'Nsobol' in kwargs.keys(), "Please provide the total number of models Nsobol that has to be generated using the Sobol sequence."
        Nsobol = kwargs['Nsobol']
    if('random' in selected_methods):
        assert 'Nrandom' in kwargs.keys(), "Please provide the total number of models Nrandom that has to be generated randomly."
        Nrand = kwargs['Nrandom']
    if('regular' in selected_methods):
        assert ('Nregulars' in kwargs.keys()) | ('regular_step' in kwargs.keys()), "Please provide the number of models Nregulars that has to be generated on a regular grid for each of the selected parameters, or provide the step size."
        
        if('Nregulars' in kwargs.keys()):
            Nregulars = kwargs['Nregulars']
        else:
            Nregulars = 1 + np.floor((maxbound - minbound)/kwargs['regular_step'])
    
    # Creating (sub)samples for each of the possible methods, for the specified parameters for each
    sobol_selection = [imeth for imeth,meth in enumerate(selected_methods) if meth == 'sobol']
    regular_selection = [imeth for imeth,meth in enumerate(selected_methods) if meth == 'regular']
    random_selection = [imeth for imeth,meth in enumerate(selected_methods) if meth == 'random']
    
    if(len(sobol_selection) > 0):
        sobol_sample = sobol_sequence(len(sobol_selection),Nsobol)
    else:
        sobol_sample = [[]]
    if(len(regular_selection) > 0):
        regular_sample = regular_grid(len(regular_selection),Nregulars)
    else:
        regular_sample = [[]]
    if(len(random_selection) > 0):
        random_sample = random(len(random_selection),Nrand)
    else:
        random_sample = [[]]
    
    # Merging the subsamples into a merged sample, combining them with itertools.product()
    merged_sample = list(it.product(sobol_sample,regular_sample,random_sample))
    
    # Now... flattening the different parameter combinations into lists, and sorting them
    flat_index = np.argsort(np.array(sobol_selection + regular_selection + random_selection,dtype=int))
    
    if(return_structured):
        #master_sample = [np.hstack(*args)[flat_index] for args in merged_sample]
        parameters = ['index']+parameters
        paramtypes = ['i4']+paramtypes
        master_sample = np.array([tuple([int(iarg+1)]+list(np.hstack(args)[flat_index])) for iarg,args in enumerate(merged_sample)],dtype=[(parameter,partype) for parameter,partype in zip(parameters,paramtypes)])
        
        if(include_raw):
            raw_sample = np.array([tuple([int(iarg+1)]+list(np.hstack(args)[flat_index])) for iarg,args in enumerate(merged_sample)],dtype=[(parameter,partype) for parameter,partype in zip(parameters,paramtypes)])
        # rescaling the normalised values so (0,1) --> (minbound,maxbound)
        for parameter,imin,imax in zip(parameters[1:],minbound,maxbound):
            master_sample[parameter] = master_sample[parameter]*(imax-imin) + imin
    else:
        if(include_raw):
            raw_sample = [np.array([int(iarg+1)]+list(np.hstack(args)[flat_index])) for iarg,args in enumerate(merged_sample)]
        master_sample = [np.array([int(iarg+1)]+list(np.hstack(args)[flat_index])) for iarg,args in enumerate(merged_sample)]
        
        # rescaling the normalised values so (0,1) --> (minbound,maxbound)
        for ipar,imin,imax in zip(np.arange(len(parameters)),minbound,maxbound):
            master_sample[:,ipar+1] = master_sample[:,ipar+1]*(imax-imin) + imin
    
    if(include_raw):
        return master_sample, raw_sample
    else:
        return master_sample



def sobol_sequence(lensobol,Nsobol):
    """
        Generate a sobol sequence

        Parameters:
            lensobol: integer
                      The number of dimensions for the sobol numbers
            Nsobol:   integer
                      The number of sobol numbers that have to be generated.
        
        Returns:
            sobol_sample: numpy array
                      The generated sequence of sobol numbers (normalised to unity along each dimension).
    """
    
    sobol_sample = sobol_seq.i4_sobol_generate(lensobol,Nsobol)
    return sobol_sample



def random(lenrand,Nrand):
    """
        Generate a random set of samples, assuming a uniform distribution within the parameter (sub)space.

        Parameters:
            lenrand: integer
                     The number of dimensions for the random numbers
            Nrand:   integer
                     The number of random numbers that have to be generated.
        
        Returns:
            random_sample: numpy array
                      The generated sequence of random numbers (normalised to unity along each dimension).
    """
    
    random_sample = np.random.rand(Nrand,lenrand)
    return random_sample



def regular_grid(lenregular,Nregulars):
    """
        Generate a regular grid of samples.

        Parameters:
            lenregular:  integer
                         The number of dimensions for the grid
            Nregulars:   array, dtype=integer
                         The number of regularly spaced numbers that have to be generated (along each axis).
        
        Returns:
            regular_sample: numpy array
                      The generated sequence of regularly spaced samples (normalised to unity along each dimension).
    """
    
    args = (np.linspace(0.,1.,iregular) for iregular in Nregulars)
    regular_sample = it.product(*args)
    return regular_sample

