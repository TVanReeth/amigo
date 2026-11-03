import numpy as np
#from astropy.stats import LombScargle
from astropy.timeseries import LombScargle as LombScargle


def scargle(time, flux, e_flux=None, f0 = 0., fn='fNyq', Nsample = 10., option='amplitude'):
    """
       Calculate the amplitude spectrum using the Lomb-Scargle (L-S) periodogram
       
       Parameters:
            time (numpy array, float): time stamps of the light curve
            flux (numpy array, float): the flux variations of the light curve
            Nsample (float, optional): oversampling rate for the periodogram. Default value = 10.
            option (string, optional): preferred output format for the periodogram. The options are
                                       'psd' (power spectrum distribution; default) and 'amplitude'.
       
       Returns:
            fr (numpy array, float): evaluated frequency values in the domain of the periodogram
            sc (numpy array, float): the values of the L-S periodogram, in the format selected with 'option'.
    
    .. codeauthor:: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
    """

    ndata = len(time)                                            # The number of data points
    if(fn == 'fNyq'):
        ffin = 0.5/np.nanmedian(time[1:]-time[:-1])                     # the Nyquist frequency
    else:
        ffin = fn
    fres = 1./(time[-1]-time[0])                                 # the frequency resolution
    
    fr = np.arange(f0,ffin,fres/float(Nsample))                  # the frequencies
    if e_flux is None:
        sc1 = LombScargle(time, flux).power(fr, normalization='psd')   # The non-normalized Lomb-Scargle "power"
    else:
        sc1 = LombScargle(time, flux, e_flux).power(fr, normalization='psd')
    
    # Computing the appropriate rescaling factors (to convert to astrophysical units)
    fct = np.sqrt(4./ndata)
    
    if(option == 'psd'):
        T = time.ptp()
        sc = fct**2. * sc1 * T
    elif(option == 'amplitude'):
        sc = fct * np.sqrt(sc1)

    # Ensuring the output does not contain nans or infs
    fr = fr[np.isfinite(sc)]
    sc = sc[np.isfinite(sc)]
    
    return fr, sc
