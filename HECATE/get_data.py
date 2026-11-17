# Personal functions to fetch data.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy.io import fits
import os
import glob

from HECATE.utils import get_phase_mu

def get_CCFs(planet_params:dict, stellar_params:dict, directory_path:str='/home/telmo/PhD_work/Eduardos_code/white_light_ccfs/', day:str='2021-08-11', index_to_remove:str=None, plot:bool=True):
    """Fetch ESPRESSO white-light CCFs data.

    Parameters
    ----------
    planet_params : `dict`
        dictionary containing the following planetary parameters: orbital period, system scale, planet-to-star radius ratio, mid-transit time, eccentricity, argument of periastron, planetary inclination and spin-orbit angle.
    stellar_params : `dict`
        dictionary containing the following stellar parameters: effective temperature and error, superficial gravity and error, metallicity and error.
    directory_path : `str`
        path where the white light CCFs are stored.
    day : `str`
        in case where the target was observed in more than one night, choose the one to use.
    index_to_remove : `str`
        in case the user wants to remove a given CCF a priori.
    plot : `bool` 
        whether to plot the CCF profiles colored by orbital phase.

    Returns
    -------
    CCFs : `numpy array` 
        matrix with the CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points).
    header_info : `dict`
        dictionary containing the following information: time (BJD), airmass, BERV, maximum BERV,  SNR at each spectral order and CCF radial velocity and its error.
    list_ccfs : `str` 
        list of CCFs file path and names.
    """
    listfiles = glob.glob(os.path.join(directory_path, '*.fits'))

    list_ccfs = [name for name in listfiles if day in name and 'CCF_SKYSUB_A' in name]
    if len(list_ccfs) == 0:
        list_ccfs = [name for name in listfiles if day in name and 'CCF_A' in name]

    list_ccfs = sorted(list_ccfs)

    #removing low SNR observations
    if index_to_remove == "last":
        list_ccfs = list_ccfs[:-1]
    elif index_to_remove == None:
        list_ccfs = list_ccfs
    else:
        list_ccfs = [x for i,x in enumerate(list_ccfs) if i not in index_to_remove]

    n_points = fits.getdata(list_ccfs[0],1)[-1].shape[0]

    CCFs = np.zeros((len(list_ccfs), 3, n_points))
    time = np.zeros(len(list_ccfs))
    airmass = np.zeros(len(list_ccfs))
    berv = np.zeros(len(list_ccfs))
    bervmax = np.zeros(len(list_ccfs))
    RV_central = np.zeros((len(list_ccfs), 2))
    snr_orders = np.zeros((len(list_ccfs),171)) # SNR for each order

    for i,name in enumerate(list_ccfs):

        d = fits.getdata(name,1)
        de = fits.getdata(name,2)
        h = fits.getheader(name)

        N = len(d[-1])
        a = h['HIERARCH ESO RV START']
        step = h['HIERARCH ESO RV STEP']
        X = np.arange(a,a+step*N,step)

        CCFs[i,0,:] = X 
        CCFs[i,1,:] = d[-1]
        CCFs[i,2,:] = de[-1]
        
        time[i] = h['HIERARCH ESO QC BJD']
        try:
            airmass[i] = h['HIERARCH ESO TEL1 AIRM START'] 
        except:
            airmass[i] = h['HIERARCH ESO TEL3 AIRM START']
        berv[i] = h['HIERARCH ESO QC BERV']
        bervmax[i] = h['HIERARCH ESO QC BERVMAX']
        RV_central[i,0] = h['HIERARCH ESO QC CCF RV']
        RV_central[i,1] = h['HIERARCH ESO QC CCF RV ERROR']

        for j in range(0,170):
            snr_orders[i,j] = h[f'HIERARCH ESO QC ORDER{j+1} SNR']

    if plot:
        phases = get_phase_mu(time, planet_params, stellar_params).phases
        norm = Normalize(vmin=phases.min(), vmax=phases.max())
        cmap = plt.get_cmap('coolwarm_r')

        fig, ax = plt.subplots(figsize=(6,3.5))

        for i in range(CCFs.shape[0]):
            color = cmap(norm(phases[i]))
            ax.plot(CCFs[i,0], CCFs[i,1], c=color)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Orbital Phase')
        ax.set_xlabel("RV [km/s]")
        ax.set_ylabel("CCF Flux")
        ax.grid()
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.show()

    header_info = {'time': time, 'airmass': airmass, 'berv': berv, 'bervmax': bervmax, 'snr_orders': snr_orders, 'RV_central': RV_central}

    return CCFs, header_info, list_ccfs


def get_spectra(directory_path:str='/home/telmo/PhD_work/Eduardos_code/telluric_corrected_spectra/', day:str='2021-08-11', type_spec:str='TELL_CORR_A', index_to_remove:str=None):
    """Fetch ESPRESSO spectra.

    Parameters
    ----------
    directory_path : `str`
        path where the spectra are stored.
    day : `str`
        in case where the target was observed in more than one night, choose the one to use.
    type_spec : `str`
        type of spectra to be retrieved (e.g., S1D_TELL_CORR_A).
    index_to_remove : `str`
        in case the user wants to remove a given spectra a priori.

    Returns
    -------
    spectra : `numpy array` 
        matrix with the spectra (wavelenth in air, flux, flux error and quality flag), with shape (N_spectra, 4, N_pixels).
    header_info : `dict`
        dictionary containing the following information: time (BJD), airmass, BERV, maximum BERV and SNR at each spectral order.
    list_spectra : `str` 
        list of spectra file path and names.
    headers : `list`
        list of headers of the spectra.
    """
    listfiles = glob.glob(os.path.join(directory_path, '*.fits'))

    list_spectra = [name for name in listfiles if day in name and type_spec in name]
    list_spectra = sorted(list_spectra)

    #removing low SNR observation
    if index_to_remove == "last":
        list_spectra = list_spectra[:-1]
    else:
        if index_to_remove is not None:
            list_spectra = [x for i,x in enumerate(list_spectra) if i not in index_to_remove]

    hdul0 = fits.open(list_spectra[0])
    tbl = hdul0[1].data   
    hdul0.close()
    npix = np.array(tbl.field(1)).shape[0]

    spectra = np.zeros((len(list_spectra),4,npix))
    time = np.zeros(len(list_spectra))
    airmass = np.zeros(len(list_spectra))
    berv = np.zeros(len(list_spectra))
    bervmax = np.zeros(len(list_spectra))
    snr_orders = np.zeros((len(list_spectra),171)) # SNR for each order
    rv = np.zeros(len(list_spectra))
    headers = []
    
    for i in range(len(list_spectra)):

        hdul = fits.open(list_spectra[i])
        hdr = hdul[0].header
        tbl = hdul[1].data   
        hdul.close()

        spectra[i,0] = np.array(tbl.field(1)) #wavelength in air
        spectra[i,1] = np.array(tbl.field(2)) #flux
        spectra[i,2] = np.array(tbl.field(3)) #flux error
        spectra[i,3] = np.array(tbl.field(4)) #quality

        time[i] = hdr['HIERARCH ESO QC BJD']
        for telescope in (1, 3, 5):
            key = f'HIERARCH ESO TEL{telescope} AIRM START'
            if key in hdr:
                airmass[i] = hdr[key]
                break
        else:
            raise KeyError('No supported airmass keyword found (TEL1, TEL3, or TEL5 AIRM START)')

        berv[i] = hdr['HIERARCH ESO QC BERV']
        bervmax[i] = hdr['HIERARCH ESO QC BERVMAX']
        rv[i] = hdr['HIERARCH ESO QC CCF RV']

        for j in range(0,170):
            snr_orders[i,j] = hdr[f'HIERARCH ESO QC ORDER{j+1} SNR']

        headers.append(hdr)

    header_info = {'time': time, 'airmass': airmass, 'berv': berv, 'bervmax': bervmax, 'snr_orders': snr_orders, 'rv': rv}

    return spectra, header_info, list_spectra, headers