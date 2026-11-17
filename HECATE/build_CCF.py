# Class to build CCF from spectra.

from bisect import bisect_left
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from utils import get_phase_mu
from numba import jit, prange

@jit(nopython=True)
def _bisect_left_numba(a, x):
    """Numba-compatible binary search (bisect_left implementation)."""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo

@jit(nopython=True, parallel=True)
def compute_ccf_slow_numba(samples, contrast, mask, RV_reference, mask_width, 
                           ll2, dll, imin, imax, nx_ccf, c):
    """JIT-compiled parallelized CCF computation for Monte Carlo error estimation.
    
    Parameters
    ----------
    samples : ndarray
        Pre-computed spectral flux samples (shape: [M, N])
    contrast : ndarray
        Contrast weights for mask
    mask : ndarray
        Spectral line-list
    RV_reference : ndarray
        RV grid
    mask_width : float
        Width of mask in velocity units
    ll2 : ndarray
        Half-pixel shifted wavelength array
    dll : ndarray
        Wavelength gradient
    imin, imax : int
        Range of mask indices to use
    nx_ccf : int
        Length of CCF
    c : float
        Speed of light in km/s
    
    Returns
    -------
    matrix : ndarray
        CCF matrix with shape [nx_ccf, N]
    """
    N = samples.shape[1]
    matrix = np.zeros((nx_ccf, N))
    
    for n in prange(N):
        ccf_flux_matrix = np.zeros(nx_ccf)
        
        for i in range(imin, imax + 1):
            w = contrast[i] * contrast[i]
            
            for j in range(nx_ccf):
                llcenter = mask[i] * (1. + RV_reference[j] / c)
                llstart = llcenter - 0.5 * mask_width / c * llcenter
                llstop = llcenter + 0.5 * mask_width / c * llcenter
                
                # Binary search for Numba compatibility
                index1 = _bisect_left_numba(ll2, llstart) + 1
                index2 = _bisect_left_numba(ll2, llcenter) + 1
                index3 = _bisect_left_numba(ll2, llstop) + 1
                
                k = j
                for index in range(index1, index3):
                    ccf_flux_matrix[k] += w * samples[index-1, n]
                
                ccf_flux_matrix[k] += w * samples[index1-1-1, n] * (ll2[index1-1] - llstart) / dll[index1-1-1]
                ccf_flux_matrix[k] -= w * samples[index3-1-1, n] * (ll2[index3-1] - llstop) / dll[index3-1-1]
        
        matrix[:, n] = ccf_flux_matrix
    
    return matrix

class build_CCF:
    """Class to build CCFs from spectra and a spectral line-list. Method adapted from iCCF.meta.espdr_compute_CCF_fast of iCCF package.
        
    Parameters
    ----------
    time : `numpy array`
        time of observations in BJD.
    planet_params : `dict`
        dictionary containing the following planetary parameters: orbital period, system scale, planet-to-star radius ratio, mid-transit time, eccentricity, argument of periastron, planetary inclination and spin-orbit angle.
    stellar_params : `dict`
        dictionary containing the following stellar parameters: effective temperature, surface gravity and metallicity.
    spectra : `numpy array`
        matrix with the spectra (wavelenth in air, flux, flux error and quality flag), with shape (N_spectra, 4, N_pixels).
    RV_reference : `numpy array`
        RV grid for interpolation.
    mask : `numpy array`
        spectral line-list to perform cross-correlation.
    berv : `numpy array`
        BERV at time of observation.
    bervmax : `numpy array`
        maximum BERV.
    error_type : `str`
        method to compute CCF error bars. "fast" computes the error bars by propagating the flux errors, while "slow" computes the error bars by building the covariance matrix of the CCF through Monte Carlo sampling of the fluxes.
    n_samples : `int`
        number of Monte Carlo samples for the "slow" error type.
    plot : `bool`
        whether to plot CCFs in function of 
    
    Methods
    -------
    compute_CCF(ll, flux, error, quality, RV_reference, mask, berv, bervmax, mask_width=0.5, plot=True)
        performs cross-correlation between a given spectrum and the line-list following the procedure of iCCF.
    """
    def __init__(self, time:np.array, planet_params:dict, stellar_params:dict, spectra:np.array, RV_reference:np.array, mask:np.array, berv:np.array=None, bervmax:np.array=None, error_type:str="fast", n_samples:int=1000, plot:bool=False):

        CCFs = np.zeros((spectra.shape[0], 3, RV_reference.shape[0]))

        if berv is None:
            berv = np.zeros(time.shape[0])
        if bervmax is None:
            bervmax = np.zeros(time.shape[0])

        n_lines_used = []
        cov_matrices = []

        for i in tqdm.tqdm(range(spectra.shape[0])):

            ccf = self.compute_CCF(ll = spectra[i,0], flux = spectra[i,1], error = spectra[i,2], quality = spectra[i,3],
                                    RV_reference = RV_reference, mask = mask, 
                                    berv = berv[i] if berv is not None else 0, bervmax = bervmax[i] if bervmax is not None else 0, 
                                    mask_width = 0.5, error_type = error_type, n_samples = n_samples, plot=False)
            
            CCFs[i,0] = RV_reference
            CCFs[i,1] = ccf[0]
            CCFs[i,2] = ccf[1]
            n_lines_used.append(ccf[3])
            if error_type == "slow":
                cov_matrices.append(ccf[4])
        
        self.CCFs = CCFs
        self.n_lines = n_lines_used
        self.cov_matrices = cov_matrices

        if plot:
            
            phases = get_phase_mu(time, planet_params, stellar_params).phases
            norm = Normalize(vmin=phases.min(), vmax=phases.max())
            cmap = plt.get_cmap('coolwarm_r')

            fig, ax = plt.subplots(figsize=(7,4))

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

    # adapted from iCCF.meta.espdr_compute_CCF_fast
    @staticmethod
    def compute_CCF(ll:np.array, flux:np.array, error:np.array, quality:np.array, RV_reference:np.array, mask:np.array, contrast:np.array=None, berv:float=0, bervmax:float=0, error_type:str="fast", n_samples:int=1000, mask_width:float=0.5, plot:bool=True):
        """Performs cross-correlation between a given spectrum and the line-list following the procedure of meta.espdr_compute_CCF_fast of iCCF.
        
        Parameters
        ----------
        ll : `numpy array`
            wavelength array in air.
        flux : `numpy array`
            flux array.
        error : `numpy array`
            flux error array in air.
        quality : `numpy array`
            quality flag array in air.
        RV_reference : `numpy array`
            RV grid.
        mask : `numpy array`
            spectral line-list.
        contrast : `numpy array`
            contrast weights for mask. If None, a box window is used.
        berv : `float`
            BERV value.
        bervmax : `float`
            maximum BERV value.
        error_type : `str`
            method to compute CCF error bars. "fast" computes the error bars by propagating the flux errors, while "slow" computes the error bars by building the covariance matrix of the CCF through Monte Carlo sampling of the fluxes.
        n_samples : `int`
            number of Monte Carlo samples for the "slow" error type.
        mask_width : `float`
            width of pixel, predefined to 0.5 km/s (ESPRESSO).
        plot : `bool`
            wether to plot the computed CCF.

        Returns
        -------
        ccf_flux : `numpy array`
            computed CCF flux.
        ccf_error : `numpy array` 
            computed CCF flux error.
        ccf_quality : `numpy array`
            quality flags of used spectra.
        cov_ccf : `numpy array`
            covariance matrix of the CCF (only returned for "slow" error type).
        """
        c = 299792.458 # km/s
        nx_s2d = flux.size
        n_mask = mask.shape[0]

        if contrast is None:
            contrast = np.ones_like(mask) # box window

        nx_ccf = RV_reference.shape[0]

        ccf_flux = np.zeros_like(RV_reference)
        ccf_error = np.zeros_like(RV_reference)
        ccf_quality = np.zeros_like(RV_reference)

        dll = np.gradient(ll)
        ll2 = ll - dll / 2.0 

        imin, imax = 1, nx_s2d
        while(imin < nx_s2d and quality[imin-1] != 0):
            imin += 1
        while(imax > 1 and quality[imax-1] != 0):
            imax -= 1

        if imin >= imax:
            return

        llmin = ll[imin + 1 - 1] / (1. + berv / c) * (1. + bervmax / c) / (1. + RV_reference[0] / c)
        llmax = ll[imax - 1 - 1] / (1. + berv / c) * (1. - bervmax / c) / (1. + RV_reference[nx_ccf - 1] / c)

        imin, imax = 0, n_mask - 1

        while (imin < n_mask and mask[imin] < (llmin + 0.5 * mask_width / c * llmin)):
            imin += 1
        while (imax >= 0     and mask[imax] > (llmax - 0.5 * mask_width / c * llmax)):
            imax -= 1

        nlines_used = 0
        for i in range(imin, imax + 1):
            nlines_used += 1

            llcenter = mask[i] * (1. + RV_reference[nx_ccf // 2] / c)
            w = contrast[i] * contrast[i]

            for j in range(0, nx_ccf):
                llcenter = mask[i] * (1. + RV_reference[j] / c)
                llstart = llcenter - 0.5 * mask_width / c * llcenter
                llstop = llcenter + 0.5 * mask_width / c * llcenter

                index1 = bisect_left(ll2, llstart) + 1
                index2 = bisect_left(ll2, llcenter) + 1
                index3 = bisect_left(ll2, llstop) + 1

                k = j
                for index in range(index1, index3):
                    ccf_flux[k] += w * flux[index-1]

                ccf_flux[k] += w * flux[index1-1-1] * (ll2[index1-1]-llstart) / dll[index1-1-1] 
                ccf_flux[k] -= w * flux[index3-1-1] * (ll2[index3-1]-llstop) / dll[index3-1-1] 

                if error_type == "fast":
                    ccf_error[k] += w * w * error[index2 - 1 - 1] * error[index2 - 1 - 1]

                ccf_quality[k] += quality[index2 - 1 - 1]

        if error_type == "fast":
            ccf_error = np.sqrt(ccf_error)

        elif error_type == "slow":

            N = n_samples
            M = len(flux)
            samples = np.zeros([M, N])
            for m in range(M):
                ymean = flux[m]
                ysigma = error[m]
                samples[m, :] = np.random.normal(ymean, ysigma, N)

            # call optimized JIT-compiled function
            matrix = compute_ccf_slow_numba(samples, contrast, mask, RV_reference, mask_width,
                                            ll2, dll, imin, imax, nx_ccf, c)

            cov_ccf = np.cov(matrix)
            ccf_error = np.sqrt(np.sum(cov_ccf, axis=1)) #np.sqrt(np.diag(cov_ccf))

        if plot:
            plt.figure(figsize=(6,4))
            plt.errorbar(x=RV_reference, y=ccf_flux, yerr=ccf_error, fmt="k.")
            plt.xlabel("RV [km/s]")
            plt.ylabel("CCF Flux")
            plt.grid()
            plt.show()

        return ccf_flux, ccf_error, ccf_quality, nlines_used, cov_ccf if error_type == "slow" else None