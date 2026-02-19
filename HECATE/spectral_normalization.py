# Class to normalize spectra, both globally and locally.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class norm_spec:
    """Class to normalize spectra, both globally and locally. 
        
    Parameters
    ----------
    phases : `numpy array`
        orbital phases of the spectra.
    spectra : `numpy array`
        matrix with the systemic velocity corrected spectra (wavelenth, flux, flux error).
    
    Methods
    -------
    global_norm(mask, plot=True, plot_masks=None)
        Performs global normalization of the spectra using a given wavelength range to act as reference.
    cut_spectrum(spectra_global_norm, wave_min=6450, wave_max=6650)
        Cut the globally normalized spectra to a specific wavelength region to spare memory.
    local_norm(spectra_region, mask_line=[(6530,6590)], mask_poly=[(6538.8,6545.8),(6546.9,6551.4),(6575.6,6579.8),(6581.4,6586.05)], plot=True, line_name='Halpha')
        Normalize locally the spectra around the spectral line of interest by fitting a linear polynomial to the continuum.
    """
    def __init__(self,phases:np.array, spectra:np.array):

        self.phases = phases
        self.spectra = spectra


    def global_norm(self, mask:np.ndarray=[(6400,6800)], plot:bool=True, plot_masks:dict=None):
        """Performs global normalization of the spectra using a given wavelength range to act as reference.
        
        Parameters
        ----------
        phases : `numpy array`
            orbital phases of the spectra.
        spectra : `numpy array`
            matrix with the systemic velocity corrected spectra (wavelenth, flux, flux error).
        mask : list of tuples
            wavelength range(s) to use as reference for the normalization.
        plot : `bool`
            wether to plot the normalized spectra and different zoomed in regions.
        plot_masks : `dict`, optional
            wavelength ranges for zoomed plots in the form of dictionary with key being the subplot title and value being the interval as a tuple. 

        Returns
        -------
        spectra_global_norm : `numpy array`
            matrix with the globally normalized spectra (wavelenth, flux, flux error).
        """
        spectra_global_norm = np.zeros_like(self.spectra)
        
        wave = self.spectra[0, 0, :]

        norm_mask = np.zeros(len(wave), dtype=bool) #mask for the normalization regions
        for interval in mask:
            norm_mask |= (wave >= interval[0]) & (wave <= interval[1])

        for i in range(spectra_global_norm.shape[0]):

            wave = self.spectra[i,0,:]
            flux = self.spectra[i,1,:]
            flux_err = self.spectra[i,2,:]

            med = np.median(flux[norm_mask])

            flux_norm = flux/med
            median_err = np.sqrt(np.sum((flux_err[norm_mask])**2))/len(flux_err[norm_mask])

            spectra_global_norm[i,0,:] = wave
            spectra_global_norm[i,1,:] = flux_norm
            spectra_global_norm[i,2,:] = flux_norm*np.sqrt((flux_err/flux)**2+(median_err/med)**2)

        self.spectra_global_norm = spectra_global_norm

        if plot:
            
            if plot_masks is None:
                plot_masks = {'Full spectrum': None,
                    '382.0 to 384.5 nm': (3820, 3845),
                    '460 to 462.5 nm': (4600, 4625),
                    '585 to 593.5 nm': (5850, 5935),
                    '640 to 680 nm': (6400, 6800),
                    '775.5 to 778.0 nm': (7755, 7780)}

            norm = Normalize(vmin=min(self.phases), vmax=max(self.phases))
            normalized_phases = norm(self.phases)
            cmap = plt.get_cmap('coolwarm')

            fig, ax = plt.subplots(nrows=len(plot_masks), figsize=(9, 3.5*len(plot_masks)))

            for i in range(spectra_global_norm.shape[0]):
                color = cmap(normalized_phases[-i-1])
                
                wave = spectra_global_norm[i, 0, :]
                flux_norm = spectra_global_norm[i, 1, :]
                
                for plot_idx, (title, wavelength_range) in enumerate(plot_masks.items()):
                    if wavelength_range is None:
                        ax[plot_idx].plot(wave, flux_norm, linewidth=1, color=color)
                    else:
                        plot_mask = (wave >= wavelength_range[0]) & (wave <= wavelength_range[1])
                        ax[plot_idx].plot(wave[plot_mask], flux_norm[plot_mask], linewidth=1, color=color)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  
            cbar = fig.colorbar(sm, ax=ax, fraction=0.05)
            cbar.set_label('Orbital Phase')

            for plot_idx, (title, _) in enumerate(plot_masks.items()):
                ax[plot_idx].set_title(title)
                ax[plot_idx].set_ylabel('Relative flux')
                if plot_idx == len(plot_masks) - 1:
                    ax[plot_idx].set_xlabel(r'Wavelength [$\AA$]')

            plt.subplots_adjust(hspace=0.4, wspace=0.5)
            plt.show()

        return spectra_global_norm
    

    def cut_spectrum(self, spectra_global_norm:np.ndarray, wave_min:float=6450, wave_max:float=6650):
        '''Cut the globally normalized spectra to a specific wavelength region to spare memory.
        
        Parameters
        ----------
        spectra_global_norm : `numpy array`
            matrix with the globally normalized spectra (wavelength, flux, flux error).
        wave_min : `float`
            minimum wavelength of the region.
        wave_max : `float`
            maximum wavelength of the region.

        Returns
        -------
        spectra_region : `numpy array`
            cut spectra (wavelength, flux, flux error).
        '''
        wv_first = spectra_global_norm[0,0,:]
        mask = (wv_first >= wave_min) & (wv_first <= wave_max)

        spectra_region = np.zeros([spectra_global_norm.shape[0],3,wv_first[mask].shape[0]])

        for i in range(spectra_global_norm.shape[0]):

            spectra_region[i,0,:] = spectra_global_norm[i,0,:][mask]
            spectra_region[i,1,:] = spectra_global_norm[i,1,:][mask]
            spectra_region[i,2,:] = spectra_global_norm[i,2,:][mask]

        return spectra_region

    
    def local_norm(self, spectra_region:np.ndarray, mask_line:np.ndarray=[(6535,6590)], mask_continuum:np.ndarray=[(6538.8,6545.8),(6546.9,6551.4),(6575.6,6579.8),(6581.4,6586.05)], plot:bool=True, line_name:str=r'H$\alpha$'):
        '''Normalize locally the spectra around the spectral line of interest by fitting a linear polynomial to the continuum.
        
        Parameters
        ----------
        spectra_region : `numpy array`
            matrix with the globally normalized and sliced spectra (wavelength, flux, flux error).
        mask_line : `list of tuples`
            list of tuples with the wavelength ranges around the line for plot and memory saving purposes.
        mask_poly : `list of tuples`
            list of tuples with the wavelength ranges of the continuum.
        plot : `bool`
            whether to plot the normalization.
        line_name : `str`
            spectral line(s) name for plot titles.

        Returns
        -------
        spectra_local_norm : `numpy array`
            locally normalized spectra (wavelength, flux, flux error).
        poly_coefs_array : `numpy array`
            polynomial coefficients used for the normalization of each spectrum.
        '''
        wave = spectra_region[0,0,:] #first spectrum as example

        line_mask = np.zeros(len(wave), dtype=bool) #mask for the line region
        for interval in mask_line:
            line_mask |= (wave >= interval[0]) & (wave <= interval[1])
        
        poly_mask = np.zeros(len(wave[line_mask]), dtype=bool) #mask for the continuum regions
        for interval in mask_continuum:
            poly_mask |= (wave[line_mask] >= interval[0]) & (wave[line_mask] <= interval[1])

        spectra_local_norm = np.zeros((spectra_region.shape[0],3,wave[line_mask].shape[0]))

        poly_coefs_array = np.zeros((spectra_region.shape[0],2))

        for i in range(spectra_region.shape[0]):

            wave = spectra_region[i,0,:][line_mask]
            flux = spectra_region[i,1,:][line_mask]
            flux_err = spectra_region[i,2,:][line_mask]
            
            poly_coefs, _ = np.polyfit(wave[poly_mask], flux[poly_mask], w=1/(flux_err[poly_mask])**2, deg=1, cov=True)
            flux_norm = flux/(poly_coefs[0]*wave+poly_coefs[1])

            spectra_local_norm[i,0,:] = wave
            spectra_local_norm[i,1,:] = flux_norm
            spectra_local_norm[i,2,:] = flux_norm*np.sqrt((flux_err/flux)**2)

            poly_coefs_array[i,:] = poly_coefs

    
        if plot: 

            norm = Normalize(vmin=min(self.phases), vmax=max(self.phases))
            normalized_phases = norm(self.phases)
            cmap = plt.get_cmap('coolwarm_r')

            fig, axes = plt.subplots(nrows=3, figsize=(9,15))

            axes[0].plot(spectra_local_norm[0,0,:], spectra_local_norm[0,1,:], color='black', linewidth=0.5) 
            for interval in mask_continuum:
                mask = (spectra_local_norm[0,0,:] >= interval[0]) & (spectra_local_norm[0,0,:] <= interval[1])
                axes[0].plot(spectra_local_norm[0,0,:][mask], spectra_local_norm[0,1,:][mask], color='red', linewidth=1)
            
            axes[0].set_title(f'Continuum around {line_name} for local normalization')
            axes[0].set_ylabel('Relative flux')

            for i in range(spectra_local_norm.shape[0]):

                flux = spectra_region[i,1,:][line_mask]
                wave = spectra_region[i,0,:][line_mask]

                color = cmap(normalized_phases[i])

                axes[1].plot(wave, flux, linewidth=0.5, color=color)
                axes[1].plot(wave, poly_coefs_array[i,0]*wave+poly_coefs_array[i,1], linewidth=1.5, color=color, linestyle='dashed', label='Continuum fit')

                axes[2].plot(wave,flux_norm,linewidth=0.5,color=color)
            
            axes[1].set_title(f'{line_name} spectra with continuum fit')
            axes[1].set_ylabel('Relative flux')

            axes[2].set_title(f'Locally normalized {line_name} spectra')
            axes[2].set_xlabel(r'Wavelength [$\AA$]')
            axes[2].set_ylabel('Relative flux')

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar_ax1 = fig.add_axes([0.92, 0.11, 0.01, 0.49])
            cbar1 = plt.colorbar(sm,cax=cbar_ax1)
            cbar1.set_label('Orbital phases')

            plt.show()

        return spectra_local_norm, poly_coefs_array