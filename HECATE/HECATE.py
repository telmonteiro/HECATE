# HECATE-DS -- HarvEsting loCAl specTra with Exoplanets (Doppler Shadow)
# Hecate is a goddess in ancient Greek religion and mythology, most often shown holding a pair of torches,
# a key, or snakes, or accompanied by dogs, and in later periods depicted as three-formed or triple-bodied. 
# Hecate is often associated with illuminating what is hidden and find your way in cross-roads.

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

from run_SOAP import run_SOAP
from nested_sampling import run_nestedsampler
from spectral_normalization import norm_spec

from utils import *
from plots import *


class HECATE:

    """Main class for HECATE operations, allowing for the easy application of the Doppler Shadow technique to high-resolution data.

    This class encapsulates the extraction of local spectra/CCFs, as well as the analysis of their shapes (width, intensity, RV) and behavior (linearity, for now).

    Parameters
    ----------
    planet_params : `dict`
        dictionary containing the following planetary parameters: orbital period, system scale, planet-to-star radius ratio, mid-transit time, eccentricity, argument of periastron, planetary inclination and spin-orbit angle.
    stellar_params : `dict`
        dictionary containing the following stellar parameters: effective temperature and error, superficial gravity and error, metallicity and error, rotation period, radius and stellar inclination.
    time : `numpy array` 
        time of observations in BJD.
    CCFs : `numpy array` 
        matrix with the CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points).
    spectra : `numpy array`
        matrix with spectral line profiles (wavelength, flux and flux error), with shape (N_spectra, 3, N_points).
    soap_wv : `numpy array`, optional
        wavelength interval [min, max] in nm for SOAP simulation. Default is [380, 788].
    plot_soap : `bool`, optional
        whether to plot the simulated light curve from SOAP. Default is False.

    Methods
    -------
    extract_local_CCF(model_fit, plot, save)
        run all steps of the Doppler Shadow extraction and returns the local CCFs. Ideal for quick extraction.
    extract_local_spectral_line(model_fit_ccf, plot, line_name, wave_lims, masks_dict, save)
        run all steps of the Doppler Shadow extraction for spectral lines and returns the local spectra.
    avg_out_of_transit_profile(profiles, x_reference, profile_type, plot, save)
        computes the average out-of-transit profile (CCF or spectral line).
    get_profile_parameters(profiles, data_type, observation_type, model, print_output, plot_fit, wave_ctr_line, mask_x, save)
        computes profile parameters (central RV, width, intensity) for CCFs or spectral lines.
    sysvel_correction_CCF(CCFs, model, print_output, plot_fits, plot_sys_vel, save)
        removes the stellar systemic velocity from the initial CCFs RVs.
    _local_params_linear_fit(local_param, indices_final, title, priors, plot_nested, axes_to_fit)
        tests the linearity of local CCF parameters via nested sampling from Dynesty.
    plot_local_params(indices_final, local_params, master_params, suptitle, linear_fit, plot_nested, linear_fit_pairs, save)
        plots the local CCF parameters in function of orbital phases and mu.

    Notes
    -----
    This tool was based on the work of Gonçalves, E. et al. (2026) and contains a wrapper of SOAPv4 (Cristo, E. et al., 2025).

    References
    ----------
    [1] Gonçalves, E. et al., "Exploring the surface of HD 189733 via Doppler Shadow Analysis of Planetary Transits," Astronomy & Astrophysics, 2026

    [2] Cristo, E. et al., "SOAPv4: A new step toward modeling stellar signatures in exoplanet research", Astronomy & Astrophysics, Vol. 702, A84, 17pp., 2025
    """
    def __init__(self, planet_params:dict, stellar_params:dict, time:np.array, CCFs:np.array, spectra:np.array, soap_wv:np.array=[380,788], plot_soap:bool=False):

        # Get orbital phases and mu
        phase_mu = get_phase_mu(planet_params, time)

        self.phases = phase_mu.phases
        self.phases_in_indices = phase_mu.in_indices # indices of in-transit phases
        self.phases_out_indices = phase_mu.out_indices # indices of out-of-transit phases

        self.tr_dur = phase_mu.tr_dur # transit duration
        self.tr_ingress_egress = phase_mu.tr_ingress_egress # times of transit ingress and egress

        self.in_phases = self.phases[self.phases_in_indices] # in-transit phases
        
        self.mu = phase_mu.mu_values
        self.mu_in = self.mu[self.phases_in_indices]
        
        self.mu_min = get_phase_mu.mu(self.tr_dur/2-self.tr_ingress_egress/2, planet_params)
        self.mu_max = get_phase_mu.mu(0, planet_params)

        self.planet_params = planet_params
        self.stellar_params = stellar_params
        
        self.time = time
        self.CCFs = CCFs
        self.spectra = spectra

        # simulated light curve
        Flux_SOAP = run_SOAP(self.time, self.stellar_params, self.planet_params, plot=plot_soap, min_wav=soap_wv[0], max_wav=soap_wv[1]).flux
        self.Flux_SOAP = Flux_SOAP
            

    def extract_local_CCF(self, model_fit:str, plot:dict, save=None):
        """Run all steps of the Doppler Shadow extraction (simulated light curve, systemic velocity correction, compute average out-of-transit CCF and subtraction) and returns the local CCFs. 
        Ideal for quick extraction of local CCFs.

        Parameters
        ----------
        model_fit : `str`
            profile model to fit to CCFs.
        plot : `dict` 
            dictionary including boolean value for each type of plot available (SOAP, fits_initial_CCF, sys_vel, avg_out_of_transit_CCF, local_CCFs and whether to photometrical rescale).
        save 
            path to save plots.

        Returns
        -------
        local_CCFs : `numpy array`
            matrix with the local (in-transit) CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points).
        CCFs_flux_corr : `numpy array`
            matrix with all CCF profiles, only flux corrected, with shape (N_CCFs, 3, N_points)
        CCFs_sub_all : `numpy array` 
            matrix with all CCF profiles, flux corrected and subtracted from average out-of-transit, with shape (N_CCFs, 3, N_points).
        average_out_of_transit_CCF : `numpy array`
            matrix with the average out-of-transit CCF profile, with shape (3, N_points).
        """
        self.data_type = "CCF"

        # systemic velocity correction
        CCFs_sysvel_corr, _, _ = self.sysvel_correction_CCF(self.CCFs, model=model_fit, print_output=False, plot_fits=plot["fits_initial_CCF"], plot_sys_vel=plot["sys_vel_ccf"], save=save)

        # RV grid as the maximum minimum to minimum maximum of sys. velocity corrected CCF with 0.5 km/s step (ESPRESSO pixel size)
        RV_reference = np.arange(round(np.max(CCFs_sysvel_corr[:,0,0])), round(np.min(CCFs_sysvel_corr[:,0,-1]))+0.5, 0.5) 

        # average out-of-transit CCF
        CCF_interp, avg_out_of_transit_CCF = self.avg_out_of_transit_profile(CCFs_sysvel_corr, RV_reference, plot=plot["avg_out_of_transit_CCF"], save=save)

        CCFs_flux_corr = np.zeros_like(CCF_interp) # only flux corrected
        CCFs_sub_all = np.zeros_like(CCF_interp) # flux corrected and subtracted
        local_CCFs = np.zeros((len(self.phases_in_indices), 3, CCF_interp.shape[2])) # same as above but only in transit (local)

        l = 0
        for i in range(CCFs_flux_corr.shape[0]):
            d = CCF_interp[i,1,:]
            de = CCF_interp[i,2,:]
            
            # performing the subtraction
            sub = avg_out_of_transit_CCF[1] - d*self.Flux_SOAP[i]

            d_corr = d*self.Flux_SOAP[i]
            de_corr = np.sqrt(avg_out_of_transit_CCF[2]**2 + (de*self.Flux_SOAP[i])**2)

            CCFs_sub_all[i,0] = RV_reference
            CCFs_sub_all[i,1] = sub
            CCFs_sub_all[i,2] = de_corr

            if i in self.phases_in_indices:
                
                CCFs_flux_corr[i,0] = RV_reference
                CCFs_flux_corr[i,1] = d_corr
                CCFs_flux_corr[i,2] = de*self.Flux_SOAP[i]

                local_CCFs[l,0] = RV_reference
                local_CCFs[l,1] = sub
                local_CCFs[l,2] = de_corr

                l += 1

        if plot["local_CCFs"] == True: #local CCFs + tomography
            plot_local_profile(self, local_CCFs, CCFs_sub_all, profile_type="CCF", photometrical_rescale=plot["photometrical_rescale"], save=save)

        self.local_CCFs_data = {
            "local_CCFs": local_CCFs,
            "CCFs_flux_corr": CCFs_flux_corr,
            "CCFs_sub_all": CCFs_sub_all,
            "avg_out_of_transit_CCF": avg_out_of_transit_CCF}

        return local_CCFs, CCFs_flux_corr, CCFs_sub_all, avg_out_of_transit_CCF
    

    def extract_local_spectral_line(self, model_fit_ccf:str, plot:dict, line_name:str, wave_lims:list, masks_dict:dict={"glob_norm":[(6400, 6800)],"spec_slice":[(6450,6650)],"line_window":[(6535,6590)],"continuum":[(6538.8,6545.8),(6546.9,6551.4),(6575.6,6579.8),(6581.4,6586.05)]}, save=None):
        """Run all steps of the Doppler Shadow extraction (simulated light curve, systemic velocity correction, compute average out-of-transit CCF and subtraction) and returns the local CCFs. 
        Ideal for quick extraction of local CCFs.

        Parameters
        ----------
        model_fit_ccf : `str`
            profile model to fit to white light CCFs.
        plot : `dict` 
            dictionary including boolean value for each type of plot available (SOAP, fits_initial_CCF, sys_vel_CCF, sys_vel_line, avg_out_of_transit_spectra, local_spec_line).
        line_name : `str`
            name of the spectral line.
        wave_lims : `list`
            wavelength interval to plot in the tomography plot.
        masks_dict : `dict`
            dictionary containing the wavvelength masks as lists of tuples for (1) global normalization (2) spectrum slice to save memory (3) window containing the line of interest (4) bits of spectrum continuum (5) spectral line to fit for systemic velocity correction.
        save 
            path to save plots.

        Returns
        -------
        local_spectra : `numpy array`
            matrix with the local (in-transit) spectral line profiles (wavelength, flux and flux error), with shape (N_spectra, 3, N_points).
        spectra_flux_corr : `numpy array`
            matrix with all spectral line profiles, only flux corrected, with shape (N_spectra, 3, N_points)
        spectra_sub_all : `numpy array` 
            matrix with all spectral line profiles, flux corrected and subtracted from average out-of-transit, with shape (N_spectra, 3, N_points).
        avg_out_of_transit_spectrum : `numpy array`
            matrix with the average out-of-transit spectral line profile, with shape (3, N_points).
        """
        self.data_type = "spectral_line"

        # systemic velocity correction
        _, _, linear_fit_params = self.sysvel_correction_CCF(self.CCFs, model=model_fit_ccf, print_output=False, plot_fits=plot["fits_initial_CCF"], plot_sys_vel=plot["sys_vel_CCF"], save=save)

        #RV correction using the slope
        spectra_sys_vel_corr = np.zeros_like(self.spectra)

        for i in range(spectra_sys_vel_corr.shape[0]):

            wave = self.spectra[i,0,:]

            vel = -linear_fit_params[0]*self.phases[i] #subtraction of systemic velocity, obtained via white light CCFs

            spectra_sys_vel_corr[i,0,:] =  wave / (1+vel/299792.458) #wave * np.sqrt((1+vel/299792.458) / (1-vel/299792.458)) #shift wavelength
            spectra_sys_vel_corr[i,1,:] = self.spectra[i,1,:]
            spectra_sys_vel_corr[i,2,:] = self.spectra[i,2,:]

        normalizer = norm_spec(self.phases, spectra_sys_vel_corr)
        spectra_global_norm = normalizer.global_norm(mask=masks_dict["glob_norm"], plot=plot["spec_global_normalization"])
        spectra_region = normalizer.cut_spectrum(spectra_global_norm, wave_min=masks_dict["spec_slice"][0][0], wave_max=masks_dict["spec_slice"][0][1]) 
        spectra_local_norm, _ = normalizer.local_norm(spectra_region, mask_line=masks_dict["line_window"], mask_continuum=masks_dict["continuum"], plot=plot["spec_local_normalization"], line_name=line_name)

        # average out-of-transit spectrum
        wave_grid = np.linspace(min(spectra_local_norm[0,0,:]), max(spectra_local_norm[0,0,:]), len(spectra_local_norm[0,0,:])) 
        spectra_interp, avg_out_of_transit_spectrum = self.avg_out_of_transit_profile(spectra_local_norm, wave_grid, profile_type="line", plot=plot["avg_out_of_transit_spectrum"], save=save)

        spectra_flux_corr = np.zeros_like(spectra_interp) # only flux corrected
        spectra_sub_all = np.zeros_like(spectra_interp) # flux corrected and subtracted
        local_spectra = np.zeros((len(self.phases_in_indices), 3, spectra_interp.shape[2])) # same as above but only in transit (local)

        l = 0
        for i in range(spectra_flux_corr.shape[0]):
            d = spectra_interp[i,1,:]
            de = spectra_interp[i,2,:]
            
            # performing the subtraction
            sub = avg_out_of_transit_spectrum[1] - d*self.Flux_SOAP[i]

            d_corr = d*self.Flux_SOAP[i]
            de_corr = np.sqrt(avg_out_of_transit_spectrum[2]**2 + (de*self.Flux_SOAP[i])**2)

            spectra_sub_all[i,0] = wave_grid
            spectra_sub_all[i,1] = sub
            spectra_sub_all[i,2] = de_corr

            if i in self.phases_in_indices:
                
                spectra_flux_corr[i,0] = wave_grid
                spectra_flux_corr[i,1] = d_corr
                spectra_flux_corr[i,2] = de*self.Flux_SOAP[i] 

                local_spectra[l,0] = wave_grid
                local_spectra[l,1] = sub
                local_spectra[l,2] = de_corr

                l += 1

        if plot["local_spec_line"] == True: #local spectral line + tomography
            plot_local_profile(self, local_spectra, spectra_sub_all, profile_type="line", wave_lims=wave_lims, line_name=line_name, photometrical_rescale=plot["photometrical_rescale"], save=save)

        self.local_spectra_data = {
            "local_spectra": local_spectra,
            "spectra_flux_corr": spectra_flux_corr,
            "spectra_sub_all": spectra_sub_all,
            "avg_out_of_transit_spectrum": avg_out_of_transit_spectrum}

        return local_spectra, spectra_flux_corr, spectra_sub_all, avg_out_of_transit_spectrum
    

    def sysvel_correction_CCF(self, CCFs:np.array, model:str, print_output:bool, plot_fits:bool, plot_sys_vel:bool, save:str=None):
        """Extract the RV component due to the star's motion around the barycentre, excluding the stellar systemic velocity.
        Fits a chosen profile to the CCF, then a linear model to the out-of-transit central RVs and subtracts it to all CCFs RVs.
        
        Parameters
        ----------
            CCFs : `numpy array`
                matrix with the CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points).
            model : `str` 
                type of profile model to fit.
            print_output : `bool`
                whether to print the fit output.
            plot_fits : `bool`
                whether to plot the fit for each CCF.
            plot_sys_vel: `bool` 
                whether to plot the central RV (systemic velocity) in function of orbital phase. 
            save : `str`, optional
                path to save the plots.

        Returns
        -------
            CCFs_corr : `numpy array`
                CCFs corrected by the systemic velocity.
            x0_corr : `numpy array`
                central RVs corrected by the systemic velocity.
            poly_coefs : `numpy array`
                coefficients of the linear fit to the systemic velocity.
        """
        phases = self.phases
        tr_dur = self.tr_dur 
        tr_ingress_egress = self.tr_ingress_egress 
        in_indices = self.phases_in_indices
        out_indices = self.phases_out_indices

        y0 = np.zeros((CCFs.shape[0],2))
        x0 = np.zeros((CCFs.shape[0],2))

        for i in range(CCFs.shape[0]):

            fit_prof = fit_profile(phase=phases[i], data=CCFs[i], data_type="CCF", observation_type="raw", model_type="modified Gaussian")
            profile_parameters, _, data, y_fit, _ = fit_prof._fit(print_output=print_output)

            if plot_fits:
                plot_profile_fit(data, y_fit, phases[i], data_type="CCF", observation_type="raw", model=model, save=save)
            
            y0[i,0] = profile_parameters["continuum"][0]
            y0[i,1] = profile_parameters["continuum"][1]

            x0[i,0] = profile_parameters["central_rv"][0]
            x0[i,1] = profile_parameters["central_rv"][1]

        poly_coefs, poly_cov = np.polyfit(phases[out_indices], x0[:,0][out_indices], w=1/x0[:,1][out_indices],deg=1,cov=True)

        x0_corr = np.zeros_like(x0)
        CCFs_corr = np.zeros_like(CCFs)

        for i in range(CCFs.shape[0]):
            d = CCFs[i,1]
            de = CCFs[i,2]
            
            d_corr = d/y0[i,0]

            CCFs_corr[i,0] = CCFs[i,0] - (poly_coefs[0]*phases[i] + poly_coefs[1])
            CCFs_corr[i,1] = d_corr
            CCFs_corr[i,2] = d_corr * np.sqrt( (y0[i,1]/y0[i,0])**2 + (de/d)**2 )
            
            x0_corr[i,0] = x0[i,0] - (poly_coefs[0]*phases[i] + poly_coefs[1])
            x0_corr[i,1] = np.sqrt(x0[i,1]**2 + poly_cov[0,0]*phases[i]**2 + poly_cov[1,1])

        if plot_sys_vel:
            plot_sysvel_corr_CCF(phases, tr_dur, tr_ingress_egress, in_indices, out_indices, x0, poly_coefs, x0_corr, save)

        return CCFs_corr, x0_corr, poly_coefs


    def avg_out_of_transit_profile(self, profiles:np.array, x_reference:np.array, profile_type:str="CCF", plot:bool=False, save:str=None):
        """Computes the average out-of-transit profile (CCF or spectral line) by linearly interpolating the profiles (CCFs or sliced spectra) into a common grid.
        In the case of CCFs, they must be corrected by the systemic velocity before averaging. The interpolated uncertainties are propagated taking the covariances into account.
        
        Parameters
        ----------
        profiles : `numpy array`
            matrix with the out-of-transit profiles (CCFs or spectral lines), with shape (N_profiles, 3, N_points).
        x_reference : `numpy array`
            grid for interpolation (RV for CCFs, wavelength for spectral lines).
        profile_type : `str`
            whether the profiles are CCFs or spectral lines, as they require slightly different treatments.
        plot : `bool`
            whether to plot the average out-of-transit profile.
        save : `str`, optional
            path to save the plot.
        
        Returns
        -------
        profile_interp : `numpy array`
            matrix with interpolated profiles (CCFs or spectral lines), with shape (N_profiles, 3, N_points).
        avg_out_of_transit_profile : `numpy array`
            matrix with the average out-of-transit profile (CCF or spectral line), with shape (3, N_points).
        """
        if profile_type == "CCF":
            M = profiles.shape[0]
            K = profiles.shape[2]

            cov_matrix = np.zeros((M, K, K)) 
            N = 10000      # covariance matrix obtained by sampling the CCFs 10 000 times
            
            for i in range(M): 
                samples = np.zeros((K, N))

                for j in range(K):
                    ymean = profiles[i,1,j]
                    ysigma = profiles[i,2,j]
                    samples[j,:] = np.random.normal(ymean, ysigma, N)

                cov_matrix[i,:,:] = np.cov(samples)

        out_of_transit_profiles = np.zeros([len(self.phases_out_indices), 3, len(x_reference)])
        profile_interp = np.zeros([profiles.shape[0], 3, profiles.shape[2]])

        k, M = 0, 0
        for l in range(profiles.shape[0]):
            x = profiles[l,0]
            flux = profiles[l,1]

            flux_e = cov_matrix[l] if profile_type == "CCF" else diags(profiles[l,2,:])**2 # full covariance matrix for CCFs, diagonal covariance matrix for spectral lines
            
            # build interpolation matrix for this CCF → target grid
            W = linear_interpolation_matrix(x, x_reference) 

            y_i = W @ flux # interpolated flux
            cov_new = W @ flux_e @ W.T # propagated covariance
            y_i_e = np.sqrt(cov_new.diagonal()) # propagated uncertainty

            profile_interp[l,0,:] = x_reference
            profile_interp[l,1,:] = y_i
            profile_interp[l,2,:] = y_i_e

            if l in self.phases_out_indices:
                out_of_transit_profiles[k,0,:] = x_reference
                out_of_transit_profiles[k,1,:] = y_i
                out_of_transit_profiles[k,2,:] = y_i_e

                k += 1
            else:
                M += 1

        average_out_of_transit_profile = np.mean(out_of_transit_profiles[:,1,:], axis=0)

        A_e = np.sum(out_of_transit_profiles[:,2,:]**2, axis=0) # propagation of uncertainty into the average profile
        average_out_of_transit_profile_e = np.sqrt(A_e) / len(self.phases_out_indices)

        avg_out_of_transit_profile = np.array([x_reference, average_out_of_transit_profile, average_out_of_transit_profile_e])
        
        if plot:
            plot_avg_out_of_transit_profile(avg_out_of_transit_profile, profile_type, save)

        return profile_interp, avg_out_of_transit_profile


    def get_profile_parameters(self, profiles:np.array, data_type:str, observation_type:str, model:str, print_output:bool, plot_fit:bool, wave_ctr_line:list=[(0,0)], mask_x:np.ndarray=None, save=None):
        """Computes the profile parameters of an array of CCFs or spectral line profiles.

        Parameters
        ----------
        profiles : `numpy array`
            matrix with profiles, with shape (N_observations, 3, N_points).
        data_type : `str`
            whether it's a CCF or spectral line profile.
        observation_type : `str`
            whether it's a local, average out-of-transit or raw CCF/spectral line. 
        model : `str`
            type of profile model to fit.
        print_output : `bool`
            whether to print fit output.
        plot_fit : `bool` 
            whether to plot the fit.
        wave_ctr_line : `list`, optional
            central wavelength of spectral lines.
        mask_x : `numpy array`, optional
            mask intervals for fitting region.
        save : `str`, optional
            path to save plot.

        Returns
        -------
        dict : `dict`
            Dictionary containing:
            - 'central_rv': central RV of the input CCFs/spectral lines. Single array for single line, list of arrays for multiple lines.
            - 'continuum': continuum level of the input CCFs/spectral lines.
            - 'intensity': intensity of the input CCFs/spectral lines. Single array for single line, list of arrays for multiple lines.
            - 'width': width measure of the input CCFs/spectral lines. Single array for single line, list of arrays for multiple lines.
            - 'R2': coefficient of determination of all fits.
            - 'flux_fit_params': list of fit parameters for all profiles.
        """
        N = profiles.shape[0] if observation_type == "local" else 1

        continuum_array = np.zeros((N,2))
        R2_array = np.zeros(N)
        flux_fit_params = []

        num_lines = len(wave_ctr_line)

        if num_lines == 1:
            intensity_array = np.zeros((N,2))
            central_rv_array = np.zeros((N,2))
            width_array = np.zeros((N,2))
        else:
            intensity_array = [np.zeros((N,2)) for _ in range(num_lines)]
            central_rv_array = [np.zeros((N,2)) for _ in range(num_lines)]
            width_array = [np.zeros((N,2)) for _ in range(num_lines)]

        for i in range(N):

            if observation_type == "local":
                phase = self.in_phases[i]
                profile = profiles[i]
            elif observation_type == "master":
                phase = None
                profile = profiles.copy()

            try:
                fit_prof = fit_profile(phase, profile, data_type, observation_type, model)
                profile_parameters, R2, data, y_fit, popt = fit_prof._fit(wave_ctr_line=wave_ctr_line, mask_x=mask_x, print_output=print_output)

                continuum = profile_parameters["continuum"]
                central_rv = profile_parameters["central_rv"]
                intensity = profile_parameters["intensity"]
                width = profile_parameters["width"]

                if plot_fit:
                    plot_profile_fit(data=data, y_fit=y_fit, phase=phase, data_type=data_type, observation_type=observation_type, model=model, save=save)

            except Exception as e: # if no fit is achieved
                print(f"Could not fit phase {str(phase)[:6]}")
                print(e)
                continuum = np.array([np.nan, np.nan])
                R2 = np.nan
                popt = np.array([np.nan])

                if num_lines == 1:
                    central_rv = np.array([np.nan, np.nan])
                    intensity = np.array([np.nan, np.nan])
                    width = np.array([np.nan, np.nan])
                else:
                    central_rv = [np.array([np.nan, np.nan]) for _ in range(num_lines)]
                    intensity = [np.array([np.nan, np.nan]) for _ in range(num_lines)]
                    width = [np.array([np.nan, np.nan]) for _ in range(num_lines)]
            
            # Store continuum (same for all lines)
            continuum_array[i,0], continuum_array[i,1] = continuum[0], continuum[1]
            R2_array[i] = R2
            flux_fit_params.append(popt)

            # Store line-specific parameters
            if num_lines == 1:
                central_rv_array[i,0], central_rv_array[i,1] = central_rv[0], central_rv[1]
                intensity_array[i,0], intensity_array[i,1] = intensity[0], intensity[1]
                width_array[i,0], width_array[i,1] = width[0], width[1]
            else:
                for j in range(num_lines):
                    central_rv_array[j][i,0], central_rv_array[j][i,1] = central_rv[j][0], central_rv[j][1]
                    intensity_array[j][i,0], intensity_array[j][i,1] = intensity[j][0], intensity[j][1]
                    width_array[j][i,0], width_array[j][i,1] = width[j][0], width[j][1]

        if plot_fit and observation_type == "local":
            plot_R2(self.in_phases, R2_array, threshold=0.8, save=save)
        
        return {"central_rv": central_rv_array, "continuum": continuum_array, 
                "intensity": intensity_array, "width": width_array, 
                "R2": R2_array, "flux_fit_params": flux_fit_params}
    

    def local_params_linear_fit(self, local_param:np.array, indices_final:np.array, title:str, priors:list, plot_nested:bool, axes_to_fit:list=None):
        """Performs a tentative linear fit by applying nested sampling through `dynesty, comparing between a constant and unconstrained models, and then between a linear model with a positive slope and one with a negative slope.
        Useful for a first approximation analysis of the local CCF parameters.

        Parameters
        ----------
        local_param : `numpy array`
            array of a given local CCFs parameter (central RV, width, intensity).
        indices_final : `numpy array`
            indices of local CCFs to use (to discard bad data).
        title : `str` 
            CCF parameter to use as title in the plot.
        priors : `list`
            half of range of linear fit parameters (m, b) to use as priors.
        plot_nested : `bool`
            whether to plot the trace and corner plots from the `dynesty` packages.
        axes_to_fit : `list`, optional
            List of axes to fit: ["phases", "mu"]. If None, defaults to both axes (backward compatible).

        Returns
        -------
        phases_data : `dict` or None
            contains phases ('x'), the label of phases ('label'), grid of phases for plotting ('x_grid'), fitted CCF parameter ('y_fit') as an array (value, error), grid of fitted CCF parameter ('y_grid') as an array (value, error) and 'residual' between y and y_fit as an array (value, error). None if "phases" not in axes_to_fit.
        mu_data : `dict` or None
            contains mu (x), the label of mu ('label'), grid of phases for plotting ('x_grid'), fitted CCF parameter ('y_fit') as an array (value, error), grid of fitted CCF parameter ('y_grid') as an array (value, error) and 'residual' between y and y_fit as an array (value, error). None if "mu" not in axes_to_fit.
        """
        if axes_to_fit is None:
            axes_to_fit = ["phases", "mu"]
        
        phases_data = None
        mu_data = None
        
        if "phases" in axes_to_fit:
            phases_data = {"x":self.in_phases[indices_final], "label":"Orbital phases"}
        if "mu" in axes_to_fit:
            mu_data = {"x":self.mu_in[indices_final], "label":r"$\mu$"}
        
        m_span, b_span = priors[0], priors[1]

        print("="*50)
        print(title)
        
        for data in [phases_data, mu_data]:
            
            if data is None:
                continue

            x = data["x"]
            y = local_param[:,0][indices_final]
            yerr = local_param[:,1][indices_final]

            print("-"*40)
            print(data["label"])
            
            valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(yerr)) # filter out NaN values
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            yerr_clean = yerr[valid_mask]
            
            if len(x_clean) < 3:
                print(f"Warning: Not enough valid data points for {data['label']} (only {len(x_clean)} after filtering)")
                continue
        
            results_nested = run_nestedsampler(x_clean, y_clean, yerr_clean, m_span, b_span, plot=plot_nested).results
            lin_params, model = results_nested[0], results_nested[1]

            x_grid = np.linspace(x_clean.min(), x_clean.max(), 100)

            if model == "zero":
                y_fit = lin_params["b"][0] * np.ones_like(x_clean)
                dy_fit = np.sqrt(lin_params["b"][1]**2) * np.ones_like(x_clean)
                y_grid = lin_params["b"][0] * np.ones_like(x_grid)
                dy_grid = np.sqrt(lin_params["b"][1]**2) * np.ones_like(x_grid)
            else:
                y_fit = x_clean*lin_params["m"][0] + lin_params["b"][0]
                dy_fit = np.sqrt((x_clean*lin_params["m"][1])**2 + lin_params["b"][1]**2)
                y_grid = x_grid*lin_params["m"][0] + lin_params["b"][0]
                dy_grid = np.sqrt((x_grid*lin_params["m"][1])**2 + lin_params["b"][1]**2)

            residual = y_clean - y_fit
            residual_err = np.sqrt(yerr_clean**2) # + dy_fit**2)

            data["x"] = x_clean
            data["x_grid"] = x_grid
            data["y_fit"] = np.array([y_fit, dy_fit])
            data["y_grid"] = np.array([y_grid, dy_grid])
            data["residual"] = np.array([residual, residual_err])

        return phases_data, mu_data


    def plot_local_params(self, indices_final:np.array, local_params:np.array, master_params:np.array, suptitle:str=None, linear_fit:bool=False, plot_nested:bool=False, linear_fit_pairs:list=None, save=None):
        """Plot local CCF parameters (central RV, line-width measure and line-center intensity) in function of orbital phases and mu.
        Optionally, a linear fit via nested sampling is tested for specified parameter-axis pairs, plotting the fit and the corresponding residuals.

        Parameters
        ----------
        indices_final : `numpy array`
            indices of local CCFs to use (to discard bad data).
        local_params : `numpy array`
            array of local CCFs parameters (central RV, width, intensity).
        master_params : `numpy array` 
            average out-of-transit CCF parameters.
        suptitle : `str`, optional
            title to display above the figure.
        linear_fit : `bool`
            whether to perform a tentative linear fit via nested sampling on all parameters and axes.
        plot_nested : `bool`
            whether to plot the trace and corner plots from the `dynesty` packages.
        linear_fit_pairs : `list`, optional
            List of tuples specifying which (axis, parameter_index) pairs should have linear fits.
            Example: [("phases", 0), ("mu", 1), ("phases", 2)].
            If None and linear_fit=True, fits all parameters on both axes (backward compatible).
            If specified, only those pairs will have linear fits.
        save
            path to save plots.
        """
        if linear_fit_pairs is None:
            linear_fit_pairs_set = {(axis, i) for axis in ["phases", "mu"] for i in range(3)} if linear_fit else set()
        else:
            linear_fit_pairs_set = set(linear_fit_pairs)
        
        need_linear_fit = len(linear_fit_pairs_set) > 0
        
        if need_linear_fit:
            if suptitle is not None:
                fig_ph, axes_ph = plt.subplots(nrows=2, ncols=3, figsize=(16,6.7), gridspec_kw={'height_ratios': [1.5, 1]}, constrained_layout=True)
                fig_mu, axes_mu = plt.subplots(nrows=2, ncols=3, figsize=(16,6.7), gridspec_kw={'height_ratios': [1.5, 1]}, constrained_layout=True)
                fig_ph.suptitle(suptitle, fontsize=20)
                fig_mu.suptitle(suptitle, fontsize=20)
            else:
                fig_ph, axes_ph = plt.subplots(nrows=2, ncols=3, figsize=(16,6.2), gridspec_kw={'height_ratios': [1.5, 1]})
                fig_mu, axes_mu = plt.subplots(nrows=2, ncols=3, figsize=(16,6.2), gridspec_kw={'height_ratios': [1.5, 1]})
        
        else: 
            if suptitle is not None:
                fig_ph, axes_ph = plt.subplots(nrows=1, ncols=3, figsize=(16,4.7), constrained_layout=True)
                fig_mu, axes_mu = plt.subplots(nrows=1, ncols=3, figsize=(16,4.7), constrained_layout=True)
                fig_ph.suptitle(suptitle, fontsize=20)
                fig_mu.suptitle(suptitle, fontsize=20)
            else:
                fig_ph, axes_ph = plt.subplots(nrows=1, ncols=3, figsize=(16,4.2))
                fig_mu, axes_mu = plt.subplots(nrows=1, ncols=3, figsize=(16,4.2))

        width_unit = "km/s" if self.data_type == "CCF" else r"$\AA$"
        titles = ['Central Radial Velocity [km/s]', f'Line-width measure [{width_unit}]', 'Line-center intensity [%]']
        ylabels = ["[km/s]", f"[{width_unit}]", "[%]"]

        ph_range = [-self.tr_dur/2, self.tr_dur/2]
        ph_range_inner = [self.tr_ingress_egress/2-self.tr_dur/2, self.tr_dur/2-self.tr_ingress_egress/2]
        mu_range = [0.6*self.mu_min, self.mu_max]
        mu_range_inner = [self.mu_min, self.mu_max]

        plot_data = {"phases":[axes_ph, ph_range, ph_range_inner, self.in_phases[indices_final]], "mu":[axes_mu, mu_range, mu_range_inner, self.mu_in[indices_final]]}

        for i in range(len(ylabels)):

            plot_index = (0,i) if need_linear_fit else (i)

            if need_linear_fit is None: 
                axes_ph[plot_index].set_xlabel("Orbital phases")
                axes_mu[plot_index].set_xlabel(r"$\mu$")

            for key in plot_data.keys():

                ax = plot_data[key][0]
                x_range = plot_data[key][1]
                x_range_inner = plot_data[key][2]
                x = plot_data[key][3]

                ax[plot_index].set_title(titles[i], fontsize=17)

                l0=ax[plot_index].axvspan(x_range[0], x_range[1], alpha=0.3, color='orange')
                l1=ax[plot_index].axvspan(x_range_inner[0], x_range_inner[1], alpha=0.4, color='orange')
                l2=ax[plot_index].axhline(y=master_params[i][:,0], color='blue', linestyle='-', lw=2, zorder=1)
                ax[plot_index].scatter(x, local_params[i][:,0][indices_final],color='blue',s=60,zorder=3)
                ax[plot_index].errorbar(x=x, y=local_params[i][:,0][indices_final], yerr=local_params[i][:,1][indices_final], capsize=6, capthick=0.5, color='black', linewidth=0, elinewidth=2)
                
                ax[plot_index].set_ylabel("Value " + ylabels[i], fontsize=15)
                ax[plot_index].grid()
                ax[plot_index].set_axisbelow(True)
                ax[plot_index].set_xlim(x_range)

            if ("phases", i) in linear_fit_pairs_set or ("mu", i) in linear_fit_pairs_set:
                axes_to_fit = []
                if ("phases", i) in linear_fit_pairs_set:
                    axes_to_fit.append("phases")
                if ("mu", i) in linear_fit_pairs_set:
                    axes_to_fit.append("mu")

                if i == 0: 
                    priors = [1000, 10]
                elif i == 1: 
                    priors = [100, 100]
                elif i == 2: 
                    priors = [100, 100]

                phases_data, mu_data = self.local_params_linear_fit(local_params[i], indices_final, titles[i], priors, plot_nested, axes_to_fit=axes_to_fit)

                for key in plot_data.keys():

                    if (key, i) not in linear_fit_pairs_set:
                        ax_temp = plot_data[key][0]
                        ax_temp[1,i].clear()
                        ax_temp[1,i].axis('off')
                        continue

                    ax = plot_data[key][0]
                    x_range = plot_data[key][1]
                    x_range_inner = plot_data[key][2]
                    data = phases_data if key == "phases" else mu_data
                    
                    if data is None:
                        ax[1,i].clear()
                        ax[1,i].axis('off')
                        continue

                    ax[0,i].plot(data["x"], data["y_fit"][0], color='blue', linestyle='--')
                    ax[0,i].fill_between(data["x_grid"], data["y_grid"][0]-data["y_grid"][1], data["y_grid"][0]+data["y_grid"][1], color='gray', alpha=0.2, zorder=1)
                    ax[0,i].set_xticklabels([])

                    ax[1,i].axvspan(x_range[0], x_range[1], alpha=0.3, color='orange')
                    ax[1,i].axvspan(x_range_inner[0], x_range_inner[1], alpha=0.4, color='orange')
                    ax[1,i].scatter(data["x"], data["residual"][0], color='blue', s=60, zorder=3)
                    ax[1,i].errorbar(x=data["x"], y=data["residual"][0], yerr=data["residual"][1], capsize=5, capthick=0.5, color='black', linewidth=0, elinewidth=2)
                    ax[1,i].set_xlabel(data["label"])
                    ax[1,i].grid()
                    ax[1,i].set_axisbelow(True)
                    ax[1,i].set_xlim(x_range)
                    ax[1,i].set_ylim([-2*np.max(np.abs(data["residual"][0])), 2*np.max(np.abs(data["residual"][0]))])
                    ax[1,i].axhline(0, lw=1, ls="--", color="black")
                    ax[1,i].set_ylabel("Residuals " + ylabels[i])

        labels = ['Partially in-transit','Fully in-transit','Master out of transit']
        for fig in [fig_ph, fig_mu]:
            fig.legend([l0,l1,l2], labels=labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=15)
            fig.tight_layout()

        if save:
            fig_ph.savefig(save+"local_parameters_phases.pdf", dpi=400)
            fig_mu.savefig(save+"local_parameters_mu.pdf", dpi=400)

        plt.show()


'''
    def avg_out_of_transit_CCF(self, CCFs:np.array, RV_reference:np.array, plot:bool, save=None):
        """Computes the average out-of-transit CCF by linearly interpolating the (systemic velocity corrected) CCFs into a common grid.
        The interpolated CCF uncertainties are propagated tooking the covariances into account.

        Parameters
        ----------
        CCFs : `numpy array`
            matrix with the out-of-transit CCF profiles, with shape (N_CCFs, 3, N_points).
        RV_reference : `numpy array`
            RV grid for interpolation.
        plot : `bool` 
            whether to plot the average out of transit CCF.
        save 
            path to save plot.

        Returns
        -------
        CCF_interp : `numpy array`
            matrix with interpolated CCF profiles.
        avg_out_of_transit_CCF : `numpy array`
            matrix with the average out-of-transit CCF profile, with shape (3, N_points).
        """
        M = CCFs.shape[0]
        K = CCFs.shape[2]

        cov_matrix = np.zeros((M, K, K))
        N = 10000

        # covariance matrix obtained by sampling the CCFs 10 000 times
        for i in range(M):
            samples = np.zeros((K, N))

            for j in range(K):
                ymean = CCFs[i,1,j]
                ysigma = CCFs[i,2,j]
                samples[j,:] = np.random.normal(ymean, ysigma, N)

            cov_matrix[i,:,:] = np.cov(samples)

        out_of_transit_CCFs = np.zeros([len(self.phases_out_indices), 3, len(RV_reference)])
        CCF_interp = np.zeros([CCFs.shape[0], 3, CCFs.shape[2]])

        k, M = 0, 0
        for l in range(CCF_interp.shape[0]):
            ccf_rv = CCFs[l,0]
            ccf_f = CCFs[l,1]
            ccf_f_e = cov_matrix[l]  # full covariance matrix
            
            # build interpolation matrix for this CCF → target grid
            W = linear_interpolation_matrix(ccf_rv, RV_reference) 

            y_i = W @ ccf_f # interpolated flux
            cov_new = W @ ccf_f_e @ W.T # propagated covariance
            y_i_e = np.sqrt(cov_new.diagonal()) # propagated uncertainty

            CCF_interp[l,0,:] = RV_reference
            CCF_interp[l,1,:] = y_i
            CCF_interp[l,2,:] = y_i_e

            if l in self.phases_out_indices:

                out_of_transit_CCFs[k,0,:] = RV_reference
                out_of_transit_CCFs[k,1,:] = y_i
                out_of_transit_CCFs[k,2,:] = y_i_e

                k += 1
            else:
                M += 1

        average_out_of_transit_CCF = np.mean(out_of_transit_CCFs[:,1,:], axis=0)

        A_e = np.sum(out_of_transit_CCFs[:,2,:]**2, axis=0) # propagation of uncertainty into the average CCF
        average_out_of_transit_CCF_e = np.sqrt(A_e) / len(self.phases_out_indices)

        avg_out_of_transit_CCF = np.array([RV_reference, average_out_of_transit_CCF, average_out_of_transit_CCF_e])
        
        if plot:
            plot_avg_out_of_transit_profile(avg_out_of_transit_CCF, save)

        return CCF_interp, avg_out_of_transit_CCF

        
    def avg_out_of_transit_spectra(self, spectra:np.array, wave_grid:np.ndarray, plot:bool, save=None):
        """Computes the average out-of-transit spectra by linearly interpolating the sliced spectra into a common grid.

        Parameters
        ----------
        spectra : `numpy array`
            matrix with the out-of-transit spectra, with shape (N_spectra, 3, N_points).
        wave_grid : `numpy array`
            wavelength grid for interpolation.
        plot : `bool` 
            whether to plot the average out of transit spectrum.
        save 
            path to save plot.

        Returns
        -------
        spectra_interp : `numpy array`
            matrix with interpolated spectra.
        avg_out_of_transit_spectrum : `numpy array`
            matrix with the average out-of-transit spectrum, with shape (3, N_points).
        """
        out_of_transit_spectra = np.zeros([len(self.phases_out_indices), 3, len(wave_grid)])
        spectra_interp = np.zeros([spectra.shape[0], 3, spectra.shape[2]])

        k, M = 0, 0
        for l in range(spectra.shape[0]):

            wave = spectra[l,0,:]
            flux = spectra[l,1,:]
            flux_err = diags(spectra[l,2,:])**2

            W = linear_interpolation_matrix(wave, wave_grid) 

            y_i = W @ flux # interpolated flux
            cov_new = W @ flux_err @ W.T # propagated covariance
            y_i_e = np.sqrt(cov_new.diagonal()) # propagated uncertainty

            spectra_interp[l,0,:] = wave_grid
            spectra_interp[l,1,:] = y_i
            spectra_interp[l,2,:] = y_i_e

            if l in self.phases_out_indices:
                out_of_transit_spectra[k,0,:] = wave_grid
                out_of_transit_spectra[k,1,:] = y_i
                out_of_transit_spectra[k,2,:] = y_i_e
                k += 1
            else:
                M += 1

        average_out_of_transit_flux = np.mean(out_of_transit_spectra[:,1,:], axis=0)

        A_e = np.sum(out_of_transit_spectra[:,2,:]**2, axis=0) # propagation of uncertainty into the average CCF
        average_out_of_transit_flux_e = np.sqrt(A_e) / len(self.phases_out_indices)

        avg_out_of_transit_spectrum = np.array([wave_grid, average_out_of_transit_flux, average_out_of_transit_flux_e])

        if plot:
            plot_avg_out_of_transit_profile(avg_out_of_transit_spectrum, profile_type="spectrum", save=save)

        return spectra_interp, avg_out_of_transit_spectrum
'''