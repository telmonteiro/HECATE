# Miscellaneous classes and functions for utility.

import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import curve_fit
from scipy.special import wofz


class get_phase_mu:
    """Collect orbital parameters and information, including orbital phases, mu, transit duration, duration between ingress and egress, array indices of in-transit and out-of-transit, 

    Parameters
    ----------
    planet_params : `dict`
        dictionary containing the following planetary parameters: orbital period, system scale, planet-to-star radius ratio, mid-transit time, eccentricity, argument of periastron, planetary inclination and spin-orbit angle.
    time : `numpy array`
        time of observations in BJD.

    Methods
    -------
    get_phase(planet_params, time)
        computes orbital phases, transit duration, time between ingress and egress, array indices in-transit and out-of-transit.
    mu(phases, planet_params)
        computes mu.
    """
    def __init__(self, planet_params:dict, time:np.array):

        phases, tr_dur, tr_ingress_egress, in_indices, out_indices = self.get_phase(planet_params, time)
        
        self.phases = phases

        self.tr_dur = tr_dur
        self.tr_ingress_egress = tr_ingress_egress

        self.in_indices = in_indices
        self.out_indices = out_indices

        mu_values = self.mu(phases, planet_params)
        self.mu_values = mu_values

    def get_phase(self, planet_params:dict, time:np.array):

        t0         = planet_params["t0"]
        dfp        = planet_params["dfp"]
        P_orb      = planet_params["P_orb"]
        inc_planet = np.radians(planet_params["inc_planet"])
        Rp_Rs      = planet_params["Rp_Rs"]
        a_R        = planet_params["a_R"]

        t_epoch = t0 + 0.5+2.4e6 - dfp*P_orb  #MBJD
        norb = (time-t_epoch)/P_orb
        nforb = [round(x) for x in norb]
        phases = norb-nforb

        tr_dur = 1/np.pi * np.arcsin(1/a_R *np.sqrt((1+Rp_Rs)**2 - a_R**2 * np.cos(inc_planet)**2))
        tr_ingress_egress = 1/np.pi * np.arcsin(1/a_R *np.sqrt((1-Rp_Rs)**2 -a_R**2 * np.cos(inc_planet)**2))

        in_indices  = np.where(np.abs(phases) <= tr_dur/2)[0]
        out_indices = np.where(np.abs(phases) >  tr_dur/2)[0]

        return phases, tr_dur, tr_ingress_egress, in_indices, out_indices

    @staticmethod
    def mu(phases:np.array, planet_params:dict):

        inc_planet = planet_params["inc_planet"]
        a_R        = planet_params["a_R"]

        b = a_R*np.cos(inc_planet*np.pi/180) # impact parameter

        return np.sqrt(1 - b**2 - (a_R*np.sin(2*np.pi*np.abs(phases)))**2)
    


# linear interpolation taking into account covariances
def linear_interpolation_matrix(x_old, x_new):
    """Builds a sparse matrix W that linearly interpolates data from x_old → x_new.
    Each row i corresponds to interpolation weights for x_new[i].

    Parameters
    ----------
    x_old, x_new : `numpy array`
        original and interpolated arrays.

    Returns
    -------
    W : `numpy array`
        linear interpolation matrix.
    """
    W = lil_matrix((len(x_new), len(x_old)))

    for i, xv in enumerate(x_new):
        if xv <= x_old[0]: #extrapolate using first two points
            j = 0
            x0, x1 = x_old[j], x_old[j+1]
            w1 = (x1 - xv) / (x1 - x0)
            w2 = (xv - x0) / (x1 - x0)
        elif xv >= x_old[-1]: #extrapolate using last two points
            j = len(x_old) - 2
            x0, x1 = x_old[j], x_old[j+1]
            w1 = (x1 - xv) / (x1 - x0)
            w2 = (xv - x0) / (x1 - x0)
        else:
            j = np.searchsorted(x_old, xv) - 1
            x0, x1 = x_old[j], x_old[j+1]
            w1 = (x1 - xv) / (x1 - x0)
            w2 = (xv - x0) / (x1 - x0)

        W[i, j]   = w1
        W[i, j+1] = w2

    W = W.tocsr()

    return W



class profile_models:
    """Spectral line/CCF profile models. Models available: modified Gaussian, Gaussian and Lorentzian.

    Parameters
    ----------
    model : `str`
        type of profile model to fit.
    num_lines : `int`
        number of lines to fit (1 for single line, 2 for doublet).

    Methods
    -------
    _build_model(model_type, num_lines)
        builds the model function for curve fitting based on the specified model type and number of lines.
    _convert_to_fit(local_spectra, flux_fit_params, indices_final)
        converts the fitted parameters into a flux array for plotting.
    """
    def __init__(self, model_type:str, num_lines:int=1):

        self.model_type = model_type
        self.num_lines = num_lines
        self.model = self._build_model(model_type, num_lines)

    def _build_model(self, model_type:str, num_lines:int):
        """Builds the model function for curve fitting based on the specified model type and number of lines.

        Parameters
        ----------
        model_type : `str`
            type of profile model to fit.
        num_lines : `int`
            number of lines to fit (1 for single line, 2 for doublet).
        
        Returns
        -------
        model_func : `function`
            model function to be used for curve fitting.
        """
        if model_type == "modified Gaussian":
            def model_func(x, *params):
                y0 = params[0]
                result = np.zeros_like(x, dtype=float)
                for i in range(num_lines):
                    idx = 1 + i * 4  # x0, sigma, a, c indices
                    x0, sigma, a, c = params[idx:idx+4]
                    result += a * np.exp(-0.5*((np.abs(x-x0)/sigma)**c))
                return y0 - result
            
        elif model_type == "Gaussian":
            def model_func(x, *params):
                y0 = params[0]
                result = np.zeros_like(x, dtype=float)
                for i in range(num_lines):
                    idx = 1 + i * 3  # x0, sigma, a indices
                    x0, sigma, a = params[idx:idx+3]
                    result += a * np.exp(-0.5*((np.abs(x-x0)/sigma)**2))
                return y0 - result
            
        elif model_type == "Lorentzian":
            def model_func(x, *params):
                y0 = params[0]
                result = np.zeros_like(x, dtype=float)
                for i in range(num_lines):
                    idx = 1 + i * 3  # x0, gamma, a indices
                    x0, gamma, a = params[idx:idx+3]
                    result += a * (gamma**2 / ((x - x0)**2 + gamma**2))
                return y0 - result
        
        elif model_type == "Voigt":
            def model_func(x, *params):
                y0 = params[0]
                result = np.zeros_like(x, dtype=float)
                for i in range(num_lines):
                    idx = 1 + i * 4  # x0, sigma, gamma, a indices
                    x0, sigma, gamma, a = params[idx:idx+4]
                    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
                    result += a * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
                return y0 - result
        
        return model_func
    
    def _convert_to_fit(self, local_spectra:np.array, flux_fit_params:np.array, indices_final:list=None):
        """Convert the fitted parameters into a flux array for plotting.

        Parameters
        ----------
        local_spectra : `numpy array`
            local spectra array (wavelength, flux and flux error).
        flux_fit_params : `numpy array`
            fitted profile parameters for each local spectrum.
        indices_final : `list`; optional
            list of indices to keep in the final flux fit array (in-transit indices), if None, keep all.

        Returns
        -------
        flux_fit_array : `numpy array`
            array with the same shape as local_spectra (wavelength, fitted flux, zeros).
        """
        flux_fit_array = np.zeros((local_spectra.shape[0], 3, local_spectra.shape[2]))

        for i in range(len(flux_fit_params)):
            flux_fit_array[i, 0, :] = local_spectra[i, 0, :]
            flux_fit_array[i, 1, :] = self.model(local_spectra[i, 0, :], *flux_fit_params[i])
            flux_fit_array[i, 2, :] = np.zeros_like(local_spectra[i, 2, :])

        if indices_final is not None:
            flux_fit_array_masked = np.full_like(flux_fit_array, np.nan, dtype=float)
            flux_fit_array_masked[indices_final] = flux_fit_array[indices_final]
            flux_fit_array_masked[:,0,:] = local_spectra[:,0,:]
            flux_fit_array = flux_fit_array_masked.copy()

        return flux_fit_array



class fit_profile:
    """Fit a CCF or spectral line profile to observed CCF/spectral line.

    Parameters
    ----------
    phase : `float`
        orbital phase.
    data : `numpy array`
        CCF profile (RV, flux and flux error) or spectral line profile (wavelength, flux and flux error).
    data_type : `str`
        whether it's a CCF or spectral line profile.
    observation_type : `str`
        whether it's a local, average out-of-transit or raw profile.
    model_type : `str`
        type of profile model to fit.

    Methods
    -------
    fit_profile(wave_ctr_line, mask_x, print_output)
        fits the specified profile model to the observed CCF/spectral line and returns the fitted profile parameters, coefficient of determination, profile data and fitted profile for plotting.   
    r2(y, yfit)
        computes coefficient of determination.
    """
    def __init__(self, phase:float, data:np.array, data_type:str="CCF", observation_type:str="raw", model_type:str="modified Gaussian"):

        self.phase = phase
        self.data = data

        self.data_type = data_type
        self.observation_type = observation_type
        self.model_type = model_type


    def _fit(self, wave_ctr_line:list=[(0,0)], mask_x:np.ndarray=None, print_output:bool=False):
        """Fit a CCF or spectral line profile to observed CCF/spectral line.

        Parameters
        ----------
        wave_ctr_line : `list`; optional
            central wavelength of the spectral line(s) (wavelength, uncertainty).
        mask_x : `numpy array`; optional
            list of tuples with intervals to mask x-axis (RV or wavelength).
        print_output : `bool` 
            whether to print the output.

        Returns
        -------
        profile_parameters : `dict`
            fitted profile parameters: central wavelength (if spectral line), central RV, continuum, line-center intensity and line-width measure.
        R2 : `numpy array` 
            coefficient of determination of fit.
        data : `list`
            profile data for plotting.
        y_fit : `numpy array`
            fitted profile for plotting.
        """
        c = 299792.458 #km/s

        if mask_x is None:
            x_mask = np.ones(len(self.data[0]), dtype=bool) 
        
        else:
            x_mask = np.zeros(len(self.data[0]), dtype=bool)
            for interval in mask_x:
                x_mask |= (self.data[0] >= interval[0]) & (self.data[0] <= interval[1])
        
        x = self.data[0][x_mask]
        d = self.data[1][x_mask]
        de = self.data[2][x_mask]

        num_lines = len(wave_ctr_line)
        model_fit = profile_models(self.model_type, num_lines=num_lines).model

        if self.data_type == "line":
            x0_guess = [wave_ctr_line[i][0] for i in range(num_lines)]
            x0_min = [wave_ctr_line[i][0] - 0.1 for i in range(num_lines)]
            x0_max = [wave_ctr_line[i][0] + 0.1 for i in range(num_lines)]
        else:
            x0_guess = [0] * num_lines
            x0_min = [np.min(x)] * num_lines
            x0_max = [np.max(x)] * num_lines

        model_config = {
            "modified Gaussian": {"width_mult": 1, "has_c": True},
            "Gaussian": {"width_mult": 2*np.sqrt(2*np.log(2)), "has_c": False},
            "Lorentzian": {"width_mult": 2, "has_c": False},
            "Voigt": {"width_mult": None, "has_c": False}}
        
        config = model_config[self.model_type]
        width_multiplier = config["width_mult"]
        has_c = config["has_c"]
        
        parameter_names = ["y0"]
        p0 = [1] if self.data_type == "line" and self.observation_type == "raw" else [np.max(d)]  # Use actual max flux for both CCF and line data
        lower_bound = [0]
        upper_bound = [np.inf]
        
        # add parameters in the order expected by the model: for each line i: x0_i, sigma_i, a_i, c_i
        width_name = "gamma" if self.model_type == "Lorentzian" else "sigma"
 
        for i in range(num_lines):
            # x0 parameter
            param_name = f"x0{i+1}" if num_lines > 1 else "x0"
            parameter_names.append(param_name)
            p0.append(x0_guess[i])
            lower_bound.append(x0_min[i])
            upper_bound.append(x0_max[i])
            
            # sigma/gamma parameter
            if self.model_type == "Voigt":
                param_name = f"gamma{i+1}" if num_lines > 1 else "gamma"
                param_name1 = f"sigma{i+1}" if num_lines > 1 else "sigma"
                parameter_names.append(param_name)
                parameter_names.append(param_name1)
                p0.append(1)
                p0.append(1)
                lower_bound.append(0)
                lower_bound.append(0)
                upper_bound.append(np.inf)
                upper_bound.append(np.inf)
            else:
                param_name = f"{width_name}{i+1}" if num_lines > 1 else width_name
                parameter_names.append(param_name)
                p0.append(1)
                lower_bound.append(0)
                upper_bound.append(np.inf)
            
            # amplitude parameter
            param_name = f"a{i+1}" if num_lines > 1 else "a"
            parameter_names.append(param_name)
            p0.append(np.max(d) - np.min(d))
            lower_bound.append(0)
            upper_bound.append(np.inf)
            
            # c parameter (only for modified Gaussian)
            if has_c:
                param_name = f"c{i+1}" if num_lines > 1 else "c"
                parameter_names.append(param_name)
                p0.append(1)
                lower_bound.append(0)
                upper_bound.append(np.inf)

        # fit the model
        popt, pcov = curve_fit(f=model_fit, xdata=x, ydata=d, sigma=de, 
                            bounds=(lower_bound, upper_bound), absolute_sigma=True, p0=p0)
        y_fit = model_fit(x, *popt)
        data = [x, d, de]

        continuum = np.array([popt[0], np.sqrt(pcov[0, 0])])

        widths = []
        intensities = []
        central_wvs = []
        central_rvs = []
        
        for i in range(num_lines):
            # parameters are ordered: y0, [x0_i, sigma_i, a_i, c_i for each i]
            x0_idx = 1 + i * (3 + int(has_c))  # 1 + i*4 for modified Gaussian, 1 + i*3 for others
            sigma_idx = 2 + i * (3 + int(has_c))
            a_idx = 3 + i * (3 + int(has_c))
            
            if width_multiplier is None: # Voigt profile: use sigma and gamma to compute an effective width
                a_idx = 4 + i * (3 + int(has_c))
                gamma_idx = 3 + i * (3 + int(has_c))

                sigma = popt[sigma_idx]
                gamma = popt[gamma_idx]
                
                effective_width = 0.5343 * gamma + np.sqrt(0.2169 * gamma**2 + sigma**2) # Olivero, J. J.; Longbothum, R. L. (February 1977)
                
                width = np.array([effective_width,
                                np.sqrt((0.5343 + 0.2169 * gamma / np.sqrt(0.2169 * gamma**2 + sigma**2))**2 * pcov[gamma_idx, gamma_idx] + 
                                    (sigma / np.sqrt(0.2169 * gamma**2 + sigma**2))**2 * pcov[sigma_idx, sigma_idx])])
            else:
                width = np.array([width_multiplier * popt[sigma_idx],
                            width_multiplier * np.sqrt(pcov[sigma_idx, sigma_idx])])
            
            intensity = np.array([(1 - popt[a_idx] / popt[0]) * 100,
                                (popt[a_idx] / popt[0] * np.sqrt(np.abs(pcov[a_idx, a_idx]) / popt[a_idx]**2 + 
                                                                np.abs(pcov[0, 0]) / popt[0]**2)) * 100])
            widths.append(width)
            intensities.append(intensity)
            
            if self.data_type == "line":
                central_wv = np.array([popt[x0_idx], np.sqrt(pcov[x0_idx, x0_idx])])
                central_rv = np.array([(popt[x0_idx] - wave_ctr_line[i][0]) * c/wave_ctr_line[i][0],
                                    c * popt[x0_idx]/wave_ctr_line[i][0] * np.sqrt((np.sqrt(pcov[x0_idx, x0_idx])/popt[x0_idx])**2 + (wave_ctr_line[i][1]/wave_ctr_line[i][0])**2)])
                central_wvs.append(central_wv)
                central_rvs.append(central_rv)
            else:
                central_rvs.append(np.array([popt[x0_idx], np.sqrt(pcov[x0_idx, x0_idx])]))

        if num_lines == 1:
            profile_parameters = {
                "central_wv": central_wvs[0] if self.data_type == "line" else np.array([0, 0]),
                "central_rv": central_rvs[0],
                "continuum": continuum,
                "intensity": intensities[0],
                "width": widths[0]}
        else:
            profile_parameters = {"continuum": continuum}
            if self.data_type == "line":
                profile_parameters["central_wv"] = central_wvs
            profile_parameters["central_rv"] = central_rvs
            profile_parameters["intensity"] = intensities
            profile_parameters["width"] = widths

        R2 = np.around(self.r2(d, y_fit), 4)

        if print_output:

            width_unit = "km/s" if self.data_type == "CCF" else r"$\AA$"

            print("#"*50)
            print(f"Fitting {self.model_type} model to {self.observation_type} {self.data_type} profile")
            if self.observation_type == "local":
                print(f"Phase: {str(self.phase)[:6]}")
            print("-"*50)
            print("Fit parameters:")
            for j, param in enumerate(parameter_names):
                print(f"{param} = {popt[j]:.06f} ± {np.sqrt(pcov[j, j]):.06f}")
            print("R^2: ", R2)
            print("-"*50)
            print("Profile parameters:")
            
            if num_lines == 1:
                if self.data_type == "line":
                    print(f"Central wavelength [Å]: {central_wvs[0][0]:.06f} ± {central_wvs[0][1]:.06f}")
                print(f"Central RV [km/s]: {central_rvs[0][0]:.06f} ± {central_rvs[0][1]:.06f}")
                print(f"Continuum: {continuum[0]:.06f} ± {continuum[1]:.06f}")
                print(f"Line-center intensity [%]: {intensities[0][0]:.06f} ± {intensities[0][1]:.06f}")
                print(f"Line-width measure [{width_unit}]: {widths[0][0]:.06f} ± {widths[0][1]:.06f}")
            else:
                for i in range(num_lines):
                    print(f"\nLine {i+1}:")
                    if self.data_type == "line":
                        print(f"  Central wavelength [Å]: {central_wvs[i][0]:.06f} ± {central_wvs[i][1]:.06f}")
                    print(f"  Central RV [km/s]: {central_rvs[i][0]:.06f} ± {central_rvs[i][1]:.06f}")
                    print(f"  Line-center intensity [%]: {intensities[i][0]:.06f} ± {intensities[i][1]:.06f}")
                    print(f"  Line-width measure [{width_unit}]: {widths[i][0]:.06f} ± {widths[i][1]:.06f}")
                print(f"\nContinuum: {continuum[0]:.06f} ± {continuum[1]:.06f}")

        return profile_parameters, R2, data, y_fit, popt

    @staticmethod
    def r2(y, yfit):
        """Compute coefficient of determination.

        Parameters
        ----------
        y : `numpy array`
            observed flux.
        yfit : `numpy array`
            fitted flux.

        Returns
        -------
        r : `float`
            coefficient of determination.
        """
        ssres = np.sum((y-yfit)**2)
        sstot = np.sum((y-np.mean(y))**2)
        r = 1 - ssres/sstot

        return r
    

"""
# Doppler shift wavelength
def doppler_shift(wavelength, RV):
    c = 299792.458 #km/s
    doppler_factor = np.sqrt((1+RV/c) / (1 - RV/c))
    return wavelength * doppler_factor

"""