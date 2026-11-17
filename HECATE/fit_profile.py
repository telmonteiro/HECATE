# File with two classes: profile_models for building the model function and fit_profile for fitting the model.

import numpy as np, matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz

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
        if indices_final is not None:
            flux_fit_array = np.full((local_spectra.shape[0], 3, local_spectra.shape[2]), np.nan, dtype=float)
            flux_fit_array[:,0,:] = local_spectra[:,0,:]
        else:
            flux_fit_array = np.zeros((local_spectra.shape[0], 3, local_spectra.shape[2]))

        for i in range(len(flux_fit_params)):
            if indices_final is None or i in indices_final:
                flux_fit_array[i, 0, :] = local_spectra[i, 0, :]
                flux_fit_array[i, 1, :] = self.model(local_spectra[i, 0, :], *flux_fit_params[i])
                flux_fit_array[i, 2, :] = np.zeros_like(local_spectra[i, 2, :])

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


    def _fit(self, prior_info:dict=None, wave_ctr_line:list=[(0,0)], mask_x:np.ndarray=None, print_output:bool=False):
        """Fit a CCF or spectral line profile to observed CCF/spectral line.

        Parameters
        ----------
        prior_info : `dict`; optional
            prior information for the fit. If not provided, default priors will be used based on the data type and model type. Each key is a parameter name and its value is a dictionary with keys "p0", "lower_bound" and "upper_bound" for the initial guess, lower bound and upper bound of the parameter, respectively.
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
            try:
                x_mask = np.ones(len(self.data[0]), dtype=bool) 
            except:
                x_mask = np.ones(self.data[0].shape, dtype=bool) 
        else:
            try:
                x_mask = np.zeros(len(self.data[0]), dtype=bool)
            except:
                x_mask = np.zeros(self.data[0].shape, dtype=bool)

            for interval in mask_x:
                x_mask |= (self.data[0] >= interval[0]) & (self.data[0] <= interval[1])
        
        x = self.data[0][x_mask]
        d = self.data[1][x_mask]
        de = self.data[2][x_mask]

        num_lines = len(wave_ctr_line)
        model_fit = profile_models(self.model_type, num_lines=num_lines).model

        if self.data_type == "line":
            x0_guess = [wave_ctr_line[j][0] for j in range(num_lines)]
            x0_min = [wave_ctr_line[j][0] - 0.3 for j in range(num_lines)]
            x0_max = [wave_ctr_line[j][0] + 0.3 for j in range(num_lines)]
        else:
            x0_guess = [x[np.argmin(d)]] * num_lines
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
        
        parameter_init = {} # initialize dictionary to store the parameters priors and guesses
    
        if prior_info is not None and "y0" in prior_info:
            y0_p0 = prior_info["y0"]["p0"]
            y0_lower_bound = prior_info["y0"]["lower_bound"]
            y0_upper_bound = prior_info["y0"]["upper_bound"]
        else:
            y0_p0 = 1 if self.data_type == "line" and self.observation_type == "raw" else np.max(d)  # Use actual max flux for both CCF and line data
            y0_lower_bound = 0.7 * np.max(d)
            y0_upper_bound = 1.2 * np.max(d)
            
        parameter_init["y0"] = {"p0": y0_p0,"lower_bound": y0_lower_bound, "upper_bound": y0_upper_bound}
        
        # add parameters in the order expected by the model: for each line i: x0_i, sigma_i, a_i, c_i
        width_name = "gamma" if self.model_type == "Lorentzian" else "sigma"
 
        for i in range(num_lines):
            # x0 parameter
            param_name = f"x0{i+1}" if num_lines > 1 else "x0"
            if prior_info is not None and param_name in prior_info:
                parameter_init[param_name] = {
                    "p0": prior_info[param_name]["p0"],
                    "lower_bound": prior_info[param_name]["lower_bound"],
                    "upper_bound": prior_info[param_name]["upper_bound"]}
            else:
                x0_p0 = x0_guess[i]
                x0_lower_bound = x0_min[i]
                x0_upper_bound = x0_max[i]
                parameter_init[param_name] = {"p0": x0_p0, "lower_bound": x0_lower_bound, "upper_bound": x0_upper_bound}
            
            # sigma/gamma parameter
            if self.model_type == "Voigt":
                param_name = f"gamma{i+1}" if num_lines > 1 else "gamma"
                param_name1 = f"sigma{i+1}" if num_lines > 1 else "sigma"

                if prior_info is not None and param_name in prior_info:
                    parameter_init[param_name] = {
                        "p0": prior_info[param_name]["p0"],
                        "lower_bound": prior_info[param_name]["lower_bound"],
                        "upper_bound": prior_info[param_name]["upper_bound"]}
                else:
                    parameter_init[param_name] = {"p0": 1, "lower_bound": 0, "upper_bound": np.inf}

                if prior_info is not None and param_name1 in prior_info:
                    parameter_init[param_name1] = {
                        "p0": prior_info[param_name1]["p0"],
                        "lower_bound": prior_info[param_name1]["lower_bound"],
                        "upper_bound": prior_info[param_name1]["upper_bound"]}
                else:
                    parameter_init[param_name1] = {"p0": 1, "lower_bound": 0, "upper_bound": np.inf}
            
            else:
                param_name = f"{width_name}{i+1}" if num_lines > 1 else width_name
                
                if prior_info is not None and param_name in prior_info:
                    parameter_init[param_name] = {
                        "p0": prior_info[param_name]["p0"],
                        "lower_bound": prior_info[param_name]["lower_bound"],
                        "upper_bound": prior_info[param_name]["upper_bound"]}
                else:
                    parameter_init[param_name] = {"p0": 1, "lower_bound": 0, "upper_bound": np.inf}
            
            # amplitude parameter
            param_name = f"a{i+1}" if num_lines > 1 else "a"
            if prior_info is not None and param_name in prior_info:
                parameter_init[param_name] = {
                    "p0": prior_info[param_name]["p0"],
                    "lower_bound": prior_info[param_name]["lower_bound"],
                    "upper_bound": prior_info[param_name]["upper_bound"]}
            else:
                parameter_init[param_name] = {"p0": np.max(d) - np.min(d), "lower_bound": 0.7 * (np.max(d) - np.min(d)), "upper_bound": 1.3 * (np.max(d) - np.min(d))}

            # c parameter (only for modified Gaussian)
            if has_c:
                param_name = f"c{i+1}" if num_lines > 1 else "c"
                if prior_info is not None and param_name in prior_info:
                    parameter_init[param_name] = {
                        "p0": prior_info[param_name]["p0"],
                        "lower_bound": prior_info[param_name]["lower_bound"],
                        "upper_bound": prior_info[param_name]["upper_bound"]}
                else:
                    parameter_init[param_name] = {"p0": 1, "lower_bound": 0, "upper_bound": np.inf}

        # fit the model
        parameter_names = list(parameter_init.keys())
        p0 = [parameter_init[name]["p0"] for name in parameter_names]
        lower_bounds = [parameter_init[name]["lower_bound"] for name in parameter_names]
        upper_bounds = [parameter_init[name]["upper_bound"] for name in parameter_names]

        popt, pcov = curve_fit(f=model_fit, xdata=x, ydata=d, sigma=de,
                                bounds=(lower_bounds, upper_bounds),
                                absolute_sigma=True, p0=p0)
        
        y_fit = model_fit(x, *popt)
        data = [x, d, de]

        continuum = np.array([popt[0], np.sqrt(pcov[0, 0])])

        widths = []
        intensities = []
        central_wvs = []
        central_rvs = []
        if has_c: 
            c_values = []
        
        for i in range(num_lines):
            # parameters are ordered: y0, [x0_i, sigma_i, a_i, c_i for each i]
            x0_idx = 1 + i * (3 + int(has_c))  # 1 + i*4 for modified Gaussian, 1 + i*3 for others
            sigma_idx = 2 + i * (3 + int(has_c))
            a_idx = 3 + i * (3 + int(has_c))
            c_idx = 4 + i * (3 + int(has_c)) if has_c else None
            
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
            if has_c:
                c_values.append(np.array([popt[c_idx], np.sqrt(pcov[c_idx, c_idx])]))
            
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
                "width": widths[0],
                "c": c_values[0] if has_c else np.array([np.nan, np.nan])
                }
        else:
            profile_parameters = {"continuum": continuum}
            if self.data_type == "line":
                profile_parameters["central_wv"] = central_wvs
            profile_parameters["central_rv"] = central_rvs
            profile_parameters["intensity"] = intensities
            profile_parameters["width"] = widths
            if has_c:
                profile_parameters["c"] = c_values

        R2 = np.around(self.r2(d, y_fit), 4)

        if print_output:

            width_unit = "km/s" if self.data_type == "CCF" else r"$\AA$"

            print("#"*50)
            print(f"Fitting {self.model_type} model to {self.observation_type} {self.data_type} profile")
            
            if self.observation_type == "local":
                print(f"Phase: {str(self.phase)[:6]}")

            print("-"*50)

            print("Priors:")
            for i in range(num_lines):
                if self.data_type == "line":
                    print(f"Line {i+1} central wavelength [Å]: {wave_ctr_line[i][0]:.06f} ± {wave_ctr_line[i][1]:.06f}")
                else:
                    print(f"Line {i+1} central RV [km/s]: {x0_guess[i]:.06f} (bounds: [{x0_min[i]:.06f}, {x0_max[i]:.06f}])")
                print(f"Line {i+1} line-width measure [{width_unit}]: 1 (bounds: [0, inf])")
                print(f"Line {i+1} line-center intensity [%]: {(1 - p0[1 + i * (3 + int(has_c)) + 2] / p0[0]) * 100:.2f} (bounds: [0, inf])")
                if has_c:
                    print(f"Line {i+1} c parameter: 1 (bounds: [0, inf])")

            print("Fit parameters:")
            for j, param in enumerate(parameter_names):
                print(f"{param} = {popt[j]:.06f} ± {np.sqrt(pcov[j, j]):.06f}   (p0: {p0[j]:.06f}, bounds: [{lower_bounds[j]:.06f}, {upper_bounds[j]:.06f}])")
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
                if has_c:
                    print(f"c parameter: {c_values[0][0]:.06f} ± {c_values[0][1]:.06f}")
            else:
                for i in range(num_lines):
                    print(f"\nLine {i+1}:")
                    if self.data_type == "line":
                        print(f"  Central wavelength [Å]: {central_wvs[i][0]:.06f} ± {central_wvs[i][1]:.06f}")
                    print(f"  Central RV [km/s]: {central_rvs[i][0]:.06f} ± {central_rvs[i][1]:.06f}")
                    print(f"  Line-center intensity [%]: {intensities[i][0]:.06f} ± {intensities[i][1]:.06f}")
                    print(f"  Line-width measure [{width_unit}]: {widths[i][0]:.06f} ± {widths[i][1]:.06f}")
                    if has_c:
                        print(f"  c parameter: {c_values[i][0]:.06f} ± {c_values[i][1]:.06f}")
                print(f"\nContinuum: {continuum[0]:.06f} ± {continuum[1]:.06f}")

        return profile_parameters, R2, data, y_fit, popt
    

    def fit_poly_cont(self, poly_degree:int=2, normalize_errors:bool=False, mask_x:np.ndarray=None, mask_wing:np.ndarray=None, plot:bool=False, hybrid_norm:bool=False):
        """Fit a polynomial to the wings of the CCF/spectral line and normalize the profile by subtracting the polynomial continuum.
        
        Parameters
        ----------
        poly_degree : `int`
            degree of the polynomial to fit to the wings.
        normalize_errors : `bool`
            whether to propagate the errors from the original profile to the normalized profile (errors add in quadrature).
        mask_x : `numpy array`; optional
            list of tuples with intervals to mask x-axis (RV or wavelength) for the fit.
        mask_wing : `numpy array`; optional
            list of tuples with intervals to select the wings for the polynomial fit (if None, use all data points not masked by mask_x).
        plot : `bool`
            whether to plot the original profile with the polynomial fit and the normalized profile.
        hybrid_norm : `bool`
            if True, normalize wings by polynomial fit and center by the average polynomial value at wings (preserves CCF structure). 
            if False, normalize entire profile by the full polynomial.

        Returns
        -------
        x : `numpy array`
            x-axis values (RV or wavelength) of the profile.
        d_normalized : `numpy array`
            normalized profile flux values.
        de_normalized : `numpy array`
            normalized profile flux error values.
        poly_continuum : `numpy array`
            polynomial continuum values at each x.
        poly_coeffs : `numpy array`
            coefficients of the fitted polynomial, ordered from highest degree to lowest (as returned by `numpy`.polyfit).
        """
        if mask_x is None:
            try: x_mask = np.ones(len(self.data[0]), dtype=bool) 
            except: x_mask = np.ones(self.data[0].shape, dtype=bool) 
        else:
            try: x_mask = np.zeros(len(self.data[0]), dtype=bool)
            except: x_mask = np.zeros(self.data[0].shape, dtype=bool)

            for interval in mask_x:
                x_mask |= (self.data[0] >= interval[0]) & (self.data[0] <= interval[1])

        x = self.data[0][x_mask]
        d = self.data[1][x_mask]
        de = self.data[2][x_mask]

        if mask_wing is None:
            try: wing_mask = np.ones(len(self.data[0]), dtype=bool) 
            except: wing_mask = np.ones(self.data[0].shape, dtype=bool) 
        else:
            try: wing_mask = np.zeros(len(self.data[0]), dtype=bool)
            except: wing_mask = np.zeros(self.data[0].shape, dtype=bool)

            for interval in mask_wing:
                wing_mask |= (self.data[0] >= interval[0]) & (self.data[0] <= interval[1])

        x_wings = x[wing_mask]
        d_wings = d[wing_mask]
        de_wings = de[wing_mask]

        # fit polynomial to wings
        poly_coeffs, poly_cov = np.polyfit(x_wings, d_wings, poly_degree, w=1/de_wings**2, cov=True)

        # normalize CCF by dividing by the polynomial (keeps wings near original continuum level)
        poly_continuum = np.polyval(poly_coeffs, x)
        
        # compute error on poly_continuum using error propagation
        poly_terms = np.array([x**(poly_degree-i) for i in range(poly_degree+1)])
        poly_continuum_err = np.sqrt(np.sum((poly_terms**2) * np.diag(poly_cov)[:, None], axis=0))

        d_normalized = d / poly_continuum  
        # propagate error from original d to normalized d (errors add in quadrature)
        if normalize_errors:
            de_normalized = np.sqrt(de**2 * 1/poly_continuum**2 + poly_continuum_err**2 * d**2/poly_continuum**4)
        else:
            de_normalized = np.abs(de / poly_continuum)

        # Hybrid normalization: normalize wings by polynomial, center by average polynomial value at wings
        if hybrid_norm:
            # Build wing mask for full x array
            try: full_wing_mask = np.zeros(len(self.data[0]), dtype=bool)
            except: full_wing_mask = np.zeros(self.data[0].shape, dtype=bool)
            
            if mask_wing is not None:
                for interval in mask_wing:
                    full_wing_mask |= (self.data[0] >= interval[0]) & (self.data[0] <= interval[1])
            else:
                full_wing_mask = np.ones(len(self.data[0]), dtype=bool)
            
            # Extract wing mask for the masked x array (after x_mask filtering)
            wing_mask_here = full_wing_mask[x_mask]
            
            # Average polynomial value at wings
            poly_avg_wings = np.mean(poly_continuum[wing_mask_here])
            
            # Create hybrid continuum: polynomial for wings, constant for center
            hybrid_continuum = np.where(wing_mask_here, poly_continuum, poly_avg_wings)
            
            d_normalized = d / hybrid_continuum
            
            if normalize_errors:
                # Error propagation for hybrid case
                # Uncertainty in poly_avg_wings comes from uncertainty in poly_continuum at wings
                poly_continuum_err_wings = poly_continuum_err[wing_mask_here]
                sigma_poly_avg_wings = np.sqrt(np.sum(poly_continuum_err_wings**2)) / np.sum(wing_mask_here)
                
                de_normalized_wings = np.sqrt(de[wing_mask_here]**2 * 1/poly_continuum[wing_mask_here]**2 + 
                                             poly_continuum_err_wings**2 * d[wing_mask_here]**2/poly_continuum[wing_mask_here]**4)
                de_normalized_center = np.sqrt(de[~wing_mask_here]**2 * 1/poly_avg_wings**2 + 
                                              sigma_poly_avg_wings**2 * d[~wing_mask_here]**2/poly_avg_wings**4)
                
                de_normalized = np.concatenate([de_normalized_wings if wing_mask_here[0] else de_normalized_center,
                                               de_normalized_center if wing_mask_here[0] else de_normalized_wings])
                # Reorder to match original x order
                de_normalized_full = np.empty_like(de_normalized)
                de_normalized_full[wing_mask_here] = de_normalized_wings
                de_normalized_full[~wing_mask_here] = de_normalized_center
                de_normalized = de_normalized_full
            else:
                de_normalized = np.abs(de / hybrid_continuum)

        if plot:
            fig, axes = plt.subplots(nrows=2, figsize=(9, 8.5))
            
            axes[0].errorbar(x, d, yerr=de, fmt='k.', markersize=10)
            axes[0].plot(x, poly_continuum, label=f'Polynomial fit (degree {poly_degree})', color='red', lw=3)
            axes[0].set_xlabel('Radial Velocities [km/s]' if self.data_type == "CCF" else 'Wavelength [Å]')
            axes[0].set_ylabel('CCF Flux')
            for interval in mask_wing:
                axes[0].axvspan(interval[0], interval[1], color='yellow', alpha=0.3)
            
            axes[1].errorbar(x, d_normalized, yerr=de_normalized, fmt='k.', markersize=10)
            axes[1].set_xlabel('Radial Velocities [km/s]' if self.data_type == "CCF" else 'Wavelength [Å]')
            axes[1].set_ylabel('Normalized CCF Flux')
            for interval in mask_wing:
                axes[1].axvspan(interval[0], interval[1], color='yellow', alpha=0.3)

            axes[0].set_title(f'Polynomial Continuum Fit to {self.observation_type} {self.data_type}')
            axes[0].legend(loc="lower left")

            plt.tight_layout()
            plt.show()

        data_norm = np.array([x, d_normalized, de_normalized])

        return data_norm, poly_continuum, poly_coeffs


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
        r : float
            coefficient of determination.
        """
        ssres = np.sum((y-yfit)**2)
        sstot = np.sum((y-np.mean(y))**2)
        r = 1 - ssres/sstot
        return r