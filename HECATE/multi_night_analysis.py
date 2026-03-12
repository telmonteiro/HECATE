# Multi-night analysis module for HECATE.
# Aggregate and compare profile parameters across multiple observation nights.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from HECATE.nested_sampling import run_nestedsampler


class multi_night_analysis:
    """Aggregate and analyze profile parameters across multiple observation nights.
    This class enables comparison of local CCF/spectral line parameters (RV, width, intensity)
    across different observation nights, with flexible fitting and visualization options.
    
    Parameters
    ----------
    nights_data : `dict`
        Dictionary mapping night identifiers to data dictionaries. Each night dict should contain:
        {
            'hecate': HECATE_instance,
            'indices': indices_array (good data indices),
            'local_params': local_params_array (shape: 3, N, 2),
            'master_params': master_params_array (shape: 3, 1, 2),
            'color': optional_matplotlib_color,
            'label': optional_label_string
        }

    data_type : `str`
        Type of data being analyzed, either 'CCF' or 'line'.
    """
    def __init__(self, nights_data:dict, data_type:str='CCF'):
        
        self.nights_data = nights_data
        self.night_names = list(nights_data.keys())
        self.data_type = data_type
        
        n_nights = len(self.night_names)
        cmap = cm.get_cmap('tab10')
        
        for i, (night, data) in enumerate(nights_data.items()):
            if 'color' not in data:
                data['color'] = cmap(i / max(1, n_nights - 1))
            if 'label' not in data:
                data['label'] = str(night)
    
    
    def plot_parameters(self, param_type:str='phases', fit_each_night:bool=False, fit_combined:bool=False, combined_night_names:np.array=None, fit_param_indices:np.array=None, plot_nested:bool=False, suptitle:str=None, save=None):
        """Plot profile parameters from all nights with optional linear fits.
        
        Parameters
        ----------
        param_type : `str`
            'phases' or 'mu' - which x-axis to plot against.
        fit_each_night : `bool`
            Whether to fit each night individually.
        fit_combined : `bool`
            Whether to fit combined nights.
        combined_night_names : `numpy array`, optional
            List of night names to combine for fitting. If None and fit_combined=True, uses all nights.
        fit_param_indices : `numpy array`, optional
            Parameter indices to fit (0, 1, 2). If None and fit_each_night or fit_combined is True, fits all.
        plot_nested : `bool`
            Whether to plot Dynesty trace/corner plots.
        suptitle : `str`, optional
            Figure title.
        save
            Path to save plots.

        Returns
        -------
        fit_results : `dict`
            Dictionary with fit results keyed by (night, param_idx, param_type).
        """        
        combined_label = None
        if fit_combined and combined_night_names is None: # use all nights
            combined_night_names = self.night_names
        
        if combined_night_names is not None:
            combined_label = '+'.join(combined_night_names)
        
        need_fits = fit_each_night or fit_combined
        
        # default fit_param_indices to all if fits are requested
        if need_fits and fit_param_indices is None:
            fit_param_indices = [0, 1, 2]

        width_unit = "km/s" if self.data_type == "CCF" else r"$\AA$"
        titles = ['Central Radial Velocity [km/s]', f'Line-width measure [{width_unit}]', 'Line-center intensity [%]']
        ylabels = ["[km/s]", f"[{width_unit}]", "[%]"]
        
        n_rows = 2 if need_fits else 1
        fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(16, 7.5 if need_fits else 5.5),
                                 gridspec_kw={'height_ratios': [1.5, 1]} if need_fits else {},
                                 constrained_layout=True if suptitle else False)
        
        if suptitle:
            fig.suptitle(suptitle, fontsize=20)
        
        fit_results = {}
        
        # calculate global mu span for consistent x-limits
        global_mu_min = None
        global_mu_max = None
        if param_type == 'mu':
            mu_all = np.array([])
            for night in self.night_names:
                hecate = self.nights_data[night]['hecate']
                indices = self.nights_data[night]['indices']
                mu_all = np.concatenate([mu_all, hecate.mu_in[indices]])
            
            if len(mu_all) > 0:
                global_mu_min = np.nanmin(mu_all)
                global_mu_max = np.nanmax(mu_all)
        
        legend_lines = []
        legend_labels = []
        
        for param_idx in range(3): # rv, width, intensity
            ax_idx = (0, param_idx) if need_fits else (param_idx,)
            axes[ax_idx].set_title(titles[param_idx], fontsize=17)
            
            if not need_fits:
                axes[ax_idx].set_xlabel("Orbital phases" if param_type == "phases" else r"$\mu$", fontsize=16)

            axes[ax_idx].set_ylabel("Value " + ylabels[param_idx], fontsize=16)
            axes[ax_idx].grid()
            axes[ax_idx].set_axisbelow(True)
            
            if param_type == 'phases':
                x_range = [-self.nights_data[self.night_names[0]]['hecate'].tr_dur/2, self.nights_data[self.night_names[0]]['hecate'].tr_dur/2]
                x_range_inner = [self.nights_data[self.night_names[0]]['hecate'].tr_ingress_egress/2 - self.nights_data[self.night_names[0]]['hecate'].tr_dur/2, 
                                 self.nights_data[self.night_names[0]]['hecate'].tr_dur/2 - self.nights_data[self.night_names[0]]['hecate'].tr_ingress_egress/2]
            else:
                if global_mu_min is None or global_mu_max is None: # first night's bounds as fallback
                    hecate_first = self.nights_data[self.night_names[0]]['hecate']
                    x_range = [0.6*hecate_first.mu_min, 1.1*hecate_first.mu_max]
                    x_range_inner = [hecate_first.mu_min, hecate_first.mu_max]
                else:
                    x_range = [0.65*global_mu_min, 1.05*global_mu_max]
                    x_range_inner = [global_mu_min, global_mu_max]
            
            for night in self.night_names:

                data = self.nights_data[night]
                hecate = data['hecate']
                indices = data['indices']
                local_params = data['local_params']
                master_params = data['master_params']
                
                if param_type == 'phases':
                    x = hecate.in_phases[indices]
                else:
                    x = hecate.mu_in[indices]
                
                if night == self.night_names[0]:
                    l0 = axes[ax_idx].axvspan(x_range[0], x_range[1], alpha=0.3, color='orange')
                    l1 = axes[ax_idx].axvspan(x_range_inner[0], x_range_inner[1], alpha=0.4, color='orange')
                
                l2 = axes[ax_idx].axhline(y=master_params[param_idx, 0, 0], color=data['color'], linestyle='-', lw=2, zorder=1, label=f'Master OoT {data["label"]}')
                
                # plot night's data
                l3 = axes[ax_idx].scatter(x, local_params[param_idx, :, 0][indices], color=data['color'], s=60, zorder=3, label=data['label']+' values')
                
                if param_idx == 0:
                    legend_lines.append(l2)
                    legend_labels.append(l2.get_label())
                    legend_lines.append(l3)
                    legend_labels.append(l3.get_label())

                axes[ax_idx].errorbar(x=x, y=local_params[param_idx, :, 0][indices], yerr=local_params[param_idx, :, 1][indices],
                                     capsize=6, capthick=0.5, color='black', linewidth=0, elinewidth=2)
                
            axes[ax_idx].set_xlim(x_range)

            if fit_each_night and param_idx in fit_param_indices:

                for night in self.night_names:
                    data = self.nights_data[night]
                    hecate = data['hecate']
                    indices = data['indices']
                    local_params = data['local_params']
                    
                    x = hecate.in_phases[indices] if param_type == 'phases' else hecate.mu_in[indices]
                    y = local_params[param_idx, :, 0][indices]
                    yerr = local_params[param_idx, :, 1][indices]
                    
                    fit_data = self._fit_parameter(x, y, yerr, param_idx, plot_nested)
                    
                    if fit_data is None:
                        continue
                    
                    key = (night, param_idx, param_type)
                    fit_results[key] = fit_data
                    
                    axes[ax_idx].plot(fit_data['x'], fit_data['y_fit'][0], color=data['color'], linestyle='--', linewidth=2, zorder=2)
                    axes[ax_idx].fill_between(fit_data['x_grid'], fit_data['y_grid'][0] - fit_data['y_grid'][1],
                                            fit_data['y_grid'][0] + fit_data['y_grid'][1], color=data['color'], alpha=0.15, zorder=0)
                    
            if fit_combined and combined_night_names and param_idx in fit_param_indices: # combine data from specified nights
                
                x_combined = np.array([])
                y_combined = np.array([])
                yerr_combined = np.array([])
                
                for night in combined_night_names:
                    data = self.nights_data[night]
                    hecate = data['hecate']
                    indices = data['indices']
                    local_params = data['local_params']
                    
                    x = hecate.in_phases[indices] if param_type == 'phases' else hecate.mu_in[indices]
                    x_combined = np.concatenate([x_combined, x])
                    y_combined = np.concatenate([y_combined, local_params[param_idx, :, 0][indices]])
                    yerr_combined = np.concatenate([yerr_combined, local_params[param_idx, :, 1][indices]])
                
                fit_data = self._fit_parameter(x_combined, y_combined, yerr_combined, param_idx, plot_nested)
                
                if fit_data is not None:
                    key = (combined_label, param_idx, param_type)
                    fit_results[key] = fit_data
                    
                    axes[ax_idx].plot(fit_data['x'], fit_data['y_fit'][0], color='black', linestyle='--', linewidth=2, zorder=2)
                    axes[ax_idx].fill_between(fit_data['x_grid'], fit_data['y_grid'][0] - fit_data['y_grid'][1],
                                             fit_data['y_grid'][0] + fit_data['y_grid'][1], color='gray', alpha=0.25, zorder=2)
            
            if need_fits and param_idx in fit_param_indices: # add residuals subplot only for fitted parameters
                ax_res = (1, param_idx)
                axes[ax_res].set_xlabel("Orbital phases" if param_type == "phases" else r"$\mu$")
                axes[ax_res].set_ylabel("Residuals " + ylabels[param_idx])
                axes[ax_res].grid()
                axes[ax_res].set_axisbelow(True)
                axes[ax_res].axhline(0, lw=1, ls="--", color="black")
                axes[ax_res].set_xlim(x_range)
                axes[ax_res].axvspan(x_range[0], x_range[1], alpha=0.3, color='orange')
                axes[ax_res].axvspan(x_range_inner[0], x_range_inner[1], alpha=0.4, color='orange')
                
                for night in self.night_names:
                    if (night, param_idx, param_type) in fit_results:
                        fit_data = fit_results[(night, param_idx, param_type)]
                        axes[ax_res].scatter(fit_data['x'], fit_data['residual'][0], color=self.nights_data[night]['color'], s=60, zorder=3, alpha=0.7)
                        axes[ax_res].errorbar(x=fit_data['x'], y=fit_data['residual'][0], yerr=fit_data['residual'][1],
                                             capsize=5, capthick=0.5, color="black", linewidth=0, elinewidth=2, alpha=0.7)
                
                if combined_label and (combined_label, param_idx, param_type) in fit_results:
                    fit_data = fit_results[(combined_label, param_idx, param_type)]
                    axes[ax_res].scatter(fit_data['x'], fit_data['residual'][0], color='black', s=60, zorder=3, marker='s', alpha=0.7)
                    axes[ax_res].errorbar(x=fit_data['x'], y=fit_data['residual'][0], yerr=fit_data['residual'][1],
                                         capsize=5, capthick=0.5, color='black', linewidth=0, elinewidth=2, alpha=0.7)
                
                axes[ax_res].set_xlim(x_range)
            
            elif need_fits:  # hide residuals subplot for unfitted parameters
                axes[(1, param_idx)].axis('off')
        
        legend_lines = [l0, l1] + legend_lines
        labels = ['Partially in-transit','Fully in-transit'] + legend_labels
        
        fig.legend(legend_lines, labels=labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), fontsize=15)
        fig.tight_layout(rect=[0, 0.12, 1, 1])

        if save:
            fig.savefig(f"{save}multi_night_parameters_{param_type}.pdf", dpi=400)
        
        plt.show()

        return fit_results
    
    
    def _fit_parameter(self, x:np.array, y:np.array, yerr:np.array, param_idx:int, plot_nested:bool=False):
        """Perform linear fit on a parameter using nested sampling.
        
        Parameters
        ----------
        x : `numpy array`
            Independent variable (orbital phases or mu values).
        y : `numpy array`
            Parameter values to fit.
        yerr : `numpy array`
            Parameter uncertainties/errors.
        param_idx : `int`
            Parameter index (0=RV, 1=width, 2=intensity) for setting prior ranges.
        plot_nested : `bool`
            Whether to plot Dynesty trace and corner plots for diagnostic purposes.
        
        Returns
        -------
        fit_results : `dict` or None
            Dictionary containing:
            - 'x': cleaned input x values
            - 'x_grid': regular grid for smooth predictions
            - 'y_fit': array [values, uncertainties] at data points
            - 'y_grid': array [values, uncertainties] on grid
            - 'residual': array [residuals, errors]
            - 'slope', 'slope_err': linear slope and uncertainty
            - 'intercept', 'intercept_err': intercept and uncertainty
            - 'model': 'linear' or 'zero' (constant)
            
            Returns None if insufficient valid data points (< 3).
        """
        if param_idx == 0:
            m_span, b_span = 1000, 10  # wider priors for RV
        else:
            m_span, b_span = 100, 100  # narrower for width/intensity
        
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(yerr)) # filter out NaN values
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        yerr_clean = yerr[valid_mask]
        
        if len(x_clean) < 3:
            return None
        
        results_nested = run_nestedsampler(x_clean, y_clean, yerr_clean, m_span, b_span, plot=plot_nested).results
        lin_params, model = results_nested[0], results_nested[1]
        
        x_grid = np.linspace(x_clean.min(), x_clean.max(), 100)
        
        if model == "zero":
            y_fit = lin_params["b"][0] * np.ones_like(x_clean)
            dy_fit = np.sqrt(lin_params["b"][1]**2) * np.ones_like(x_clean)
            y_grid = lin_params["b"][0] * np.ones_like(x_grid)
            dy_grid = np.sqrt(lin_params["b"][1]**2) * np.ones_like(x_grid)
            slope, slope_err = 0, 0
            intercept, intercept_err = lin_params["b"][0], lin_params["b"][1]
        else:
            y_fit = x_clean * lin_params["m"][0] + lin_params["b"][0]
            dy_fit = np.sqrt((x_clean * lin_params["m"][1])**2 + lin_params["b"][1]**2)
            y_grid = x_grid * lin_params["m"][0] + lin_params["b"][0]
            dy_grid = np.sqrt((x_grid * lin_params["m"][1])**2 + lin_params["b"][1]**2)
            slope, slope_err = lin_params["m"][0], lin_params["m"][1]
            intercept, intercept_err = lin_params["b"][0], lin_params["b"][1]
        
        residual = y_clean - y_fit
        residual_err = np.sqrt(yerr_clean**2)

        fit_results = {"x": x_clean,
                        "x_grid": x_grid,
                        "y_fit": np.array([y_fit, dy_fit]),
                        "y_grid": np.array([y_grid, dy_grid]),
                        "residual": np.array([residual, residual_err]),
                        "slope": slope,
                        "slope_err": slope_err,
                        "intercept": intercept,
                        "intercept_err": intercept_err,
                        "model": model}
        
        return fit_results
