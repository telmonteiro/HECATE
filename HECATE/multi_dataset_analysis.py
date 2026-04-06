# Multi-data set analysis module for HECATE.
# Aggregate and compare profile parameters across multiple observation nights.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from HECATE.nested_sampling import run_nestedsampler


class multi_dataset_analysis:
    """Aggregate and analyze profile parameters across multiple observation nights or data sets.
    This class enables comparison of local CCF/spectral line parameters (RV, width, intensity)
    across different observation nights, with flexible fitting and visualization options.
    
    Parameters
    ----------
    data_sets : `dict`
        Dictionary mapping data set identifiers to data dictionaries. Each data set dict should contain:
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
    def __init__(self, data_sets:dict, data_type:str='CCF'):
        
        self.data_sets = data_sets
        self.data_set_names = list(data_sets.keys())
        self.data_type = data_type
        
        n_data_sets = len(self.data_set_names)
        cmap = cm.get_cmap('tab10')
        
        for i, (data_set, data) in enumerate(data_sets.items()):
            if 'color' not in data:
                data['color'] = cmap(i / max(1, n_data_sets - 1))
            if 'label' not in data:
                data['label'] = str(data_set)


    def plot_master_oot_difference(self):
        """Plot difference between master out-of-transit profiles across data sets.
        """
        avg_oot_profiles = {}

        for data_set, data in self.data_sets.items():
            if self.data_type == "CCF":
                avg_oot_profile = data['hecate'].local_data['avg_out_of_transit_CCF']
            else:
                avg_oot_profile = data['hecate'].local_data['avg_out_of_transit_spectrum']

            avg_oot_profiles[data_set] = avg_oot_profile
        
        # one subplot per each combination of data sets. if 2 data-sets, 1 subplot. if 3 data-sets, 3 subplots (A-B, A-C, B-C)
        # 3 data sets = grid 1x2. 4 data sets = 1x2. 5 data sets = 2x2. 6 data sets = 2x3. 7 data sets = 3x3 (last 2 empty). 8 data sets = 3x3 (last empty). 9 data sets = 3x3.
        data_set_names = list(avg_oot_profiles.keys())
        n_data_sets = len(data_set_names)
        n_cols = min(2, n_data_sets - 1)
        n_comparisons = n_data_sets * (n_data_sets - 1) // 2
        n_rows = (n_comparisons + n_cols - 1) // n_cols

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6*n_cols, 4 if n_rows == 1 else 4.5*n_rows), constrained_layout=True)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        idx = 0
        for i in range(n_data_sets):
            for j in range(i+1, n_data_sets):
                data_set_i = data_set_names[i]
                data_set_j = data_set_names[j]

                profile_i = avg_oot_profiles[data_set_i]
                profile_j = avg_oot_profiles[data_set_j]

                diff_profile = profile_j[1] - profile_i[1]

                axes[idx].scatter(profile_i[0], diff_profile, color='black')
                axes[idx].errorbar(profile_i[0], diff_profile, yerr=np.sqrt(profile_i[2]**2+profile_j[2]**2), color='black', capsize=5, linewidth=0, elinewidth=1)
                
                axes[idx].set_title(f"{data_set_j} - {data_set_i}", fontsize=16)
                axes[idx].set_xlabel("Radial Velocity [km/s]" if self.data_type == "CCF" else r"Wavelength [$\AA$]", fontsize=14)
                axes[idx].set_ylabel("Relative Flux", fontsize=14)

                axes[idx].axhline(0, color='black', linestyle='--', linewidth=1)
                idx += 1
        
        if idx < len(axes):
            for k in range(idx, len(axes)):
                axes[k].axis('off')
        
        plt.show()
            
    
    
    def plot_parameters(self, param_type:str='phases', fit_each:bool=False, fit_combined:bool=False, combined_dataset_names:np.array=None, fit_param_indices:np.array=None, plot_nested:bool=False, suptitle:str=None, save=None):
        """Plot profile parameters from all data sets with optional linear fits.
        
        Parameters
        ----------
        param_type : `str`
            'phases' or 'mu' - which x-axis to plot against.
        fit_each : `bool`
            Whether to fit each data set individually.
        fit_combined : `bool`
            Whether to fit combined data sets.
        combined_dataset_names : `numpy array`, optional
            List of data set names to combine for fitting. If None and fit_combined=True, uses all data sets.
        fit_param_indices : `numpy array`, optional
            Parameter indices to fit (0, 1, 2). If None and fit_each or fit_combined is True, fits all.
        plot_nested : `bool`
            Whether to plot Dynesty trace/corner plots.
        suptitle : `str`, optional
            Figure title.
        save
            Path to save plots.

        Returns
        -------
        fit_results : `dict`
            Dictionary with fit results keyed by (data_set, param_idx, param_type).
        """        
        combined_label = None
        if fit_combined and combined_dataset_names is None: # use all data sets
            combined_dataset_names = self.data_set_names
        
        if combined_dataset_names is not None:
            combined_label = '+'.join(combined_dataset_names)
        
        need_fits = fit_each or fit_combined
        
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
                x_range = [-self.data_sets[self.data_set_names[0]]['hecate'].tr_dur/2, self.data_sets[self.data_set_names[0]]['hecate'].tr_dur/2]
                x_range_inner = [self.data_sets[self.data_set_names[0]]['hecate'].tr_ingress_egress/2, 
                                 - self.data_sets[self.data_set_names[0]]['hecate'].tr_ingress_egress/2]
            else:
                hecate_first = self.data_sets[self.data_set_names[0]]['hecate']
                x_range = [np.nanmin(hecate_first.mu), hecate_first.mu_max]
                x_range_inner = [hecate_first.mu_min, hecate_first.mu_max]
            
            for night in self.data_set_names:

                data = self.data_sets[night]
                hecate = data['hecate']
                indices = data['indices']
                local_params = data['local_params']
                master_params = data['master_params']
                
                if param_type == 'phases':
                    x = hecate.in_phases[indices]
                else:
                    x = hecate.mu_in[indices]
                
                if night == self.data_set_names[0]:
                    l0 = axes[ax_idx].axvspan(x_range[0], x_range[1], alpha=0.3, color='orange')
                    l1 = axes[ax_idx].axvspan(x_range_inner[0], x_range_inner[1], alpha=0.4, color='orange')
                
                l2 = axes[ax_idx].axhline(y=master_params[param_idx, 0, 0], color=data['color'], linestyle='-', lw=2, zorder=1, label=f'Master OoT {data["label"]}')
                
                # plot night's data
                l3 = axes[ax_idx].scatter(x, local_params[param_idx, :, 0][indices], color=data['color'], s=55, zorder=3, label=data['label']+' values')
                
                if param_idx == 0:
                    legend_lines.append(l2)
                    legend_labels.append(l2.get_label())
                    legend_lines.append(l3)
                    legend_labels.append(l3.get_label())

                axes[ax_idx].errorbar(x=x, y=local_params[param_idx, :, 0][indices], yerr=local_params[param_idx, :, 1][indices],
                                     capsize=5, capthick=0.5, color='black', linewidth=0, elinewidth=2)
                
            axes[ax_idx].set_xlim(x_range)

            if fit_each and param_idx in fit_param_indices:

                for night in self.data_set_names:
                    data = self.data_sets[night]
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
                    
            if fit_combined and combined_dataset_names and param_idx in fit_param_indices: # combine data from specified data sets
                
                x_combined = np.array([])
                y_combined = np.array([])
                yerr_combined = np.array([])
                
                for dataset in combined_dataset_names:
                    data = self.data_sets[dataset]
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
                                             fit_data['y_grid'][0] + fit_data['y_grid'][1], color='gray', alpha=0.35, zorder=2)
            
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
                
                for night in self.data_set_names:
                    if (night, param_idx, param_type) in fit_results:
                        fit_data = fit_results[(night, param_idx, param_type)]
                        axes[ax_res].scatter(fit_data['x'], fit_data['residual'][0], color=self.data_sets[night]['color'], s=60, zorder=3)
                        axes[ax_res].errorbar(x=fit_data['x'], y=fit_data['residual'][0], yerr=fit_data['residual'][1],
                                             capsize=5, capthick=0.5, color="black", linewidth=0, elinewidth=2,zorder=2)
                
                if combined_label and (combined_label, param_idx, param_type) in fit_results:
                    fit_data = fit_results[(combined_label, param_idx, param_type)]
                    axes[ax_res].scatter(fit_data['x'], fit_data['residual'][0], color='black', s=60, zorder=3, marker='s')
                    axes[ax_res].errorbar(x=fit_data['x'], y=fit_data['residual'][0], yerr=fit_data['residual'][1],
                                         capsize=5, capthick=0.5, color='black', linewidth=0, elinewidth=2,zorder=2)
                
                axes[ax_res].set_xlim(x_range)
            
            elif need_fits:  # hide residuals subplot for unfitted parameters
                axes[0,param_idx].set_xlabel("Orbital phases" if param_type == "phases" else r"$\mu$")
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
        fit_data : `dict` or None
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
            m_span, b_span = 2000, 1000  # wider priors for RV
        else:
            m_span, b_span = 2000, 1000  # narrower for width/intensity
        
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
        
        return {"x": x_clean,
            "x_grid": x_grid,
            "y_fit": np.array([y_fit, dy_fit]),
            "y_grid": np.array([y_grid, dy_grid]),
            "residual": np.array([residual, residual_err]),
            "slope": slope,
            "slope_err": slope_err,
            "intercept": intercept,
            "intercept_err": intercept_err,
            "model": model}