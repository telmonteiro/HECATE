# File with plotting functions.

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from .utils import get_phase_mu

def plot_air_snr(planet_params:dict, time:np.array, airmass:np.array, snr:np.array, save=None):
    """Plot airmass and SNR at spectral order 111 (midpoint in the selected Fe I spectral lines) of spectra used.

    Parameters
    ----------
    planet_params : `dict`
        dictionary containing the following planetary parameters: orbital period, system scale, planet-to-star radius ratio, mid-transit time, eccentricity, argument of periastron, planetary inclination and spin-orbit angle.
    time : `numpy array`
        time of observations in BJD.
    airmass : `numpy array`
        airmass at the time of observation.
    snr : `numpy array`
        signal-to-noise ratio (SNR) at spectral order 111.
    save
        path to save plot. 
    """
    phase_mu = get_phase_mu(planet_params, time)
    phases, tr_dur, tr_ingress_egress = phase_mu.phases, phase_mu.tr_dur, phase_mu.tr_ingress_egress

    fig, ax0 = plt.subplots(figsize=(7,4.5))

    l0 = ax0.axvspan(-tr_dur/2., tr_dur/2., alpha=0.3, color='orange')
    l1 = ax0.axvspan(tr_ingress_egress/2.-tr_dur/2, -tr_ingress_egress/2.+tr_dur/2, alpha=0.4, color='orange')
    l2 = ax0.scatter(phases, airmass, color='black')

    ax0.set_xlabel('Orbital Phase', fontsize=14)
    ax0.set_ylabel('Airmass', fontsize=14)
    ax0.tick_params(axis="y")

    ax1 = ax0.twinx()
    l3 = ax1.scatter(phases, snr, color='black', marker="x")
    ax1.set_ylabel('SNR order 111', fontsize=14)
    ax1.tick_params(axis="y")

    labels = ['Partially in-transit','Fully in-transit', 'Airmass', 'SNR']
    fig.legend([l0, l1, l2, l3], labels=labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=12)

    plt.tight_layout()

    if save:
        plt.savefig(save+"airmass_snr.pdf", dpi=300, bbox_inches="tight")

    plt.show()


def plot_R2(phases:np.array, R2_array:np.array, threshold:float=0.8, save:str=None):
    """Plot coefficient of determination R² of fit(s).

    Parameters
    ----------
    phases : `numpy array`
        orbital phases.
    R2_array : `numpy array`
        coefficient of determination scores.
    threshold : `float`
        threshold to consider a fit good.
    save
        path to save plot. 
    """
    _, ax = plt.subplots(figsize=(6,4))
    ax.scatter(phases, R2_array, color="k")
    ax.axhline(y=threshold, color='black',linestyle='-')

    ax.set_title('R² of fits to profiles', fontsize=14)
    ax.set_xlabel('Orbital Phase', fontsize=15)
    ax.set_ylabel('R²', fontsize=15)
    ax.grid()
    ax.set_axisbelow(True)

    if save is not None: 
        plt.savefig(save+"R2_fits.pdf", dpi=200, bbox_inches="tight")

    plt.show()


def plot_sysvel_corr_CCF(phases:np.array, tr_dur:float, tr_ingress_egress:float, in_indices:np.array, out_indices:np.array, x0:np.array, poly_coefs:np.array, x0_corr:np.array, save=None):
    """Plot stellar systemic velocity showing the R-M effect and it's correction.

    Parameters
    ----------
    phases : `numpy array``
        orbital phases.
    tr_dur : `float`
        transit duration.
    tr_ingress_egress : `float`
        duration between ingress and egress of transit.
    in_indices : `numpy array`
        array indices where planet is in transit.
    out_indices : `numpy array`
        array indices where planet is not in transit.
    x0 : `numpy array`
        non-corrected central RVs.
    poly_coefs : `numpy array`
        linear polynomial fit coefficients.
    x0_corr : `numpy array`
        corrected central RVs.
    save
        path to save plot. 
    """
    fig, axes = plt.subplots(ncols=2, figsize=(13,5))

    l0 = axes[0].axvspan(-tr_dur/2., tr_dur/2., alpha=0.3, color='orange')
    l1 = axes[0].axvspan(tr_ingress_egress/2.-tr_dur/2, -tr_ingress_egress/2.+tr_dur/2, alpha=0.4, color='orange')
    axes[0].errorbar(phases[in_indices], x0[:,0][in_indices], x0[:,1][in_indices], fmt="r.", markersize=10, elinewidth=10)
    axes[0].errorbar(phases[out_indices], x0[:,0][out_indices], x0[:,1][out_indices], fmt="k.", markersize=10, elinewidth=10)
    l2 = axes[0].plot(phases[out_indices], poly_coefs[0]*phases[out_indices]+poly_coefs[1], color="black", lw=1)

    axes[0].set_ylabel('Radial Velocities [km/s]')
    axes[0].set_xlabel('Orbital Phases')
    axes[0].grid()
    axes[0].set_axisbelow(True)

    axes[1].axvspan(-tr_dur/2, tr_dur/2, alpha=0.3, color="orange")
    axes[1].axvspan(tr_ingress_egress/2.-tr_dur/2, -tr_ingress_egress/2.+tr_dur/2, alpha=0.4, color='orange')
    axes[1].errorbar(phases[in_indices], x0_corr[:,0][in_indices], x0_corr[:,1][in_indices], fmt="r.", markersize=10, elinewidth=10)
    axes[1].errorbar(phases[out_indices], x0_corr[:,0][out_indices], x0_corr[:,1][out_indices], fmt="k.", markersize=10, elinewidth=10)
    axes[1].axhline(0, lw=1, ls= "--", color="k")
    
    axes[1].set_ylabel('Radial Velocities [km/s]')
    axes[1].set_xlabel('Orbital Phases')
    axes[1].grid()
    axes[1].set_axisbelow(True)

    labels = ['Partially in transit','Fully in transit','out of transit linear fit']
    fig.legend([l0,l1,l2], labels=labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle('Central values of CCFs',fontsize=19)

    plt.tight_layout()

    if save:
        plt.savefig(save+"sys_vel_correction_CCF.pdf", dpi=300, bbox_inches="tight")

    plt.show()


def plot_avg_out_of_transit_profile(avg_out_of_transit_prof:np.array, profile_type:str="CCF", save:str=None):
    """Plot average out-of-transit profile, either CCF or spectral line.
    
    Parameters
    ----------
    avg_out_of_transit_prof : `numpy array`
        average out-of-transit profile (RV/wavelength, flux and flux error).
    profile_type : `str`
        whether it's a CCF or spectral line profile.
    save : `str`
        path to save plot.
    """
    _, ax = plt.subplots(figsize=(7,4))

    ax.set_title(f'Averaged out of transit {profile_type}')
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlabel(r'Wavelength [$\AA$]' if profile_type != "CCF" else 'Radial Velocities [km/s]')
    ax.set_ylabel('Normalized Flux')

    if profile_type == "CCF":
        ax.scatter(avg_out_of_transit_prof[0], avg_out_of_transit_prof[1])
        ax.errorbar(x=avg_out_of_transit_prof[0], y=avg_out_of_transit_prof[1], yerr=avg_out_of_transit_prof[2], capsize=7, capthick=1, color='black', linewidth=0, elinewidth=1)
    else:
        ax.plot(avg_out_of_transit_prof[0], avg_out_of_transit_prof[1], color='black') 

    if save: 
        plt.savefig(save+f"avg_out_of_transit_{profile_type}.pdf", dpi=200, bbox_inches="tight")

    plt.show()


def plot_profile_fit(data:np.ndarray, y_fit:np.ndarray, phase:float, data_type:str, observation_type:str, model:str, save):
    """Plot fit of spectral line or CCF profile. Four subplots: (1) observed and fitted profile; (2) residuals; (3) distribution of residuals; (4) distribution of data uncertainties.

    Parameters
    ----------
    data : `numpy array`
        CCF profile (RV, flux and flux error) or spectral line profile (wavelength, flux and flux error).
    y_fit : `numpy array`
        fitted profile flux.
    phase : `float`
        orbital phase of observation.
    data_type : `str`
        whether it's a CCF or spectral line.
    observation_type : `str`
        whether it's a local, average out-of-transit or raw CCF.
    model : `str`
        type of profile model to fit.
    save
        path to save plot. 
    """
    x = data[0]
    y = data[1]
    y_err = data[2]
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,7), gridspec_kw={'height_ratios': [1.7, 1]})
    
    if observation_type == "local":
        title = f'Local {data_type}, Model: {model}, Phase: {str(phase)[:6]}'
    elif observation_type == "master":
        title = f'Master out-of-transit {data_type}, Model: {model}'
    elif observation_type == "raw":
        title = f'Model: {model}, Phase: {str(phase)[:6]}'

    fig.suptitle(title)

    x_label = 'Radial Velocities [km/s]' if data_type == "CCF" else r'Wavelength [$\AA$]'

    axes[0,0].scatter(x, y, color="k")
    axes[0,0].errorbar(x, y, yerr=y_err, color='black', capsize=5, linewidth=0, elinewidth=1)
    axes[0,0].plot(x, y_fit, label='fit', color="r", lw=2)
    axes[0,0].set_xlabel(x_label)
    axes[0,0].set_ylabel('Flux')
    axes[0,0].grid()
    axes[0,0].set_axisbelow(True)
    axes[0,0].legend()

    axes[0,1].scatter(x, y-y_fit, color="k")
    axes[0,1].set_xlabel(x_label)
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].grid(); axes[0,1].set_axisbelow(True)

    axes[1,0].hist(y-y_fit, bins=10, edgecolor='k', color="k")
    axes[1,0].set_xlabel('Residuals')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(); axes[1,0].set_axisbelow(True)

    axes[1,1].hist(y_err, bins=10, edgecolor='k', color="k")
    axes[1,1].set_xlabel('Uncertainties')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(); axes[1,1].set_axisbelow(True)
    axes[1,1].tick_params(axis='x', which='major', labelsize=12)

    plt.tight_layout()

    if save:
        file_name = save+f"{data_type}_fit_master.pdf" if observation_type == "master" else save+f"{data_type}_fit_{str(phase)[:6]}.pdf"
        plt.savefig(file_name, dpi=300, bbox_inches="tight")

    plt.show()


def plot_local_profile(hecate, local_profiles:np.array, profiles_sub_all:np.array, profile_type:str="CCF", wave_lims:list=None, ylim_plot:list=[-0.01,0.04], line_name:str=None, photometrical_rescale:bool=False, save:str=None):
    """Plot local profiles for a given profile type (CCF or spectral line) and tomography in function of orbital phases.

    Parameters
    ----------
    hecate
        HECATE class object.
    local_profiles : `numpy array`
        local profiles (RV/wavelength, flux and flux error), with shape (N_profiles, 3, N_points).
    profiles_sub_all : `numpy array`
        all subtracted profiles (RV/wavelength, flux and flux error), with shape (N_profiles, 3, N_points).
    profile_type : `str`
        whether it's a CCF or spectral line profile.
    wave_lims : `list`
        wavelength interval to plot (only for spectral line profiles).
    ylim_plot : `list`
        y-axis limits for the plot (only for spectral line profiles).
    line_name : `str`
        name of the spectral line (only for spectral line profiles).
    photometrical_rescale : `bool`
        whether to rescale the local profiles by the photometric transit light curve.
    save : `str`
        path to save the plot.
    """
    phases = hecate.phases
    in_indices = hecate.phases_in_indices

    fig, axes = plt.subplots(nrows=2, figsize=(12,9.5), gridspec_kw={'height_ratios': [1.5, 1]})
    norm = Normalize(vmin=phases[in_indices].min(), vmax=phases[in_indices].max())
    cmap = plt.get_cmap('coolwarm_r')

    if photometrical_rescale:
        local_profiles_rescaled = local_profiles.copy()
        flux_corrections = hecate.Flux_SOAP[in_indices]
        local_profiles_rescaled[:,1] = local_profiles_rescaled[:,1] / flux_corrections[:, np.newaxis]
        local_profiles = local_profiles_rescaled

    x_0 = local_profiles[0,0] # wavelength or RV grid of first observation

    if profile_type != "CCF" and wave_lims is not None:
        mask = (x_0 >= wave_lims[0]) & (x_0 <= wave_lims[1])
    else:
        mask = np.ones(local_profiles.shape[2], dtype=bool)

    for k, idx in enumerate(in_indices):
        x     = local_profiles[k,0]
        sub   = local_profiles[k,1]
        sub_e = local_profiles[k,2]

        color = cmap(norm(phases[idx]))

        if np.allclose(sub_e, 0) or profile_type != "CCF":
            axes[0].plot(x[mask], sub[mask], color=color, lw=2)
        else:
            axes[0].scatter(x[mask], sub[mask], color=color, s=50)
            axes[0].errorbar(x[mask], sub[mask], yerr=sub_e[mask], color='black', capsize=5, linewidth=0, elinewidth=1)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    cbar1 = fig.colorbar(sm, ax=axes[0])
    cbar1.set_label('Orbital Phase')

    axes[0].set_ylabel('Residual flux [total stellar flux]')
    axes[0].grid()
    axes[0].set_axisbelow(True)
    axes[0].set_xlim([x_0[mask].min(),x_0[mask].max()])
    axes[0].set_xticklabels([])

    if profile_type != "CCF" and ylim_plot is not None:
        axes[0].set_ylim(ylim_plot)

    im = axes[1].imshow(profiles_sub_all[:,1][:,mask],
                        extent=[x_0[mask].min(), x_0[mask].max(), phases.min(), phases.max()], 
                        aspect='auto', origin='lower', cmap='jet', vmin=0)

    axes[1].axhline(-hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    axes[1].axhline(hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    axes[1].axhline(hecate.tr_ingress_egress/2 - hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    axes[1].axhline(-hecate.tr_ingress_egress/2 + hecate.tr_dur/2, lw=1.5, ls="--", color="white")

    axes[1].set_ylabel('Orbital Phase')
    
    axes[1].set_xlabel('Radial Velocities [km/s]' if profile_type == "CCF" else r'Wavelength [$\AA$]')
    axes[0].set_title(f'Local CCFs (Out-of-transit - In-transit)' if profile_type == "CCF" else f'Local {line_name} (Out-of-transit - In-transit)')  

    cbar2 = fig.colorbar(im, ax=axes[1])
    cbar2.set_label('Residual flux [total stellar flux]')

    plt.tight_layout()

    if save is not None: 
        plt.savefig(save+f"local_CCFs.pdf" if profile_type == "CCF" else save+f"local_{line_name}.pdf", dpi=300, bbox_inches="tight")

    plt.show()


'''
def plot_local_CCFs(hecate, local_CCFs:np.array, CCFs_sub_all:np.array, plot_fit:bool, save=None):
    """Plot local CCFs and tomography in function of orbital phases.

    Parameters
    ----------
    hecate
        HECATE class object.
    local_CCFs : `numpy array`
        local CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points).
    CCFs_sub_all : `numpy array`
        all subtracted CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points).
    RV_reference : `numpy array`
        RV grid for plotting.
    save
        path to save plot. 
    """
    phases = hecate.phases
    in_indices = hecate.phases_in_indices

    fig, axes = plt.subplots(nrows=2, figsize=(12,9.5), gridspec_kw={'height_ratios': [1.5, 1]})
    norm = Normalize(vmin=phases[in_indices].min(), vmax=phases[in_indices].max())
    cmap = plt.get_cmap('coolwarm_r')

    axes[0].set_title(f'Local CCFs (Out-of-transit - In-transit)')

    for k, idx in enumerate(in_indices):
        rv      = local_CCFs[k,0]
        sub     = local_CCFs[k,1]
        de_corr = local_CCFs[k,2]

        color = cmap(norm(phases[idx]))

        if plot_fit:
            axes[0].plot(rv, sub, color=color, lw=2)
        else:
            axes[0].scatter(rv, sub, color=color, s=50)
            axes[0].errorbar(rv, sub, yerr=de_corr, color='black', capsize=5, linewidth=0, elinewidth=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    cbar1 = fig.colorbar(sm, ax=axes[0])
    cbar1.set_label('Orbital Phase')

    axes[0].set_ylabel('Residual flux [total stellar flux]')
    axes[0].grid()
    axes[0].set_axisbelow(True)
    axes[0].set_xlim([rv.min(), rv.max()])
    axes[0].set_xticklabels([])

    im = axes[1].imshow(CCFs_sub_all[:,1], cmap='jet', extent=[rv.min(), rv.max(), phases.min(), phases.max()], aspect='auto', origin='lower')

    axes[1].axhline(-hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    axes[1].axhline(hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    axes[1].axhline(hecate.tr_ingress_egress/2 - hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    axes[1].axhline(-hecate.tr_ingress_egress/2 + hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    
    axes[1].set_xlabel('Radial Velocities [km/s]')
    axes[1].set_ylabel('Orbital Phase')

    cbar2 = fig.colorbar(im, ax=axes[1])
    cbar2.set_label('Residual flux [total stellar flux]')

    plt.tight_layout()

    if save: 
        plt.savefig(save+"local_CCFs.pdf", dpi=300, bbox_inches="tight")

    plt.show()


def plot_local_spectral_line(hecate, local_spectra:np.array, spectra_sub_all:np.array, wave_lims:list, line_name:str, photometrical_rescale:bool=False, ylim_plot:list=[-0.01,0.04], save=None):
    """Plot local CCFs and tomography in function of orbital phases.

    Parameters
    ----------
    hecate
        HECATE class object.
    local_spectra : `numpy array`
        local spectral line profiles (wavelength, flux and flux error), with shape (N_spectra, 3, N_points).
    spectra_sub_all : `numpy array`
        all subtracted spectral line profiles (wavelength, flux and flux error), with shape (N_spectra, 3, N_points).
    wave_lims : `list`
        wavelength interval to plot.
    line_name : `str`
        name of the spectral line.
    photometrical_rescale : `bool`
        whether to rescale the local spectral line profiles by the photometric transit light curve.
    ylim_plot : `list`
        y-axis limits for the plot.
    save
        path to save plot. 
    """
    phases = hecate.phases
    in_indices = hecate.phases_in_indices

    fig, axes = plt.subplots(nrows=2, figsize=(12,9.5), gridspec_kw={'height_ratios': [1.5, 1]})
    norm = Normalize(vmin=phases[in_indices].min(), vmax=phases[in_indices].max())
    cmap = plt.get_cmap('coolwarm_r')

    axes[0].set_title(f'Local {line_name} (Out-of-transit - In-transit)')

    mask = (local_spectra[0,0] >= wave_lims[0]) & (local_spectra[0,0] <= wave_lims[1])

    if photometrical_rescale:
        local_spectra_rescaled = local_spectra.copy()
        flux_corrections = hecate.Flux_SOAP[in_indices]
        local_spectra_rescaled[:,1] = local_spectra_rescaled[:,1] / flux_corrections[:, np.newaxis]
        local_spectra = local_spectra_rescaled

    for k, idx in enumerate(in_indices):
        sub     = local_spectra[k,1]
        #de_corr = local_spectra[k,2]

        color = cmap(norm(phases[idx]))
        axes[0].plot(local_spectra[k,0][mask], sub[mask], color=color, lw=3)
        #axes[0].errorbar(wave_grid, sub, yerr=de_corr, color='black', capsize=5, linewidth=0, elinewidth=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    cbar1 = fig.colorbar(sm, ax=axes[0])
    cbar1.set_label('Orbital Phase')

    axes[0].set_ylabel('Residual flux [total stellar flux]')
    axes[0].grid()
    axes[0].set_axisbelow(True)
    axes[0].set_xlim([local_spectra[0,0][mask].min(),local_spectra[0,0][mask].max()])
    axes[0].set_xticklabels([])

    axes[0].set_ylim(ylim_plot)

    im = axes[1].imshow(spectra_sub_all[:,1][:,mask], cmap='jet', 
                        extent=[local_spectra[0,0][mask].min(), local_spectra[0,0][mask].max(), phases.min(), phases.max()], 
                        aspect='auto', origin='lower', vmin=0)

    axes[1].axhline(-hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    axes[1].axhline(hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    axes[1].axhline(hecate.tr_ingress_egress/2 - hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    axes[1].axhline(-hecate.tr_ingress_egress/2 + hecate.tr_dur/2, lw=1.5, ls="--", color="white")

    axes[1].set_xlabel(r'Wavelength [$\AA$]')
    axes[1].set_ylabel('Orbital Phase')

    cbar2 = fig.colorbar(im, ax=axes[1])
    cbar2.set_label('Residual flux [total stellar flux]')

    plt.tight_layout()

    if save: 
        plt.savefig(save+"local_spectral_line.pdf", dpi=300, bbox_inches="tight")

    plt.show()


def plot_sysvel_corr_line(phases:np.array, tr_dur:float, tr_ingress_egress:float, out_indices:np.array, x0:np.array, rv:np.array, rv_corr:np.array, b0:np.array, wave_ctr_line:np.ndarray=[0,0], line_name:str=r"H$\alpha$", save=None):
    """Plot stellar systemic velocity showing the R-M effect and it's correction.

    Parameters
    ----------
    phases : `numpy array``
        orbital phases.
    tr_dur : `float`
        transit duration.
    tr_ingress_egress : `float`
        duration between ingress and egress of transit.
    out_indices : `numpy array`
        array indices where planet is not in transit.
    x0 : `numpy array`
        non-corrected central wavelength.
    rv : `numpy array`
        non-corrected central RVs in km/s.
    rv_corr : `numpy array`
        corrected central RVs in km/s.
    poly_coefs : `numpy array`
        linear polynomial fit coefficients.
    wave_ctr_line : `numpy array`
        central wavelength of the spectral line and its uncertainty.
    line_name : `str`
        name of the spectral line.
    save
        path to save plot. 
    """
    fig, axes = plt.subplots(nrows=3, figsize=(6, 10))
    fig.suptitle(f'Systemic velocity correction of {line_name} line')

    axes[0].scatter(phases, x0[:,0]-wave_ctr_line[0], color="black", zorder=2)
    axes[0].errorbar(x=phases, y=x0[:,0]-wave_ctr_line[0], yerr=x0[:,1], capsize=3, capthick=0.5, color='black', linewidth=0, elinewidth=1, zorder=2)
    axes[0].axvspan(-tr_dur/2, tr_dur/2, alpha=0.3, color='orange')
    axes[0].axvspan(-tr_ingress_egress/2, tr_ingress_egress/2, alpha=0.4, color='orange')
    axes[0].grid()
    axes[0].set_xlabel('Orbital Phases')
    axes[0].set_ylabel(f'Fitted central $\\lambda$ - $\\lambda_{{\\mathrm{{{line_name}}}}}$'+'\n[$\\AA$]')

    axes[1].scatter(phases, rv[:,0], color ='red', zorder=3)
    axes[1].scatter(phases[out_indices], rv[:,0][out_indices], color="black", zorder=4)
    axes[1].errorbar(x=phases, y=rv[:,0], yerr=rv[:,1], capsize=3, capthick=0.5, color='black', linewidth=0, elinewidth=1, zorder=2)
    axes[1].axvspan(-tr_dur/2, tr_dur/2, alpha=0.3, color='orange')
    axes[1].axvspan(-tr_ingress_egress/2, tr_ingress_egress/2, alpha=0.4, color='orange')
    axes[1].plot(phases[out_indices], np.zeros_like(phases[out_indices])+b0, color="black", linestyle='dashed')
    axes[1].grid()
    axes[1].set_xlabel('Orbital Phases')
    axes[1].set_ylabel('Central Radial Velocity\n[km/s]')

    axes[2].scatter(phases, rv_corr[:,0], color="red", zorder=3)
    axes[2].scatter(phases[out_indices], rv_corr[:,0][out_indices], color="black", zorder=3)
    axes[2].errorbar(x=phases, y=rv_corr[:,0], yerr=rv_corr[:,1], capsize=3, capthick=0.5, color='black', linewidth=0, elinewidth=1, zorder=2)
    axes[2].axvspan(-tr_dur/2., tr_dur/2., alpha=0.3, color='orange')
    axes[2].axvspan(-tr_ingress_egress/2., tr_ingress_egress/2., alpha=0.4, color='orange')
    axes[2].set_xlabel('Orbital Phases')
    axes[2].set_ylabel('Central Radial Velocity\n[km/s]')
    axes[2].grid()

    plt.tight_layout()

    if save:
        plt.savefig(save+"sys_vel_correction_line.pdf", dpi=300, bbox_inches="tight")

    plt.show()
'''