# Miscellaneous classes and functions for utility.

import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.sparse import lil_matrix

from ldtk import LDPSetCreator, BoxcarFilter

class get_limb_darkening:
    """Class to compute limb darkening coefficients and specific intensity I(mu) for a given limb darkening law using LDTK.

    Parameters
    ----------
    stellar_params : `dict`
        dictionary containing the following stellar parameters: effective temperature and error, superficial gravity and error, metallicity and error.
    planet_params : `dict`
        dictionary containing the following planetary parameters: system scale, planet-to-star radius ratio and planetary inclination.
    wav_range : `list`
        list containing the minimum and maximum wavelengths [nm] of the spectrograph.
    
    Methods
    -------
    ld_coeffs(law)
        computes limb darkening coefficients for a given law using LDTK.
    compute_I(mu, coeffs, law)
        computes the specific intensity I(mu) for a given limb darkening law and coefficients.
    get_weighted_mu(phases_in, law, plot)
        computes the brightness weighted mu values for a given set of orbital phases and limb darkening law.
    get_geometric_mu(phases_in)
        computes the geometric mu values for a given set of orbital phases.
    """
    def __init__(self, stellar_params:dict, planet_params:dict, wav_range:list):

        self.stellar_params = stellar_params
        self.planet_params = planet_params
        self.wav_range = wav_range

    def ld_coeffs(self, law:str):
        """Computes limb darkening coefficients for a given law using LDTK.

        Parameters
        ----------
        law : `str`
            limb darkening law. Supported laws: linear, quadratic, square-root, exponential, claret-nonlinear.
        
        Returns
        -------
        coeffs : `numpy array`
            limb darkening coefficients for the specified law.
        """
        filters = [BoxcarFilter('filter', self.wav_range[0], self.wav_range[1])]
        sc = LDPSetCreator(
                       teff=(self.stellar_params["Teff"], self.stellar_params["Teff_err"]),   
                       logg=(self.stellar_params["logg"], self.stellar_params["logg_err"]),
                       z=(self.stellar_params["FeH"], self.stellar_params["FeH_err"]),
                       filters=filters)

        ps = sc.create_profiles(nsamples=200)

        if law == "linear":
            coeffs, _ = ps.coeffs_ln(do_mc=True)
            return coeffs[0]
        elif law == "quadratic":
            coeffs, _ = ps.coeffs_qd(do_mc=True)
            return coeffs[0]
        elif law == "square-root":
            coeffs, _ = ps.coeffs_sq(do_mc=True)
            return coeffs[0]
        elif law == "exponential":
            coeffs, _ = ps.coeffs_p2(do_mc=True)
            return coeffs[0]
        elif law == "claret-nonlinear":
            coeffs, _ = ps.coeffs_nl(do_mc=True)
            return coeffs[0]
        else:
            raise ValueError("Unsupported limb darkening law.\nSupported laws: linear, quadratic, square-root, exponential, claret-nonlinear.")
    
    def compute_I(self, mu:np.array, coeffs:np.array, law:str):
        """Computes the specific intensity I(mu) for a given limb darkening law and coefficients.
        
        Parameters
        ----------
        mu : `numpy array`
            array of mu values (cosine of the angle between the line of sight and the normal to the stellar surface).
        coeffs : `numpy array`
            limb darkening coefficients for the specified law.
        law : `str`
            limb darkening law. Supported laws: linear, quadratic, square-root, exponential, claret-nonlinear.

        Returns
        -------
        I : `numpy array`
            specific intensity I(mu) for the given limb darkening law and coefficients.
        """
        if law == "linear":
            return 1 - coeffs[0] * (1 - mu)
        elif law == "quadratic":
            return 1 - coeffs[0] * (1 - mu) - coeffs[1] * (1 - mu)**2
        elif law == "square-root":
            return 1 - coeffs[0] * (1 - mu) - coeffs[1] * (1 - np.sqrt(mu))
        elif law == "exponential":
            return 1 - coeffs[0] * (1 - mu ** coeffs[1])
        elif law == "claret-nonlinear":
            return 1 - coeffs[0] * (1 - mu**0.5) - coeffs[1] * (1 - mu) - coeffs[2] * (1 - mu**1.5) - coeffs[3] * (1 - mu**2)
        else:
            raise ValueError("Unsupported limb darkening law.\nSupported laws: linear, quadratic, square-root, exponential, claret-nonlinear.")

    def get_weighted_mu(self, phases_in:np.array, law:str, plot:bool=False):
        """Computes the brightness weighted mu values for a given set of orbital phases and limb darkening law.

        Parameters
        ----------
        phases_in : `numpy array`
            array of in-transit orbital phases.
        law : `str`
            limb darkening law. Supported laws: linear, quadratic, square-root, exponential, claret-nonlinear.
        plot : `bool`
            whether to plot the intensity map in a given position of the planet.
        
        Returns
        -------
        weighted_mu : `numpy array`
            array of brightness weighted mu values for the given in-transit orbital phases and limb darkening law.
        """
        b = self.planet_params["a_R"] * np.cos(np.radians(self.planet_params["inc_planet"]))
        
        grid = np.linspace(-1, 1, 600) # 600 x 600 grid
        X, Y = np.meshgrid(grid, grid)

        stellar_mask = X**2 + Y**2 <= 1.0

        x_array = (self.planet_params["a_R"]*np.sin(2*np.pi*np.abs(phases_in)))
        OCCULT = [stellar_mask & ((X - x)**2 + (Y + b)**2 <= self.planet_params["Rp_Rs"]**2) for x in x_array]
        mid_idx = int(np.median(phases_in)) # example at median phase
    
        coeffs = self.ld_coeffs(law)

        mask = OCCULT[mid_idx]
        weighted_mu = np.zeros(len(OCCULT))

        for i, mask in enumerate(OCCULT):

            Z = X[mask]**2+Y[mask]**2
            MU = np.sqrt(1-Z)
            I = self.compute_I(MU, coeffs, law)

            weighted_mu[i] = np.sum(MU*I)/np.sum(I)

        if plot:
            MU_2d = np.zeros_like(X) # 2D MU array for visualization
            Z_masked = X[mask]**2 + Y[mask]**2
            MU_2d[mask] = np.sqrt(np.maximum(1 - Z_masked, 0))
            I_2d = np.zeros_like(X)
            I_2d[mask] = self.compute_I(MU_2d[mask], coeffs, law)

            plt.figure()
            plt.pcolor(X, Y, I_2d)
            plt.gca().add_patch(Circle((0, 0), 1, edgecolor='k', facecolor='none', linewidth=2))
            plt.gca().set_aspect('equal')
            plt.colorbar(label=r"$I(\mu)$"+f" ({law})")
            plt.show()

        return weighted_mu
    
    def get_geometric_mu(self, phases_in:np.array):
        """Computes the geometric mu values for a given set of orbital phases.
        
        Parameters
        ----------
        phases_in : `numpy array`
            array of in-transit orbital phases.
        
        Returns
        -------
        geometric_mu : `numpy array`
            array of geometric mu values for the given in-transit orbital phases.
        """
        inc_planet = self.planet_params["inc_planet"]
        a_R        = self.planet_params["a_R"]
        b = a_R * np.cos(np.radians(inc_planet)) # impact parameter

        geometric_mu = np.sqrt(1 - b**2 - (a_R*np.sin(2*np.pi*np.abs(phases_in)))**2)

        return geometric_mu



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
    """
    def __init__(self, time:np.array, planet_params:dict, stellar_params:dict, mu_type:str="weighted", ld_law:str="quadratic", wav_range:list=[380,788]):

        phases, tr_dur, tr_ingress_egress, in_indices, out_indices = self.get_phase(planet_params, time)
        
        self.phases = phases

        self.tr_dur = tr_dur
        self.tr_ingress_egress = tr_ingress_egress

        self.in_indices = in_indices
        self.out_indices = out_indices

        ld_class = get_limb_darkening(stellar_params, planet_params, wav_range)

        if mu_type == "weighted":
            self.mu_values = ld_class.get_weighted_mu(phases, ld_law)
            self.mu_min = ld_class.get_weighted_mu([self.tr_ingress_egress/2], ld_law)[0]
            self.mu_max = ld_class.get_weighted_mu([0], ld_law)[0]

        elif mu_type == "geometric":
            self.mu_values = ld_class.get_geometric_mu(phases)
            self.mu_min = ld_class.get_geometric_mu([self.tr_ingress_egress/2])[0]
            self.mu_max = ld_class.get_geometric_mu([0])[0]

        elif mu_type is None:
            self.mu_values = None
            self.mu_min = None
            self.mu_max = None
            
        else:
            raise ValueError("Unsupported mu type. Supported types: 'weighted' and 'geometric'.")

    def get_phase(self, planet_params:dict, time:np.array):

        t0         = planet_params["t0"]
        dfp        = planet_params["dfp"]
        P_orb      = planet_params["P_orb"]
        inc_planet = np.radians(planet_params["inc_planet"])
        Rp_Rs      = planet_params["Rp_Rs"]
        a_R        = planet_params["a_R"]

        t_epoch = t0 + 0.5+2.4e6 - dfp*P_orb  # MBJD
        norb = (time-t_epoch)/P_orb
        nforb = [round(x) for x in norb]
        phases = norb-nforb

        tr_dur = 1/np.pi * np.arcsin(1/a_R *np.sqrt((1+Rp_Rs)**2 - a_R**2 * np.cos(inc_planet)**2))
        tr_ingress_egress = 1/np.pi * np.arcsin(1/a_R *np.sqrt((1-Rp_Rs)**2 -a_R**2 * np.cos(inc_planet)**2))

        in_indices  = np.where(np.abs(phases) <= tr_dur/2)[0]
        out_indices = np.where(np.abs(phases) >  tr_dur/2)[0]

        return phases, tr_dur, tr_ingress_egress, in_indices, out_indices



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

    return W.tocsr()