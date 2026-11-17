# Runs nested sampler on a linear regression.

import numpy as np
import matplotlib.pyplot as plt

from dynesty import DynamicNestedSampler, NestedSampler, plotting as dyplot
from dynesty.utils import resample_equal


class run_nestedsampler:
    """Nested sampler for comparing linear models. Choose between static or dynamic sampler.
    
    Compares between linear models with constant slope (m=0) or unconstrained slope (m!=0) 
    regarding CCF parameters in function of orbital phases or mu.
    
    Includes optional jittering via ln f parameter, with prior [-20,18] in log-scale.

    Parameters
    ----------
    x : `numpy array`
        orbital phases or mu arrays.
    y : `numpy array`
        CCF parameter (central RV, line-width measure or line-center intensity).
    yerr : `numpy array`
        uncertainty of CCF parameter.
    m_span : `int`
        half of the range of prior on the slope.
    b_span : `int`
        half of the range of prior on the intercept.
    verbose : `bool`
        print output.
    plot : `bool`
        whether to plot the trace and corner plots from the `dynesty` packages.
    save
        path to save plots.
    sampler_type : str
        Either "static" or "dynamic" nested sampler. Default is "dynamic".
    jitter : bool
        Whether to include jitter term (ln_f parameter) in the model. Default is True.
    """
    
    def __init__(self, x:np.array, y:np.array, yerr:np.array, m_span:int=100, b_span:int=100, verbose:bool=True, plot:bool=True, save=None, sampler_type:str="dynamic", jitter:bool=False):
        
        if len(x) != len(y) or len(x) != len(yerr):
            raise ValueError("x, y, yerr must have equal length")
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            raise ValueError("x contains NaN or inf values")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or inf values")
        if np.any(np.isnan(yerr)) or np.any(np.isinf(yerr)):
            raise ValueError("yerr contains NaN or inf values")
        
        if sampler_type.lower() not in ["static", "dynamic"]:
            raise ValueError(f"Unknown sampler_type: {sampler_type}. Use 'static' or 'dynamic'.")
        if not isinstance(jitter, bool):
            raise ValueError(f"jitter must be a boolean, got {type(jitter)}")

        self.x = x
        self.y = y
        self.yerr = yerr
        self.m_span = m_span
        self.b_span = b_span
        self.sampler_type = sampler_type.lower()
        self.jitter = jitter

        resA, resB, resPos, resNeg, model, logK_linear_vs_const, logK_sign, logZ = self.compare_models(x, y, yerr, m_span, b_span)

        if model == "zero":
            result = resB
        elif model == "positive":
            result = resPos
        elif model == "negative":
            result = resNeg

        weights = np.exp(result.logwt - result.logz[-1])
        samples = result.samples
        posterior_samples = resample_equal(samples, weights)
        
        linear_fit_params = {}
        labels = self._get_labels(model)

        if verbose: 
            print("Linear fit parameters:")

        for i, label in enumerate(labels):
            q50 = np.percentile(posterior_samples[:, i], 50)
            q16 = np.percentile(posterior_samples[:, i], 16)
            q84 = np.percentile(posterior_samples[:, i], 84)
            linear_fit_params[label] = [q50, np.mean([q84-q50,q50-q16])]
            if verbose:
                print(f"{label} = {q50:.06f} +/- {np.mean([q84-q50,q50-q16]):.06f}")

        print("-"*30)
                
        if plot:
            fig, _ = dyplot.traceplot(result, labels=labels, fig=plt.subplots(len(labels), 2, figsize=(8, 6)))
            fig.tight_layout()
            if save: 
                fig.savefig(save+"traceplot.pdf", dpi=300)
            
            fig, _ = dyplot.cornerplot(result, show_titles=True, labels=labels, fig=plt.subplots(len(labels), len(labels), figsize=(7, 7)), quantiles=(0.16,0.5,0.84), title_quantiles=[0.16,0.5,0.84])
            if save: 
                fig.savefig(save+"cornerplot.pdf", dpi=300)
        
        self.results = (linear_fit_params, model, logK_linear_vs_const, logK_sign, logZ)
    
    def _get_labels(self, model):
        """Get parameter labels based on model type and jitter setting."""
        if self.jitter:
            if model != "zero":
                return ["m", "b", "ln_f"]
            else:
                return ["b", "ln_f"]
        else:  # no jitter
            if model != "zero":
                return ["m", "b"]
            else:
                return ["b"]
    
    def _get_ndim_linear(self):
        """Get dimensionality for linear model."""
        return 3 if self.jitter else 2
    
    def _get_ndim_constant(self):
        """Get dimensionality for constant model."""
        return 2 if self.jitter else 1

    def compare_models(self, x:np.array, y:np.array, yerr:np.array, m_span:int=100, b_span:int=100, verbose:bool=True):
        """Compares between an unconstrained and a constant model. If unconstrained is preferred, 
        compares a model with positive slope vs negative slope.
        
        Parameters
        ----------
        x : `numpy array`
            orbital phases or mu arrays.
        y : `numpy array`
            CCF parameter (central RV, line-width measure or line-center intensity).
        yerr : `numpy array`
            uncertainty of CCF parameter.
        m_span : `int`
            half of the range of prior on the slope.
        b_span : `int`
            half of the range of prior on the intercept.
        verbose : `bool`
            print output.

        Returns
        -------
        resA
            dynesty results for initial linear model (m!=0).
        resB
            dynesty results for constant model (m=0).
        resPos
            dynesty results for linear model with positive slope (m>0).
        resNeg 
            dynesty results for linear model with negative slope (m<0).
        model : `str`
            type of model obtained.
        logK_linear_vs_const : `float`
            evidence difference between constant and unconstrained models.
        logK_sign : `float`
            evidence difference between positive and negative models.
        logZ : `float`
            evidence of final model.
        """
        log_Bayes_factor_threshold = 2.3  # corresponds to "strong" evidence in Jeffreys' scale
        
        # 1) Linear model m,b
        like_A = lambda th: self.loglike_linear(th, x, y, yerr)
        pt_A = lambda u: self.ptform_linear(u, m_span, b_span)
        resA = self.run_dynesty(like_A, pt_A, self._get_ndim_linear())

        # 2) Constant model (m=0)
        like_B = lambda th: self.loglike_constant(th, x, y, yerr)
        pt_B = lambda u: self.ptform_constant(u, b_span)
        resB = self.run_dynesty(like_B, pt_B, self._get_ndim_constant())

        logZ_A, logZerr_A = resA.logz[-1], resA.logzerr[-1]
        logZ_B, logZerr_B = resB.logz[-1], resB.logzerr[-1]
        logK_linear_vs_const = logZ_A - logZ_B

        if verbose:
            print("-"*30)
            print("Linear vs Constant")
            print(f"logZ(linear)   = {logZ_A:.3f} ± {logZerr_A:.3f}")
            print(f"logZ(constant) = {logZ_B:.3f} ± {logZerr_B:.3f}")
            print(f"log Bayes factor = {logK_linear_vs_const:.3f}")
        if logK_linear_vs_const > log_Bayes_factor_threshold:
            if verbose:
                print("Unconstrained model favored.")
                print("-"*30)
        else:
            if verbose:
                print("Zero-slope model favored")
                print("========================")
            model = "zero"
            return resA, resB, None, None, model, logK_linear_vs_const, None, logZ_B
        
        #if linear is favored: compare sign of slope
        #positive slope
        pt_pos = lambda u: self.ptform_positive_slope(u, m_span, b_span)
        resPos = self.run_dynesty(lambda th: self.loglike_linear(th, x, y, yerr), pt_pos, self._get_ndim_linear())
        #negative slope
        pt_neg = lambda u: self.ptform_negative_slope(u, m_span, b_span)
        resNeg = self.run_dynesty(lambda th: self.loglike_linear(th, x, y, yerr), pt_neg, self._get_ndim_linear())

        logZ_pos, logZerr_pos = resPos.logz[-1], resPos.logzerr[-1]
        logZ_neg, logZerr_neg = resNeg.logz[-1], resNeg.logzerr[-1]
        logK_sign = logZ_neg - logZ_pos

        if logK_sign > log_Bayes_factor_threshold:
            model = "negative"
            logZ = logZ_neg
        else:
            model = "positive"
            logZ = logZ_pos

        if verbose:
            print("Positive vs Negative slope")
            print("===========================")
            print(f"logZ(m>0)  = {logZ_pos:.3f} ± {logZerr_pos:.3f}")
            print(f"logZ(m<0)  = {logZ_neg:.3f} ± {logZerr_neg:.3f}")
            print(f"log Bayes factor = {logK_sign:.3f}")
            if logK_sign > log_Bayes_factor_threshold:
                print("Negative slope favored.")
            else:
                print("Positive slope favored.")
            print("==========================")

        return resA, resB, resPos, resNeg, model, logK_linear_vs_const, logK_sign, logZ

    def run_dynesty(self, loglike, ptform, ndim):
        """Run the appropriate nested sampler based on sampler_type."""
        if self.sampler_type == "dynamic":
            dsampler = DynamicNestedSampler(loglike, ptform, ndim, nlive=1000)
        else:  # static
            dsampler = NestedSampler(loglike, ptform, ndim, nlive=1000)
        dsampler.run_nested()
        return dsampler.results

    def loglike_linear(self, theta, x, y, yerr):
        """Log-likelihood for linear model."""
        if self.jitter:
            m, b, lnf = theta
            model = m * x + b
            inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2 * lnf))
            return -0.5 * (np.sum((y-model)**2 * inv_sigma2 - np.log(inv_sigma2)))
        else:  # no jitter
            m, b = theta
            model = m * x + b
            inv_sigma2 = 1.0 / (yerr**2)
            return -0.5 * (np.sum((y-model)**2 * inv_sigma2))

    def loglike_constant(self, theta, x, y, yerr):
        """Log-likelihood for constant model."""
        if self.jitter:
            b, lnf = theta
            model = b
            inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2 * lnf))
            return -0.5 * (np.sum((y-model)**2 * inv_sigma2 - np.log(inv_sigma2)))
        else:  # no jitter
            b, = theta
            model = b
            inv_sigma2 = 1.0 / (yerr**2)
            return -0.5 * (np.sum((y-model)**2 * inv_sigma2))

    def ptform_linear(self, utheta, m_span, b_span):
        """Prior transform for linear model."""
        if self.jitter:
            um, ub, ulf = utheta
            m = -m_span + 2*m_span * um  
            b = -b_span + 2*b_span * ub 
            lnf = -32 + 10*ulf 
            return m, b, lnf
        else:  # no jitter
            um, ub = utheta
            m = -m_span + 2*m_span * um  
            b = -b_span + 2*b_span * ub 
            return m, b

    def ptform_constant(self, utheta, b_span):
        """Prior transform for constant model."""
        if self.jitter:
            ub, ulf = utheta 
            b = -b_span + 2*b_span * ub 
            lnf = -32 + 10*ulf
            return b, lnf
        else:  # no jitter
            ub, = utheta
            b = -b_span + 2*b_span * ub 
            return b,

    def ptform_positive_slope(self, utheta, m_span, b_span):
        """Prior transform for positive slope model."""
        if self.jitter:
            um, ub, ulf = utheta
            m = 0 + m_span * um 
            b = -b_span + 2*b_span * ub
            lnf = -32 + 10*ulf 
            return m, b, lnf
        else:  # no jitter
            um, ub = utheta
            m = 0 + m_span * um 
            b = -b_span + 2*b_span * ub
            return m, b

    def ptform_negative_slope(self, utheta, m_span, b_span):
        """Prior transform for negative slope model."""
        if self.jitter:
            um, ub, ulf = utheta
            m = -m_span + m_span * um
            b = -b_span + 2*b_span * ub
            lnf = -32 + 10*ulf
            return m, b, lnf
        else:  # no jitter
            um, ub = utheta
            m = -m_span + m_span * um
            b = -b_span + 2*b_span * ub
            return m, b