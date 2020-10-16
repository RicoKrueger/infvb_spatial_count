from joblib import Parallel, delayed
import os
import sys
import time
import numpy as np
import pandas as pd
from numba import jit
import pypolyagamma as pypolyagamma
import h5py
from scipy.sparse.linalg import expm
from scipy.stats import invwishart
from scipy.special import loggamma, digamma, betainc, hyp2f1
import bayestree as bt
from scipy.stats import rankdata
from scipy.spatial import distance_matrix

class Data:
    """ Holds the training data.
    
    Attributes:
        y (array): dependent count data.
        N (int): number of dependent data.
        x_fix (array): covariates pertaining to fixed link function parameters.
        n_fix (int): number of covariates pertaining to fixed link function 
        parameters.
        x_rnd (array): covariates pertaining to random link function parameters.
        n_rnd (int): number of covariates pertaining to random link function 
        parameters.
        W (array): row-normalised spatial weight matrix.
    """   
    
    def __init__(self, y, x_fix, x_rnd, offset, W):
        self.y = y
        self.N = y.shape[0]
        
        self.x_fix = x_fix
        if x_fix is not None:
            self.n_fix = x_fix.shape[1]
        else:
            self.n_fix = 0
            
        self.x_rnd = x_rnd
        if x_rnd is not None:
            self.n_rnd = x_rnd.shape[1] 
        else:
            self.n_rnd = 0
            
        if offset is not None:
            self.offset = offset
        else:
            self.offset = 0
            
        self.W = W
        

class OptionsMcmc:
    """ Contains options for MCMC algorithm.
    
    Attributes:
        model_name (string): name of the model to be estimated.
        nChain (int): number of Markov chains.
        nBurn (int): number of samples to be discarded for burn-in. 
        nSample (int): number of samples after burn-in period.
        nIter (int): total number of samples.
        nThin (int): thinning factors.
        nKeep (int): number of samples to be retained.
        nMem (int): number of samples to retain in memory.
        disp (int): number of samples after which progress is displayed. 
        mh_step_initial (float): Initial MH step size.
        mh_target (float): target MH acceptance rate.
        mh_correct (float): correction of MH step to reach desired target 
        acceptance rate.
        mh_window (int): number of samples after which to adjust MG step size.
        delete_draws (bool): Boolean indicating whether simulation draws should
        be deleted.
        seed (int): random seed.
    """   

    def __init__(
            self, 
            model_name='test',
            nChain=1, nBurn=500, nSample=500, nThin=2, nMem=None, disp=100, 
            mh_step_initial=0.1, mh_target=0.3, mh_correct=0.01, mh_window=50,
            delete_draws=True, seed=4711
            ):
        self.model_name = model_name
        self.nChain = nChain
        self.nBurn = nBurn
        self.nSample = nSample
        self.nIter = nBurn + nSample
        self.nThin = nThin
        self.nKeep = int(nSample / nThin)
        if nMem is None:
            self.nMem = self.nKeep
        else:
            self.nMem = nMem
        self.disp = disp
        
        self.mh_step_initial = mh_step_initial
        self.mh_target = mh_target
        self.mh_correct = mh_correct
        self.mh_window = mh_window
        
        self.delete_draws = delete_draws
        self.seed = seed
        
class OptionsVb:
    """ Contains options for VB algorithm.
    
    Attributes:
        model_name (string): name of the model to be estimated.
        max_iter (int): Max. number of iterations.
        tol (real): Tolerance of stopping criterion.
        psi_quad_nodes (int): Number of quadrature nodes for integration over 
        variational distribution of psi_i.
        tau_quad_nodes (int): Number of quadrature nodes for integration over 
        variational distribution of tau.
        r_sim_draws (int): Number of simulation draws for integration over
        variational distribution of r. 
    """   

    def __init__(
            self, 
            model_name='test',
            max_iter=500, tol=0.005,
            infvb_n_jobs=-1,
            psi_quad_nodes=10,
            tau_quad_nodes=10,
            r_sim_draws = 1000,
            infvb_sim_draws = 10000
            ):
        self.model_name = model_name
        self.max_iter = max_iter
        self.tol = tol
        self.infvb_n_jobs = infvb_n_jobs
        self.psi_quad_nodes = psi_quad_nodes
        self.tau_quad_nodes = tau_quad_nodes
        self.r_sim_draws = r_sim_draws
        self.infvb_sim_draws = infvb_sim_draws
        
class ResultsMcmc:
    """ Holds MCMC simulation results.
    
    Attributes:
        options (OptionsMcmc): Options used for MCMC algorithm.
        bart_options (BartOptions): Options used for BART component.
        estimation time (float): Estimation time.
        lppd (float): Log pointwise predictive density.
        waic (float): Widely applicable information criterion.
        post_ls (DataFrame): Posterior summary of log-score.
        post_dss (DataFrame): Posterior summary of Dawid-Sebastiani score.
        post_rps (DataFrame): Posterior summary of ranked prob. score.
        post_mean_lam (array): Posterior mean of the expected count of each 
        observation.
        rmse (float): Root mean squared error between the predicted count and 
        observed count.
        mae (float): Mean absolute error between the predicted count and the 
        observed count.
        rmsle (float): Mean squared logarithmic errror between the predicted 
        count and the observed count.
        post_mean_ranking (array): Posterior mean site rank.
        post_mean_ranking_top (array): Posterior mean probability that a site
        belongs to the top m most hazardous sites.
        ranking_top_m_list (list): List of m values used for calculating the 
        posterior mean probability that a site belongs to the top m most 
        hazardous sites.
        post_r (DataFrame): Posterior summary of the negative binomial success 
        rate.
        post_mean_f (array): Posterior mean of the residual value of the link
        function excluding fixed, random and spatial link function components
        (required for BART calibration).
        post_variable_inclusion_probs (DataFrame): Posterior summary of BART 
        variable inclusion proportions. 
        post_beta (DataFrame): Posterior summary of fixed link function 
        parameters.
        post_mu (DataFrame): Posterior summary of mean of random link function
        parameters.
        post_sigma (DataFrame): Posterior summary of standard deviation of 
        random link function parameters.
        post_Sigma (DataFrame): Posterior summary of covariance of random link
        function parameters. 
        post_sigma_mess: Posterior summary of spatial error scale.
        post_tau: Posterior summary of MESS association parameters. 
        post_phi: Posterior summary of spatial errors.
    """   
    
    def __init__(
            self, 
            options, bart_options, toc, 
            lppd, waic, 
            post_ls, post_dss, post_rps, 
            post_mean_lam, 
            rmse, mae, rmsle,
            post_mean_ranking, 
            post_mean_ranking_top, ranking_top_m_list,
            post_r, post_mean_f, 
            post_beta,
            post_mu, post_sigma, post_Sigma,
            post_variable_inclusion_props,
            post_sigma_mess, post_tau, post_phi
            ):
        self.options = options
        self.bart_options = bart_options
        self.estimation_time = toc
        
        self.lppd = lppd
        self.waic = waic
        
        self.post_ls = post_ls
        self.post_dss = post_dss
        self.post_rps = post_rps
        
        self.post_mean_lam = post_mean_lam
        self.rmse = rmse
        self.mae = mae
        self.rmsle = rmsle
        
        self.post_mean_ranking = post_mean_ranking
        self.post_mean_ranking_top = post_mean_ranking_top
        self.ranking_top_m_list = ranking_top_m_list
        
        self.post_r = post_r
        self.post_mean_f = post_mean_f
        
        self.post_variable_inclusion_props = post_variable_inclusion_props
        
        self.post_beta = post_beta
        
        self.post_mu = post_mu
        self.post_sigma = post_sigma
        self.post_Sigma = post_Sigma
        
        self.post_sigma_mess = post_sigma_mess
        self.post_tau = post_tau
        self.post_phi = post_phi
        
class ResultsVb:
    """ Holds VB results.
    
    Attributes:
        options (OptionsVB): Options used for VB algorithm.
        estimation time (float): Estimation time.
        lppd (float): Log pointwise predictive density.
        waic (float): Widely applicable information criterion.
        post_mean_lam (array): Posterior mean of the expected count of each 
        observation.
        rmse (float): Root mean squared error between the predicted count and 
        observed count.
        mae (float): Mean absolute error between the predicted count and the 
        observed count.
        rmsle (float): Mean squared logarithmic errror between the predicted 
        count and the observed count.

        
    """   
    
    def __init__(
            self, 
            options, toc, 
            lppd, waic,
            post_mean_lam, 
            rmse, mae, rmsle,
            r_b, r_c, 
            beta_mu, beta_Si,
            gamma_mu, gamma_Si, mu_mu, mu_Si, Sigma_rho, Sigma_B,
            phi_mu, phi_Si, sigma2_b, sigma2_c, tau_mu, tau_si
       ):
        self.options = options
        self.estimation_time = toc
        
        self.lppd = lppd
        self.waic = waic
        self.post_mean_lam = post_mean_lam
        self.rmse = rmse
        self.mae = mae
        self.rmsle = rmsle
        
        self.r_b = r_b
        self.r_c = r_c
        
        self.beta_mu = beta_mu
        self.beta_Si = beta_Si
        
        self.gamma_mu = gamma_mu
        self.gamma_Si = gamma_Si
        self.mu_mu = mu_mu
        self.mu_Si = mu_Si
        self.Sigma_rho = Sigma_rho
        self.Sigma_B = Sigma_B
        
        self.phi_mu = phi_mu
        self.phi_Si = phi_Si
        self.sigma2_b = sigma2_b
        self.sigma2_c = sigma2_c
        self.tau_mu = tau_mu
        self.tau_si = tau_si
        
class ResultsCaviInfvb:
    """ Holds CAVI INFVB results.
    
    Attributes:        
    """   
    
    def __init__(
            self, 
            elbo,
            r_b, r_c, 
            psi_mu, psi_sigma,
            beta_mu, beta_Si,
            gamma_mu, gamma_Si, mu_mu, mu_Si, Sigma_rho, Sigma_B,
            phi_mu, phi_Si,
            sigma, tau
       ):
        self.elbo = elbo
        
        self.r_b = r_b
        self.r_c = r_c
        
        self.psi_mu = psi_mu
        self.psi_sigma = psi_sigma
        
        self.beta_mu = beta_mu
        self.beta_Si = beta_Si
        
        self.gamma_mu = gamma_mu
        self.gamma_Si = gamma_Si
        self.mu_mu = mu_mu
        self.mu_Si = mu_Si
        self.Sigma_rho = Sigma_rho
        self.Sigma_B = Sigma_B
        
        self.phi_mu = phi_mu
        self.phi_Si = phi_Si
        
        self.sigma = sigma
        self.tau = tau
        
class GridInfvb:
    """ Holds gridded INFVB results.
    
    Attributes:        
    """   
    
    def __init__(
            self, 
            tau, sigma, tau_sigma, 
            results, weights, 
            estimation_time
       ):
        self.tau = tau
        self.sigma = sigma
        self.tau_sigma = tau_sigma 
        self.results = results
        self.weights = weights
        self.estimation_time = estimation_time
        
class ResultsSimulationInfvb:
    """ Holds simulation results of INFVB approximation.
    
    Attributes:
        options (OptionsMcmc): Options used for INFVB algorithm.
        estimation time (float): Estimation time of INVFB algorithm.
        lppd (float): Log pointwise predictive density.
        waic (float): Widely applicable information criterion.
        post_ls (DataFrame): Posterior summary of log-score.
        post_dss (DataFrame): Posterior summary of Dawid-Sebastiani score.
        post_rps (DataFrame): Posterior summary of ranked prob. score.
        post_mean_lam (array): Posterior mean of the expected count of each 
        observation.
        rmse (float): Root mean squared error between the predicted count and 
        observed count.
        mae (float): Mean absolute error between the predicted count and the 
        observed count.
        rmsle (float): Mean squared logarithmic errror between the predicted 
        count and the observed count.
        post_r (DataFrame): Posterior summary of the negative binomial success 
        rate.
        post_beta (DataFrame): Posterior summary of fixed link function 
        parameters.
        post_mu (DataFrame): Posterior summary of mean of random link function
        parameters.
        post_sigma (DataFrame): Posterior summary of standard deviation of 
        random link function parameters.
        post_Sigma (DataFrame): Posterior summary of covariance of random link
        function parameters. 
        post_mean_gamma: Posterior mean of individual-specific link function
        parameters
        post_sigma_mess: Posterior summary of spatial error scale.
        post_tau: Posterior summary of MESS association parameters. 
        post_phi: Posterior summary of spatial errors.
    """   
    
    def __init__(
            self, 
            options, estimation_time,
            lppd, waic,
            post_ls, post_dss, post_rps,
            post_mean_lam, 
            rmse, mae, rmsle,
            post_r,
            post_beta,
            post_mu, post_sigma, post_Sigma, post_mean_gamma,
            post_sigma_mess, post_tau, post_phi
            ):
        self.options = options
        self.estimation_time = estimation_time
        
        self.lppd = lppd
        self.waic = waic
        
        self.post_ls = post_ls
        self.post_dss = post_dss
        self.post_rps = post_rps
        
        self.post_mean_lam = post_mean_lam
        self.rmse = rmse
        self.mae = mae
        self.rmsle = rmsle
        
        self.post_r = post_r
        
        self.post_beta = post_beta
        
        self.post_mu = post_mu
        self.post_sigma = post_sigma
        self.post_Sigma = post_Sigma
        self.post_mean_gamma = post_mean_gamma
        
        self.post_sigma_mess = post_sigma_mess
        self.post_tau = post_tau
        self.post_phi = post_phi
        
class NegativeBinomial:
    """ MCMC method for posterior inference in negative binomial model. """
    
    @staticmethod
    def _F_matrix(y):
        """ Calculates F matrix. """
        y_max = np.max(y)
        F = np.zeros((y_max, y_max))
        for m in np.arange(y_max):
            for j in np.arange(m+1):
                if m==0 and j==0:
                    F[m,j] = 1
                else:
                    F[m,j] = m/(m+1) * F[m-1,j] + (1/(m+1)) * F[m-1,j-1]
        return F
    
    def __init__(self, data, data_bart=None):
        self.data = data
        self.data_bart = data_bart
        self.F = self._F_matrix(data.y)
        self.N = data.N
        self.n_fix = data.n_fix
        self.n_rnd = data.n_rnd
        self.mess = data.W is not None

    ###
    #Convenience
    ###
    
    @staticmethod
    def _log_det(x):
        """ Calculates log determinant of matrix x. """
        return np.linalg.slogdet(x)[1]
    
    @staticmethod
    def _pg_rnd(a, b):
        """ Takes draws from Polya-Gamma distribution with parameters a, b. """
        ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2**16))
        N = a.shape[0]
        r = np.zeros((N,))
        ppg.pgdrawv(a, b, r)
        return r
    
    @staticmethod
    def _mvn_rnd(mu, Sigma):
        """ Takes draws from multivariate normal with mean mu and covariance
        Sigma. 
        
        Keywords:
            mu (array): Mean vector.
            Sigma (array): Covariance matrix.
            
        Returns:
            r (array): Multivariate normal draw.
        """ 
        
        d = mu.shape[0]
        ch = np.linalg.cholesky(Sigma)
        r = mu + (ch @ np.random.randn(d,1)).reshape(d,)
        return r
    
    @staticmethod
    def _mvn_rnd_arr(mu, Sigma):
        """ Takes draws from multivariate normal with mean mu and covariance
        Sigma.
        
        Keywords:
            mu (array): Array of mean vectors with shape N x d.
            Sigma (array): Array of precision matrices with shape N x d x d.
            
        Returns:
            r (array): Array of multivariate normal draws with shape N x d.
        """
        
        N, d = mu.shape
        ch = np.linalg.cholesky(Sigma)
        r = mu + (ch @ np.random.randn(N,d,1)).reshape(N,d)
        return r 
    
    @staticmethod
    def _mvn_prec_rnd(mu, P):
        """ Takes draws from multivariate normal with mean mu and precision P. 
        
        Keywords:
            mu (array): Mean vector.
            P (array): Precision matrix.
            
        Returns:
            r (array): Multivariate normal draw.
        """ 
        
        d = mu.shape[0]
        P_chT = np.linalg.cholesky(P).T
        r = mu + np.linalg.solve(P_chT, np.random.randn(d,1)).reshape(d,)
        return r 
    
    @staticmethod
    def _mvn_prec_rnd_arr(mu, P):
        """ Takes draws from multivariate normal with mean mu and precision P.
        
        Keywords:
            mu (array): Array of mean vectors with shape N x d.
            P (array): Array of precision matrices with shape N x d x d.
            
        Returns:
            r (array): Array of multivariate normal draws with shape N x d.
        """
        
        N, d = mu.shape
        P_chT = np.moveaxis(np.linalg.cholesky(P), [1,2], [2,1])
        r = mu + np.linalg.solve(P_chT, np.random.randn(N,d,1)).reshape(N,d)
        return r 
    
    @staticmethod
    def _nb_lpmf(y, psi, r):
        """ Calculates log pmf of negative binomial. """
        lc = loggamma(y + r) - loggamma(r) - loggamma(y + 1)
        lp = y * psi - (y + r) * np.log1p(np.exp(psi))
        lpmf = lc + lp
        return lpmf
    
    @staticmethod
    def _nb_cdf(y, r, p):
        y_pos_idx = y >= 0
        F = np.zeros((y.shape[0],))
        F[y_pos_idx] = 1 - betainc(y[y_pos_idx] + 1, r, p[y_pos_idx])  
        return F
    
    @staticmethod
    def _dss(y, psi, r, lam):
        mu = lam
        var = (np.exp(psi) + np.exp(2 * psi)) * r
        dss = np.sum((y - mu)**2 / var + np.log(var))
        return dss
    
    def _rps(self, y, psi, r):
        p = np.exp(psi) / (1 + np.exp(psi))
        p2 = np.exp(psi) + np.exp(2 * psi)
        
        F1 = self._nb_cdf(y, r, p)
        F2 = self._nb_cdf(y - 1, r + 1, p)
        H = hyp2f1(r + 1, 0.5, 2, -4 * p2)
        
        rps = np.sum(y * (2 * F1 - 1) - r * p2 * ((1 - p) * (2 * F2 - 1) + H))
        return rps
         
    ###
    #Sampling
    ###
        
    def _next_omega(self, r, psi):
        """ Updates omega. """
        omega = self._pg_rnd(self.data.y + r, psi)
        Omega = np.diag(omega)
        z = (self.data.y - r) / 2 / omega
        return omega, Omega, z

    def _next_beta(self, z, psi_rnd, psi_bart, phi, 
                   Omega, beta_mu0, beta_Si0Inv):
        """ Updates beta. """
        beta_SiInv = self.data.x_fix.T @ Omega @ self.data.x_fix + beta_Si0Inv
        beta_mu = np.linalg.solve(beta_SiInv,
                                  self.data.x_fix.T @ Omega \
                                  @ (z - self.data.offset \
                                     - psi_rnd - psi_bart - phi) \
                                  + beta_Si0Inv @ beta_mu0)
        beta = self._mvn_prec_rnd(beta_mu, beta_SiInv)
        return beta
    
    def _next_gamma(self, z, psi_fix, psi_bart, phi, omega, mu, SigmaInv):
        """ Updates gamma. """
        gamma_SiInv = omega.reshape(self.N,1,1) * \
            self.data.x_rnd.reshape(self.N,self.n_rnd,1) \
            @ self.data.x_rnd.reshape(self.N,1,self.n_rnd) \
            + SigmaInv.reshape(1,self.n_rnd,self.n_rnd)
        gamma_mu_pre = (omega * (z - self.data.offset \
                                 - psi_fix - psi_bart - phi)) \
            .reshape(self.N,1) \
            * self.data.x_rnd + (SigmaInv @ mu).reshape(1,self.n_rnd)
        gamma_mu = np.linalg.solve(gamma_SiInv, gamma_mu_pre)
        gamma = self._mvn_prec_rnd_arr(gamma_mu, gamma_SiInv)            
        return gamma
 
    def _next_mu(self, gamma, SigmaInv, mu_mu0, mu_Si0Inv):
        """ Updates mu. """
        mu_SiInv = self.N * SigmaInv + mu_Si0Inv
        mu_mu = np.linalg.solve(mu_SiInv,
                                SigmaInv @ np.sum(gamma, axis=0)
                                + mu_Si0Inv @ mu_mu0)
        mu = self._mvn_prec_rnd(mu_mu, mu_SiInv)
        return mu

    def _next_Sigma(self, gamma, mu, a, nu):    
        """ Updates Sigma. """
        diff = gamma - mu
        Sigma = (invwishart.rvs(nu + self.N + self.n_rnd - 1, 
                                2 * nu * np.diag(a) + diff.T @ diff))\
            .reshape((self.n_rnd, self.n_rnd))
        SigmaInv = np.linalg.inv(Sigma)
        return Sigma, SigmaInv

    def _next_a(self, SigmaInv, nu, A):
        """ Updates a. """
        a = np.random.gamma((nu + self.n_rnd) / 2, 
                            1 / (1 / A**2 + nu * np.diag(SigmaInv)))
        return a

    def _next_phi(self, z, psi_fix, psi_rnd, psi_bart, Omega, Omega_tilde):
        """ Updates phi. """
        phi_SiInv = Omega + Omega_tilde
        phi_mu = np.linalg.solve(phi_SiInv, 
                                 Omega \
                                 @ (z - self.data.offset \
                                    - psi_fix - psi_rnd - psi_bart))
        phi = self._mvn_prec_rnd(phi_mu, phi_SiInv)
        return phi      
    
    def _next_sigma2(self, phi, S2, sigma2_b0, sigma2_c0):
        """ Updates sigma_mess**2. """
        b = sigma2_b0 + self.data.N / 2
        c = sigma2_c0 + phi.T @ S2 @ phi / 2
        sigma2 = 1 / np.random.gamma(b, 1 / c)
        return sigma2
    
    def _log_target_tau(self, tau, S, phi, sigma2, tau_mu0, tau_si0):
        """ Calculates target density for MH. """
        if S is None:
            S = expm(tau * self.data.W)
        Omega_tilde = S.T @ S / sigma2
        lt = - phi.T @ Omega_tilde @ phi / 2 \
             - (tau - tau_mu0)**2 / 2 / tau_si0**2
        return lt, S
    
    def _next_tau(self, tau, S, phi, sigma2, tau_mu0, tau_si0, mh_step):
        """ Updates tau. """
        lt_tau, S = self._log_target_tau(tau, S, phi, sigma2, tau_mu0, tau_si0)
        tau_star = tau + np.sqrt(mh_step) * np.random.randn()
        lt_tau_star, S_star = self._log_target_tau(
            tau_star, None, phi, sigma2, tau_mu0, tau_si0
            )
        log_r = np.log(np.random.rand())
        log_alpha = lt_tau_star - lt_tau
        if log_r <= log_alpha:
            tau = tau_star
            S = np.array(S_star)
            mh_tau_accept = True
        else:
            mh_tau_accept = False
        return tau, S, mh_tau_accept
    
    @staticmethod
    def _next_h(r, r0, b0, c0):
        """ Updates h. """
        h = np.random.gamma(r0 + b0, 1/(r + c0))
        return h
    
    @staticmethod
    @jit
    def _next_L(y, r, F):
        """ Updates L. """
        N = y.shape[0]
        L = np.zeros((N,))
        for n in np.arange(N):
            if y[n]:
                log_numer = np.zeros((y[n],))
                for j in np.arange(y[n]):
                    log_numer[j] = np.log(F[y[n]-1,j]) + np.log(r) * (j+1)
                log_numer_max = np.max(log_numer)
                log_denom = log_numer_max \
                    + np.log(np.sum(np.exp(log_numer - log_numer_max)))
                L_p = np.exp(log_numer - log_denom)
                L[n] = np.searchsorted(np.cumsum(L_p), np.random.rand()) + 1
        return L
    
    @staticmethod
    def _next_r(r0, L, h, psi):
        """ Updates r. """
        sum_p = np.sum(np.log1p(np.exp(psi)))
        r = np.random.gamma(r0 + np.sum(L), 1 / (h + sum_p))
        return r
    
    @staticmethod
    def _rank(lam, ranking_top_m_list):
        """ Computes the rank of each site.
        
        Keywords:
            lam (array): Expected counts.
            ranking_top_m_list (list): List of m values used for extracting 
            whether a site belongs to the top m most hazardous sites.
            
        Returns:
            ranking (array): Ranks 
            ranking_top (array): Booleans indicating whether a site belongs to 
            the top m most hazardous sites. 
        """
        
        ranking = rankdata(-lam, method='min')
        ranking_top = np.zeros((lam.shape[0], len(ranking_top_m_list)))
        for j, m in enumerate(ranking_top_m_list):
            ranking_top[:,j] = ranking <= m
        return ranking, ranking_top
    
    def _mcmc_chain(
            self,
            chainID, 
            options, bart_options,
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_init, beta_init, mu_init, Sigma_init,
            ranking_top_m_list):
        """ Markov chain for MCMC simulation for negative binomial model.
        
        Keywords:
            chainID (int): ID of Markov chain.
            options (Options): Simulation options.
            bart_options (BartOptions): Options for BART component.
            r0 (float): Hyperparameter of prior on r; r ~ Gamma(r0, h).
            b0 (float): Hyperparameter of prior on h; h ~ Gamma(b0, c0).
            c0 (float): Hyperparameter of prior on h; h ~ Gamma(b0, c0).
            beta_mu0 (array): Hyperparameter of prior on beta; 
            beta ~ N(beta_mu0, beta_Si0)
            beta_Si0Inv (array): Inverse of hyperparameter of prior on beta; 
            beta ~ N(beta_mu0, beta_Si0)
            mu_mu0 (array): Hyperparameter of prior on mu; 
            mu ~ N(mu_mu0, mu_Si0)
            mu_Si0Inv (array): Inverse of hyperparameter of prior on mu; 
            mu ~ N(mu_mu0, mu_Si0).
            nu (float) ~ Hyperparameter of prior on a; a ~ Gamma(nu, A_k)
            A (array): Hyperparameter of prior on a; a ~ Gamma(nu, A_k)
            sigma2_b0 (float): Hyperparameter of prior on sigma2; 
            1/sigma2 ~ Gamma(sigma2_b0, sigma2_c0).
            sigma2_c0 (float): Hyperparameter of prior on sigma2; 
            sigma2 ~ Gamma(sigma2_b0, sigma2_c0).
            tau_mu0 (float): Hyperparameter of prior on tau; 
            tau ~ N(tau_mu0, tau_si0**2).
            tau_si0 (float): Hyperparameter of prior on tau; 
            tau ~ N(tau_mu0, tau_si0**2).
            r_init (float): Initial value of r.
            beta_init (array): Initial value of beta.
            mu_init (array): Initial value of mu.
            Sigma_init (array): Initial value of Sigma.
            ranking_top_m_list (list): List of m values used for extracting 
            whether a site belongs to the top m most hazardous sites.
        """
        
        ###
        #Storage
        ###
        
        file_name = options.model_name + '_draws_chain' + str(chainID + 1) \
        + '.hdf5'
        if os.path.exists(file_name):
            os.remove(file_name) 
        file = h5py.File(file_name, 'a')    
        
        lp_store = file.create_dataset('lp_store', \
        (options.nKeep,self.N), dtype='float64')
        lp_store_tmp = np.zeros((options.nMem,self.N))
        
        lam_store = file.create_dataset('lam_store', \
        (options.nKeep,self.N), dtype='float64')
        lam_store_tmp = np.zeros((options.nMem,self.N))
        
        ranking_store = file.create_dataset('ranking_store', \
        (options.nKeep,self.N), dtype='float64')
        ranking_store_tmp = np.zeros((options.nMem,self.N))
        
        ranking_top_store = file.create_dataset('ranking_top_store', \
        (options.nKeep,self.N,len(ranking_top_m_list)), dtype='float64')
        ranking_top_store_tmp = np.zeros((options.nMem, self.N, 
                                          len(ranking_top_m_list)))
        r_store = file.create_dataset('r_store', \
        (options.nKeep,), dtype='float64')
        r_store_tmp = np.zeros((options.nMem,))
        
        f_store = file.create_dataset('f_store', \
        (options.nKeep,self.N), dtype='float64')
        f_store_tmp = np.zeros((options.nMem,self.N))
        
        if self.n_fix:
            beta_store = file.create_dataset('beta_store', \
            (options.nKeep, self.data.n_fix), dtype='float64')
            beta_store_tmp = np.zeros((options.nMem, self.data.n_fix))
            
        if self.n_rnd:
            mu_store = file.create_dataset('mu_store', \
            (options.nKeep, self.data.n_rnd), dtype='float64')
            mu_store_tmp = np.zeros((options.nMem, self.data.n_rnd))
            
            sigma_store = file.create_dataset('sigma_store', \
            (options.nKeep, self.data.n_rnd), dtype='float64')
            sigma_store_tmp = np.zeros((options.nMem, self.data.n_rnd))
            
            Sigma_store = file.create_dataset('Sigma_store', \
            (options.nKeep, self.data.n_rnd, self.data.n_rnd), dtype='float64')
            Sigma_store_tmp \
                = np.zeros((options.nMem, self.data.n_rnd, self.data.n_rnd))
                
        if self.data_bart is not None:
            avg_tree_acceptance_store = np.zeros((options.nIter,))
            avg_tree_depth_store = np.zeros((options.nIter,))
                
            variable_inclusion_props_store \
                = file.create_dataset('variable_inclusion_props_store', \
                                      (options.nKeep, self.data_bart.J), 
                                      dtype='float64')
            variable_inclusion_props_store_tmp \
                = np.zeros((options.nMem, self.data_bart.J))              
        
        if self.mess:
            sigma_mess_store = file.create_dataset('sigma_mess_store', \
            (options.nKeep,), dtype='float64')
            sigma_mess_store_tmp = np.zeros((options.nMem,))

            tau_store = file.create_dataset('tau_store', \
            (options.nKeep,), dtype='float64')
            tau_store_tmp = np.zeros((options.nMem,))
            
            phi_store = file.create_dataset('phi_store', \
            (options.nKeep, self.N), dtype='float64')
            phi_store_tmp = np.zeros((options.nMem, self.N))
            
            mh_tau_accept_store = np.zeros((options.nIter,))
            
        ls_store = file.create_dataset('ls_store', \
        (options.nKeep,), dtype='float64')
        ls_store_tmp = np.zeros((options.nMem,)) 
        
        dss_store = file.create_dataset('dss_store', \
        (options.nKeep,), dtype='float64')
        dss_store_tmp = np.zeros((options.nMem,)) 
        
        rps_store = file.create_dataset('rps_store', \
        (options.nKeep,), dtype='float64')
        rps_store_tmp = np.zeros((options.nMem,)) 
        
        ###
        #Initialise
        ###
        
        r = max(r_init - 0.5 + 1.0 * np.random.rand(), 0.25)
        
        if self.n_fix:
            beta = beta_init - 0.5 + 1 * np.random.rand(self.n_fix,)
            psi_fix = self.data.x_fix @ beta
        else:
            beta = None
            psi_fix = 0
            
        if self.n_rnd:
            mu = mu_init - 0.5 + 1 * np.random.rand(self.n_rnd,)
            Sigma = Sigma_init.copy()
            SigmaInv = np.linalg.inv(Sigma)
            a = np.random.gamma(1/2, A**2)
            gamma = mu + (np.linalg.cholesky(Sigma) \
                     @ np.random.randn(self.n_rnd,self.N)).T
            psi_rnd = np.sum(self.data.x_rnd * gamma, axis=1)
        else:
            psi_rnd = 0
            
        if self.data_bart is not None:
            forest = bt.Forest(bart_options, self.data_bart)
            psi_bart = self.data_bart.unscale(forest.y_hat)
        else:
            forest = None
            psi_bart = 0
        
        if self.mess:
            sigma2 = np.sqrt(0.4)
            tau = -float(-1 + 0.8 * np.random.rand())
            S = expm(tau * self.data.W)
            S2 = S.T @ S
            Omega_tilde = S2 / sigma2         
            eps = np.sqrt(sigma2) * np.random.randn(self.data.N,)
            phi = np.linalg.solve(S, eps)
            mh_step = options.mh_step_initial
        else:
            sigma2 = None
            tau = None       
            S = None
            Omega_tilde = None
            phi = 0
            mh_step = None
  
        psi = self.data.offset + psi_fix + psi_rnd + psi_bart + phi
        
        ###
        #Sampling
        ###
    
        j = -1
        ll = 0
        sample_state = 'burn in'    
        
        for i in np.arange(options.nIter):
            omega, Omega, z = self._next_omega(r, psi)
            
            if self.n_fix:
                beta = self._next_beta(z, psi_rnd, psi_bart, phi, Omega, 
                                       beta_mu0, beta_Si0Inv)
                psi_fix = self.data.x_fix @ beta
                
            if self.n_rnd:
                gamma = self._next_gamma(z, psi_fix, psi_bart, phi, 
                                         omega, mu, SigmaInv)
                mu = self._next_mu(gamma, SigmaInv, mu_mu0, mu_Si0Inv)
                Sigma, SigmaInv = self._next_Sigma(gamma, mu, a, nu)
                a = self._next_a(SigmaInv, nu, A)
                psi_rnd = np.sum(self.data.x_rnd * gamma, axis=1) 
                
            if self.data_bart is not None:
                sigma_weights = np.sqrt(1/omega)
                f = z - self.data.offset - psi_fix - psi_rnd - phi
                self.data_bart.update(f, sigma_weights)
                avg_tree_acceptance_store[i], avg_tree_depth_store[i] = \
                    forest.update(self.data_bart)
                psi_bart = self.data_bart.unscale(forest.y_hat)
                
            if self.mess:
                phi = self._next_phi(z, psi_fix, psi_rnd, psi_bart,
                                     Omega, Omega_tilde)
            
            psi = self.data.offset + psi_fix + psi_rnd + psi_bart + phi 
            
            if self.mess:
                sigma2 = self._next_sigma2(phi, S2, sigma2_b0, sigma2_c0)
                tau, S, mh_tau_accept_store[i] = self._next_tau(
                    tau, S, phi, sigma2, tau_mu0, tau_si0, mh_step
                    )
                S2 = S @ S.T
                Omega_tilde = S2 / sigma2
            
            h = self._next_h(r, r0, b0, c0)
            L = self._next_L(self.data.y, r, self.F)
            r = self._next_r(r0, L, h, psi)
            
            #Adjust MH step size
            if self.mess and ((i+1) % options.mh_window) == 0:
                sl = slice(max(i+1-options.mh_window,0), i+1)
                mean_accept = mh_tau_accept_store[sl].mean()
                if mean_accept >= options.mh_target:
                    mh_step += options.mh_correct
                else:
                    mh_step -= options.mh_correct
                     
            if ((i+1) % options.disp) == 0:  
                if (i+1) > options.nBurn:
                    sample_state = 'sampling'
                verbose = 'Chain ' + str(chainID + 1) \
                + '; iteration: ' + str(i + 1) + ' (' + sample_state + ')'
                if self.data_bart is not None:
                    sl = slice(max(i+1-100,0),i+1)
                    ravg_depth = np.round(
                        np.mean(avg_tree_depth_store[sl]), 2)
                    ravg_acceptance = np.round(
                        np.mean(avg_tree_acceptance_store[sl]), 2)
                    verbose = verbose \
                    + '; avg. tree depth: ' + str(ravg_depth) \
                    + '; avg. tree acceptance: ' + str(ravg_acceptance)
                if self.mess:
                    verbose = verbose \
                    + '; avg. tau acceptance: ' \
                    + str(mh_tau_accept_store[sl].mean())
                print(verbose)
                sys.stdout.flush()
                
            if (i+1) > options.nBurn:                  
                if ((i+1) % options.nThin) == 0:
                    j+=1
                    
                    lp = self._nb_lpmf(self.data.y, psi, r)
                    lp_store_tmp[j,:] = lp
                    lam = np.exp(psi + np.log(r))
                    lam_store_tmp[j,:] = lam
                    ranking_store_tmp[j,:], ranking_top_store_tmp[j,:,:] \
                        = self._rank(lam, ranking_top_m_list)
                    r_store_tmp[j] = r
                    f_store_tmp[j,:] = z - phi
                    
                    if self.n_fix:
                        beta_store_tmp[j,:] = beta
                        
                    if self.n_rnd:
                        mu_store_tmp[j,:] = mu
                        sigma_store_tmp[j,:] = np.sqrt(np.diag(Sigma))
                        Sigma_store_tmp[j,:,:] = Sigma
                        
                    if self.data_bart is not None:
                        variable_inclusion_props_store_tmp[j,:] \
                            = forest.variable_inclusion()
                        
                    if self.mess:
                        sigma_mess_store_tmp[j] = np.sqrt(sigma2)
                        tau_store_tmp[j] = tau
                        phi_store_tmp[j,:] = phi
                        
                    ls_store_tmp[j] = -lp.sum()
                    dss_store_tmp[j] = self._dss(self.data.y, psi, r, lam)
                    rps_store_tmp[j] = self._rps(self.data.y, psi, r)
                        
                if (j+1) == options.nMem:
                    l = ll
                    ll += options.nMem
                    sl = slice(l, ll)
                    
                    print('Storing chain ' + str(chainID + 1))
                    sys.stdout.flush()
                    
                    lp_store[sl,:] = lp_store_tmp
                    lam_store[sl,:] = lam_store_tmp
                    ranking_store[sl,:] = ranking_store_tmp
                    ranking_top_store[sl,:,:] = ranking_top_store_tmp
                    r_store[sl] = r_store_tmp
                    f_store[sl,:] = f_store_tmp
                    
                    if self.n_fix:
                        beta_store[sl,:] = beta_store_tmp
                        
                    if self.n_rnd:
                        mu_store[sl,:] = mu_store_tmp
                        sigma_store[sl,:] = sigma_store_tmp
                        Sigma_store[sl,:,:] = Sigma_store_tmp
                        
                    if self.data_bart is not None:
                        variable_inclusion_props_store[sl,:] \
                            = variable_inclusion_props_store_tmp
                    
                    if self.mess:
                        sigma_mess_store[sl] = sigma_mess_store_tmp
                        tau_store[sl] = tau_store_tmp
                        phi_store[sl,:] = phi_store_tmp
                        
                    ls_store[sl] = ls_store_tmp
                    dss_store[sl] = dss_store_tmp
                    rps_store[sl] = rps_store_tmp
                    
                    j = -1 
        
        if self.data_bart is not None:
            file.create_dataset('avg_tree_acceptance_store', 
                                data=avg_tree_acceptance_store)
            file.create_dataset('avg_tree_depth_store', 
                                data=avg_tree_depth_store)
            
    ###
    #VB updates
    ###
            
    @staticmethod
    def _quad_psi(h, psi_mu, psi_sigma, psi_quad_x, psi_quad_w):
        """ Performs quadrature over variational distribution of psi. """
        psi_mu = psi_mu.reshape(-1,1)
        psi_sigma = psi_sigma.reshape(-1,1)
        psi_quad_x = psi_quad_x.reshape(1,-1)
        psi_quad_w = psi_quad_w.reshape(1,-1)
        y = np.sqrt(2) * psi_sigma * psi_quad_x + psi_mu
        e = np.sum(psi_quad_w * h(y), axis=1) / np.sqrt(np.pi)
        return e
           
    def _quad_omega(self, psi_mu, psi_sigma, psi_quad_x, psi_quad_w):
        """ Calculates expectation of omega via quadrature. """
        h = lambda x: np.tanh(x / 2) / 2 / x
        e = self._quad_psi(h, psi_mu, psi_sigma, psi_quad_x, psi_quad_w)
        return e
            
    def _calc_omega(
            self, 
            psi_mu, psi_sigma, psi_quad_x, psi_quad_w, r_b, r_c
            ):
        """ Returns expectations of omega, Omega and z_star. """
        omega_e = (self.data.y + r_b / r_c) \
            * self._quad_omega(psi_mu, psi_sigma, psi_quad_x, psi_quad_w)
        Omega_e = np.diag(omega_e)
        z_star_e = (self.data.y - r_b / r_c) / 2
        return omega_e, Omega_e, z_star_e
    
    def _update_beta(
            self, 
            z_star_e, Omega_e, 
            beta_mu0, beta_Si0Inv,
            psi_mu_rnd, phi_mu
            ):
        """ Updates variational distribution over beta. """
        beta_SiInv = self.data.x_fix.T @ Omega_e @ self.data.x_fix \
            + beta_Si0Inv
        beta_Si = np.linalg.inv(beta_SiInv)
        beta_mu = beta_Si @ (self.data.x_fix.T @ (z_star_e \
            - Omega_e @ (self.data.offset + psi_mu_rnd + phi_mu)) \
            + beta_Si0Inv @ beta_mu0)
        return beta_mu, beta_Si

    def _update_gamma(
            self, 
            z_star_e, omega_e, 
            mu_mu, SigmaInv_e,
            psi_mu_fix, phi_mu
            ):  
        """ Updates variational distribution over gamma. """
        gamma_SiInv = omega_e.reshape(self.N,1,1) * \
            self.data.x_rnd.reshape(self.N,self.n_rnd,1) \
            @ self.data.x_rnd.reshape(self.N,1,self.n_rnd) \
            + SigmaInv_e.reshape(1,self.n_rnd,self.n_rnd)
        gamma_Si = np.linalg.inv(gamma_SiInv)
        gamma_mu_pre = (z_star_e \
                        - omega_e * (self.data.offset + psi_mu_fix + phi_mu))\
                        .reshape(self.N,1) \
            * self.data.x_rnd + (SigmaInv_e @ mu_mu).reshape(1,self.n_rnd)
        gamma_mu = np.linalg.solve(gamma_SiInv, gamma_mu_pre\
                                   .reshape(self.N,self.n_rnd,1))\
            .reshape(self.N,self.n_rnd)
        return gamma_mu, gamma_Si

    def _update_mu(
            self,
            SigmaInv_e, gamma_mu, mu_mu0, mu_Si0Inv
            ):
        """ Updates variational distribution over mu. """
        mu_Si = np.linalg.inv(self.N * SigmaInv_e + mu_Si0Inv)
        mu_mu = mu_Si @ (SigmaInv_e @ np.sum(gamma_mu, axis=0) \
                         + mu_Si0Inv @ mu_mu0)
        return mu_mu, mu_Si

    def _update_Sigma(
            self,
            Sigma_rho,
            gamma_mu, gamma_Si, mu_mu, mu_Si, nu, a_b, a_c
            ):
        """ Updates variational distribution over Sigma. """
        diff = gamma_mu - mu_mu
        Sigma_B = 2 * nu * np.diag(a_b / a_c) \
            + self.N * mu_Si + np.sum(gamma_Si, axis=0) + diff.T @ diff
        SigmaInv_e = Sigma_rho * np.linalg.inv(Sigma_B)
        return Sigma_B, SigmaInv_e
    
    @staticmethod
    def _update_a(SigmaInv_e, nu, A):
        """ Updates variational distribution over a. """
        a_c = 1/A**2 + nu * np.diag(SigmaInv_e)
        return a_c

    def _update_phi(
            self, 
            z_star_e, psi_mu_fix, psi_mu_rnd, omega_e, Omega_e, Omega_tilde_e
            ):
        """ Updates variational distribution over phi. """
        phi_Si = np.linalg.inv(Omega_e + Omega_tilde_e)
        phi_mu = phi_Si @ (z_star_e \
                           - omega_e * (self.data.offset + \
                                        psi_mu_fix + psi_mu_rnd))
        return phi_mu, phi_Si  
    
    def _update_psi(
            self, 
            psi_mu_fix, beta_mu, beta_Si,
            psi_mu_rnd, gamma_mu, gamma_Si,
            phi_mu, phi_Si
            ):
        """ Updates variational distribution over psi. """
        #mu        
        psi_mu = self.data.offset + psi_mu_fix + psi_mu_rnd + phi_mu
        
        #sigma
        psi_var_fix = 0
        psi_var_rnd = 0
        psi_var_phi = 0
        
        if self.n_fix:
            psi_var_fix = np.diag(self.data.x_fix @ beta_Si 
                                  @ self.data.x_fix.T)
        if self.n_rnd:
            psi_var_rnd = (self.data.x_rnd.reshape(self.N,1,self.n_rnd) \
                           @ gamma_Si \
                           @ self.data.x_rnd.reshape(self.N,self.n_rnd,1))\
                           .reshape(self.N,)
        
        if self.mess:
            psi_var_phi = np.diag(phi_Si)      
        
        psi_var = psi_var_fix + psi_var_rnd + psi_var_phi
        psi_sigma = np.sqrt(psi_var)
        return psi_mu, psi_sigma
    
    def _update_sigma2(self, phi_mu, phi_Si, S2_e, sigma2_c0):
        """ Updates variational distribution over sigma2. """
        sigma2_c = sigma2_c0 \
            + phi_mu.T @ S2_e @ phi_mu / 2 \
            + np.trace(phi_Si @ S2_e) / 2
        return sigma2_c
    
    def _quad_S2_i(self, tau_mu, tau_si, tau_quad_x):
        """ Calculates expectation of S via quadrature. """
        y = np.sqrt(2) * tau_si * tau_quad_x + tau_mu
        S_i = np.stack([expm(y[i] * self.data.W) \
                        for i in np.arange(y.shape[0])])
        ST_i = np.moveaxis(S_i, [0, 1, 2], [0, 2, 1])
        return S_i, ST_i
    
    def _quad_S2_e_Omega_tilde_e(
            self, 
            tau_mu, tau_si, tau_quad_x, tau_quad_w, sigma2_b, sigma2_c
            ):
        """ Returns expectationsof S.T @ S and Omega_tilde. """
        S_i, ST_i = self._quad_S2_i(tau_mu, tau_si, tau_quad_x)
        S2_i = ST_i @ S_i
        S2_e = np.sum(tau_quad_w.reshape(-1,1,1) * S2_i, axis=0) \
            / np.sqrt(np.pi)
        Omega_tilde_e = sigma2_b / sigma2_c * S2_e
        return S2_e, Omega_tilde_e
    
    def _grad_tau(
            self, 
            tau_mu0, tau_si0,
            tau_mu, tau_si, tau_quad_x, tau_quad_w,
            phi_mu, phi_Si, sigma2_b, sigma2_c
            ):
        """ Calculates gradients of parameters of variational distribution over
        tau. """
        W = np.array(self.data.W).reshape(1,self.N,self.N)
        WT = np.array(self.data.W.T).reshape(1,self.N,self.N)
        S_i, ST_i = self._quad_S2_i(tau_mu, tau_si, tau_quad_x)
        
        p0_i = WT @ ST_i @ S_i
        p1_i = ST_i @ W @ S_i
        p_i = p0_i + p1_i
        
        tau_quad_x = tau_quad_x.reshape(-1,1,1)
        tau_quad_w = tau_quad_w.reshape(-1,1,1)
        
        S2_tau_mu_gr = np.sum(tau_quad_w * p_i, axis=0) / np.sqrt(np.pi)
        S2_tau_si2_gr = np.sum(tau_quad_x * tau_quad_w * p_i, axis=0) \
            / np.sqrt(2 * np.pi) / tau_si
        
        t0_tau_mu_gr = -(tau_mu - tau_mu0) / tau_si0**2
        t0_tau_si2_gr = -(1 / 2 / tau_si0**2)
        
        aa = -sigma2_b / sigma2_c / 2
        bb = np.outer(phi_mu, phi_mu) + phi_Si
        t1_tau_mu_gr = aa * np.trace(bb @ S2_tau_mu_gr)
        t1_tau_si2_gr = aa * np.trace(bb @ S2_tau_si2_gr)
        
        tau_mu_gr = t0_tau_mu_gr + t1_tau_mu_gr
        tau_si2_gr = t0_tau_si2_gr + t1_tau_si2_gr
        return tau_mu_gr, tau_si2_gr
    
    def _update_tau(
            self, 
            tau_mu, tau_si, tau_mu0, tau_si0, tau_quad_x, tau_quad_w,
            phi_mu, phi_Si, sigma2_b, sigma2_c
            ):
        """ Updates variational distribution over tau via NCVMP. """
        tau_mu_gr, tau_si2_gr = self._grad_tau(
            tau_mu0, tau_si0,
            tau_mu, tau_si, tau_quad_x, tau_quad_w,
            phi_mu, phi_Si, sigma2_b, sigma2_c
            )
        tau_si2 = -2 * tau_si2_gr
        tau_si = np.sqrt(1 / tau_si2)
        tau_mu += tau_mu_gr / tau_si2
        return tau_mu, tau_si     
    
    @staticmethod
    def _update_h(r_b, r_c, r0, b0, c0):
        """ Updates variational distribution over h. """
        h_b = r0 + b0
        h_c = r_b / r_c + c0
        return h_b, h_c
    
    @staticmethod
    @jit
    def _calc_L(y, log_r_e, F):
        """ Calculates expectation of L. """
        N = y.shape[0]
        L_e = np.zeros((N,))
        for n in np.arange(N):
            if y[n]:
                log_numer = np.zeros((y[n],))
                for j in np.arange(y[n]):
                    log_numer[j] = np.log(F[y[n]-1,j]) + log_r_e * (j+1)
                log_numer_max = np.max(log_numer)
                log_denom = log_numer_max \
                    + np.log(np.sum(np.exp(log_numer - log_numer_max)))
                log_L_p = log_numer - log_denom
                for j in np.arange(y[n]):
                    L_e[n] += np.exp(log_L_p[j] + np.log(j+1))
        return L_e
    
    def _calc_elp(self, psi_mu, psi_sigma, psi_quad_x, psi_quad_w):
        """ Calculates expectation of log(1 + psi) via quadrature. """
        h = lambda x: np.log1p(np.exp(x))
        e = self._quad_psi(h, psi_mu, psi_sigma, psi_quad_x, psi_quad_w)
        return e

    def _update_r(
            self, 
            psi_mu, psi_sigma, psi_quad_x, psi_quad_w, 
            r0, L_e, h_b, h_c):
        """ Updates variational distribution over r. """
        sum_p = np.sum(
            self._calc_elp(psi_mu, psi_sigma, psi_quad_x, psi_quad_w)
            )
        r_b = r0 + np.sum(L_e)
        r_c = h_b / h_c + sum_p
        return r_b, r_c    
    
    def _elbo_lnp_nb(
            self, 
            psi_mu, psi_sigma, psi_quad_x, psi_quad_w, r_b, r_c,
            n_draws=1000
            ):
        """ Computes part of ELBO that pertains to NB likelihood. """
        f1 = lambda x: loggamma(self.data.y.reshape(self.N,1) + x)
        f2 = lambda x: loggamma(x)
        x1 = np.random.gamma(r_b, 1/r_c, (self.N,n_draws))
        x2 = np.random.gamma(r_b, 1/r_c, n_draws)
        e1 = np.mean(f1(x1), axis=1) 
        e2 = np.mean(f2(x2)) 

        e3 = self._calc_elp(psi_mu, psi_sigma, psi_quad_x, psi_quad_w)            

        elbo_lnp_nb = np.sum(
            e1 - e2 + self.data.y * psi_mu - (self.data.y + r_b / r_c) * e3
            )
        return elbo_lnp_nb  
                    
    ###
    #Posterior summary
    ###  
    
    @staticmethod
    def _posterior_summary(options, param_name, nParam, nParam2, verbose):
        """ Returns summary of posterior draws of parameters of interest. """
        headers = ['mean', 'std. dev.', '2.5%', '97.5%', 'Rhat']
        q = (0.025, 0.975)
        nSplit = 2
        
        draws = np.zeros((options.nChain, options.nKeep, nParam, nParam2))
        for c in range(options.nChain):
            file = h5py.File(options.model_name + '_draws_chain' + str(c + 1) \
                             + '.hdf5', 'r')
            draws[c,:,:,:] = np.array(file[param_name + '_store'])\
            .reshape((options.nKeep, nParam, nParam2))
            
        mat = np.zeros((nParam * nParam2, len(headers)))
        post_mean = np.mean(draws, axis=(0,1))
        mat[:, 0] = np.array(post_mean).reshape((nParam * nParam2,))
        mat[:, 1] = np.array(np.std(draws, axis=(0,1)))\
        .reshape((nParam * nParam2,))
        mat[:, 2] = np.array(np.quantile(draws, q[0], axis=(0,1)))\
        .reshape((nParam * nParam2,))
        mat[:, 3] = np.array(np.quantile(draws, q[1], axis=(0,1)))\
        .reshape((nParam * nParam2,))
        
        m = int(options.nChain * nSplit)
        n = int(options.nKeep / nSplit)
        draws_split = np.zeros((m, n, nParam, nParam2))
        draws_split[:options.nChain,:,:,:] = draws[:,:n,:,:]
        draws_split[options.nChain:,:,:,:] = draws[:,n:,:,:]
        mu_chain = np.mean(draws_split, axis=1, keepdims=True)
        mu = np.mean(mu_chain, axis=0, keepdims=True)
        B = (n / (m - 1)) * np.sum((mu_chain - mu)**2, axis=(0,1))
        ssq = (1 / (n - 1)) * np.sum((draws_split - mu_chain)**2, axis=1)
        W = np.mean(ssq, axis=0)
        varPlus = ((n - 1) / n) * W + B / n
        Rhat = np.empty((nParam, nParam2)) * np.nan
        W_idx = W > 0
        Rhat[W_idx] = np.sqrt(varPlus[W_idx] / W[W_idx])
        mat[:,4] = np.array(Rhat).reshape((nParam * nParam2,))
            
        df = pd.DataFrame(mat, columns=headers) 
        if verbose:
            print(' ')
            print(param_name + ':')
            print(df)
        return df  
    
    @staticmethod
    def _posterior_mean(options, param_name, nParam, nParam2): 
        """ Calculates mean of posterior draws of parameter of interest. """
        draws = np.zeros((options.nChain, options.nKeep, nParam, nParam2))
        for c in range(options.nChain):
            file = h5py.File(options.model_name + '_draws_chain' + str(c + 1) \
                             + '.hdf5', 'r')
            draws[c,:,:,:] = np.array(file[param_name + '_store'])\
            .reshape((options.nKeep, nParam, nParam2))
        post_mean = draws.mean(axis=(0,1))
        return post_mean
    
    def _posterior_fit(self, options, verbose):
        """ Calculates LPPD and WAIC. """
        lp_draws = np.zeros((options.nChain, options.nKeep, self.N))
        for c in range(options.nChain):
            file = h5py.File(options.model_name + '_draws_chain' + str(c + 1) \
                             + '.hdf5', 'r')
            lp_draws[c,:,:] = np.array(file['lp' + '_store'])
        
        p_draws = np.exp(lp_draws)
        lppd = np.log(p_draws.mean(axis=(0,1))).sum()
        p_waic = lp_draws.var(axis=(0,1)).sum()
        waic = -2 * (lppd - p_waic)
        
        if verbose:
            print(' ')
            print('LPPD: ' + str(lppd))
            print('WAIC: ' + str(waic))
        return lppd, waic
    
    ###
    #Estimate
    ###            
    
    def estimate_mcmc(
            self,
            options, bart_options,
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_init, beta_init, mu_init, Sigma_init,
            ranking_top_m_list):
        """ Performs MCMC simulation for negative binomial model. 
        
        Keywords:
            options (OptionsMcmc): Simulation options.
            bart_options (BartOptions): Options for BART component.
            r0 (float): Hyperparameter of prior on r; r ~ Gamma(r0, h).
            b0 (float): Hyperparameter of prior on h; h ~ Gamma(b0, c0).
            c0 (float): Hyperparameter of prior on h; h ~ Gamma(b0, c0).
            beta_mu0 (array): Hyperparameter of prior on beta; 
            beta ~ N(beta_mu0, beta_Si0)
            beta_Si0Inv (array): Inverse of hyperparameter of prior on beta; 
            beta ~ N(beta_mu0, beta_Si0)
            mu_mu0 (array): Hyperparameter of prior on mu; 
            mu ~ N(mu_mu0, mu_Si0)
            mu_Si0Inv (array): Inverse of hyperparameter of prior on mu; 
            mu ~ N(mu_mu0, mu_Si0).
            nu (float) ~ Hyperparameter of prior on a; a ~ Gamma(nu, A_k)
            A (array): Hyperparameter of prior on a; a ~ Gamma(nu, A_k)
            sigma2_b0 (float): Hyperparameter of prior on sigma2; 
            sigma2 ~ Gamma(sigma2_b0, sigma2_c0).
            sigma2_c0 (float): Hyperparameter of prior on sigma2; 
            sigma2 ~ Gamma(sigma2_b0, sigma2_c0).
            tau_mu0 (float): Hyperparameter of prior on tau; 
            tau ~ N(tau_mu0, tau_si0**2).
            tau_si0 (float): Hyperparameter of prior on tau; 
            tau ~ N(tau_mu0, tau_si0**2).
            r_init (float): Initial value of r.
            beta_init (array): Initial value of beta.
            mu_init (array): Initial value of mu.
            Sigma_init (array): Initial value of Sigma.
            ranking_top_m_list (list): List of m values used for extracting 
            whether a site belongs to the top m most hazardous sites.
            
        Returns:
            results: MCMC simulation results
        """
        
        np.random.seed(options.seed)
        
        ###
        #Posterior sampling
        ###
        
        tic = time.time()
        
        """
        for c in range(options.nChain):
            self._mcmc_chain(
                c, options, bart_options,
                r0, b0, c0,
                beta_mu0, beta_Si0Inv,
                mu_mu0, mu_Si0Inv, nu, A,
                sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
                r_init, beta_init, mu_init, Sigma_init,
                ranking_top_m_list) 
        """    
        
        Parallel(n_jobs = options.nChain)(delayed(self._mcmc_chain)(
                c, options, bart_options,
                r0, b0, c0,
                beta_mu0, beta_Si0Inv,
                mu_mu0, mu_Si0Inv, nu, A,
                sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
                r_init, beta_init, mu_init, Sigma_init,
                ranking_top_m_list) 
        for c in range(options.nChain))
        
        toc = time.time() - tic
        
        print(' ')
        print('Estimation time [s]: ' + str(toc))
            
        ###
        #Posterior summary
        ###
        
        lppd, waic = self._posterior_fit(options, verbose=True)
        
        post_mean_lam = self._posterior_mean(options, 'lam', self.N, 1) \
            .reshape(self.N,)
            
        rmse = np.sqrt(np.mean((post_mean_lam - self.data.y)**2))
        mae = np.mean(np.abs(post_mean_lam - self.data.y))
        rmsle = np.sqrt(np.mean((np.log1p(post_mean_lam) \
                                - np.log1p(self.data.y))**2))
        print(' ')
        print('RMSE: ' + str(rmse))
        print('MAE: ' + str(mae))
        print('RMSLE: ' + str(rmsle))
        
        post_mean_ranking = self._posterior_mean(options, 'ranking', 
                                                 self.N, 1).reshape(self.N,)
        
        post_mean_ranking_top = self._posterior_mean(options, 'ranking_top', 
                                                     self.N, 
                                                     len(ranking_top_m_list))
        
        post_r = self._posterior_summary(options, 'r', 1, 1, verbose=True)
        post_mean_f = self._posterior_mean(options, 'f', self.N, 1)\
            .reshape(self.N,)
        
        if self.n_fix:
            post_beta = self._posterior_summary(options, 'beta', 
                                                self.data.n_fix, 1,
                                                verbose=True)
        else:
            post_beta = None
            
        if self.n_rnd:
            post_mu = self._posterior_summary(options, 'mu', 
                                              self.data.n_rnd, 1,
                                              verbose=True) 
            post_sigma = self._posterior_summary(options, 'sigma', 
                                                 self.data.n_rnd, 1,
                                                 verbose=True) 
            post_Sigma = self._posterior_summary(options, 'Sigma', 
                                                 self.data.n_rnd,
                                                 self.data.n_rnd,
                                                 verbose=True) 
        else:
            post_mu = None
            post_sigma = None
            post_Sigma = None
            
        if self.data_bart is not None:
            post_variable_inclusion_props \
                = self._posterior_summary(options, 'variable_inclusion_props', 
                                          self.data_bart.J, 1,
                                          verbose=False) 
        else:
            post_variable_inclusion_props = None
        
        if self.mess:
            post_sigma_mess = self._posterior_summary(options, 'sigma_mess', 
                                                      1, 1,
                                                      verbose=True)
            post_tau = self._posterior_summary(options, 'tau', 1, 1,
                                               verbose=True)
            post_phi = self._posterior_summary(options, 'phi', self.N, 1,
                                               verbose=False)
        else:
            post_sigma_mess = None
            post_tau = None
            post_phi = None
            
        post_ls = self._posterior_summary(options, 'ls', 1, 1, verbose=True)
        post_dss = self._posterior_summary(options, 'dss', 1, 1, verbose=True)
        post_rps = self._posterior_summary(options, 'rps', 1, 1, verbose=True)
        
        ###
        #Delete draws
        ###
        
        if options.delete_draws:
            for c in range(options.nChain):
                os.remove(options.model_name + '_draws_chain' + str(c+1) \
                          + '.hdf5')          
        
        ###
        #Results
        ###
        
        results = ResultsMcmc(
            options, bart_options, toc,
            lppd, waic,
            post_ls, post_dss, post_rps,
            post_mean_lam, 
            rmse, mae, rmsle,
            post_mean_ranking, 
            post_mean_ranking_top, ranking_top_m_list,
            post_r, post_mean_f,
            post_beta,
            post_mu, post_sigma, post_Sigma,
            post_variable_inclusion_props,
            post_sigma_mess, post_tau, post_phi
            )
        
        return results  
    
    def estimate_vb(
            self,
            options, 
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_b_init, r_c_init, 
            beta_mu_init, beta_Si_init, 
            mu_mu_init, mu_Si_init,
            Sigma_B_init,
            tau_mu_init, tau_si_init,
            sigma2_c_init, phi_si_init
            ):
        """ Performs VB estimation of negative binomial model. """
        
        ###
        #CAVI
        ###
        
        psi_quad_x, psi_quad_w = np.polynomial.hermite.hermgauss(
            options.psi_quad_nodes
            )
        
        r_b = r_b_init
        r_c = r_c_init
        beta_mu = np.array(beta_mu_init)
        beta_Si = np.array(beta_Si_init)
        
        mu_mu = np.array(mu_mu_init)
        mu_Si = np.array(mu_Si_init)
        a_b = (nu + self.n_rnd) / 2
        a_c = a_b * np.ones((self.n_rnd,))
        Sigma_rho = nu + self.N + self.n_rnd - 1
        Sigma_B = np.array(Sigma_B_init)
        Sigma_BInv = np.linalg.inv(Sigma_B)
        SigmaInv_e = Sigma_rho * Sigma_BInv
        gamma_mu = np.zeros((self.N,self.n_rnd))
        gamma_Si = np.stack([0.01 * np.eye(self.n_rnd) for i in \
                             np.arange(self.N)])
         
        tau_mu = tau_mu_init
        tau_si = tau_si_init
        tau_quad_x = None
        tau_quad_w = None
        sigma2_b = sigma2_b0 + self.data.N / 2
        sigma2_c = sigma2_c_init
        phi_mu = 0
        phi_Si = 0
        if self.mess:
            tau_quad_x, tau_quad_w = np.polynomial.hermite.hermgauss(
                options.tau_quad_nodes
                )
            S2_e, Omega_tilde_e = self._quad_S2_e_Omega_tilde_e(
                tau_mu, tau_si, tau_quad_x, tau_quad_w, 
                sigma2_b, sigma2_c
                )
            phi_mu = np.zeros(self.N,)
            phi_Si = phi_si_init**2 * np.eye(self.N,)
        
        psi_mu_fix = np.zeros((self.N,))
        psi_mu_rnd = np.zeros((self.N,))
        if self.n_fix:
            psi_mu_fix = self.data.x_fix @ beta_mu
        if self.n_rnd:
            psi_mu_rnd = np.sum(self.data.x_rnd * gamma_mu, axis=1)    
        psi_mu = psi_mu_fix + psi_mu_rnd + phi_mu
        psi_sigma = 0.01 * np.ones((self.data.N,))
        
        par_old = np.concatenate((
                np.array(r_b).reshape(1,), np.array(r_c).reshape(1,),
                beta_mu, np.diag(beta_Si_init),
                mu_mu, np.diag(Sigma_B), a_c
                ))
        par_mat = np.zeros((5, par_old.shape[0]))
        
        tic = time.time()
        
        for i in np.arange(options.max_iter):
            
            ###
            #Updates
            ###
            
            omega_e, Omega_e, z_star_e = self._calc_omega(
                psi_mu, psi_sigma, psi_quad_x, psi_quad_w, r_b, r_c
                )
            
            if self.n_fix:
                beta_mu, beta_Si = self._update_beta(
                    z_star_e, Omega_e, 
                    beta_mu0, beta_Si0Inv, 
                    psi_mu_rnd, phi_mu
                    )   
                psi_mu_fix = self.data.x_fix @ beta_mu
                
            if self.n_rnd:
                gamma_mu, gamma_Si = self._update_gamma(
                    z_star_e, omega_e, 
                    mu_mu, SigmaInv_e,
                    psi_mu_fix, phi_mu
                    )
                mu_mu, mu_Si = self._update_mu(
                    SigmaInv_e, gamma_mu, mu_mu0, mu_Si0Inv
                    )
                Sigma_B, SigmaInv_e = self._update_Sigma(
                    Sigma_rho,
                    gamma_mu, gamma_Si, mu_mu, mu_Si, nu, a_b, a_c
                    )
                a_c = self._update_a(SigmaInv_e, nu, A)
                psi_mu_rnd = np.sum(self.data.x_rnd * gamma_mu, axis=1) 
                
            if self.mess:
                phi_mu, phi_Si = self._update_phi(
                    z_star_e, psi_mu_fix, psi_mu_rnd, 
                    omega_e, Omega_e, Omega_tilde_e
                    )
               
            psi_mu, psi_sigma = self._update_psi(
                psi_mu_fix, beta_mu, beta_Si,
                psi_mu_rnd, gamma_mu, gamma_Si,
                phi_mu, phi_Si
                )
            
            if self.mess:
                sigma2_c = self._update_sigma2(phi_mu, phi_Si, S2_e, sigma2_c0)
                tau_mu, tau_si = self._update_tau(
                    tau_mu, tau_si, tau_mu0, tau_si0, tau_quad_x, tau_quad_w,
                    phi_mu, phi_Si, sigma2_b, sigma2_c
                    )
                S2_e, Omega_tilde_e = self._quad_S2_e_Omega_tilde_e(
                    tau_mu, tau_si, tau_quad_x, tau_quad_w, sigma2_b, sigma2_c
                    )
                
            h_b, h_c = self._update_h(r_b, r_c, r0, b0, c0)
            log_r_e = digamma(r_b) - np.log(r_c)
            L_e = self._calc_L(self.data.y, log_r_e, self.F)
            r_b, r_c = self._update_r(
                psi_mu, psi_sigma, psi_quad_x, psi_quad_w, 
                r0, L_e, h_b, h_c
                )
            
            ###
            #Check for convergence
            ###
            
            par = np.concatenate((
                np.array(r_b).reshape(1,), np.array(r_c).reshape(1,),
                beta_mu, np.diag(beta_Si),
                mu_mu, np.diag(Sigma_B), a_c
                ))
            par_mat = np.vstack((par_mat[1:,:], par.reshape(1,-1)))
            par_new = par_mat.mean(axis=0)
            par_change = np.max(np.absolute((par_new - par_old)) / 
                                np.absolute(par_old + 1e-8))
            converged = par_change < options.tol
            par_old = np.array(par_new)
            
            ###
            #Display progress
            ###
            
            print('Iteration: ' + str(i+1) + '; ' \
                  + 'max. rel. change param.: ' + str(par_change))
            
            print(beta_mu)
            print(mu_mu)
            print(tau_mu)
            print(tau_si)
            print(sigma2_c / (sigma2_b - 1))
            print(r_b / r_c)
            
            ###
            #If converged, break
            ###
            
            if i > 4 and converged:
                break
             
        toc = time.time() - tic
        
        print(' ')
        print('Estimation time [s]: ' + str(toc))
            
        ###
        #Simulate conditional expectation of count for each unit
        ###
        
        lppd=0
        waic=0
        post_mean_lam=0 
        rmse=0
        mae=0
        rmsle=0
        
        ###
        #Results
        ###
        
        if self.n_fix == 0:
            beta_mu = None
            beta_Si = None
            
        if self.n_rnd == 0:
            gamma_mu = None
            gamma_Si = None
            mu_mu = None
            mu_Si = None
            Sigma_rho = None 
            Sigma_B = None  
            
        if not self.mess:
            phi_mu = None
            phi_Si = None
            sigma2_b = None
            sigma2_c = None
            tau_mu = None
            tau_si = None
            
        results = ResultsVb(
            options, toc,
            lppd, waic,
            post_mean_lam, 
            rmse, mae, rmsle,
            r_b, r_c,
            beta_mu, beta_Si,
            gamma_mu, gamma_Si, mu_mu, mu_Si, Sigma_rho, Sigma_B,
            phi_mu, phi_Si, sigma2_b, sigma2_c, tau_mu, tau_si
            )        
        
        return results      
        
    def _cavi_infvb(
            self,
            options, 
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_b_init, r_c_init, 
            beta_mu_init, beta_Si_init, 
            mu_mu_init, mu_Si_init,
            Sigma_B_init,
            phi_si_init,
            tau, sigma,
            psi_quad_x, psi_quad_w
            ):
        """ Implements CAVI for one grid point. """
        
        sigma2 = sigma**2
        
        ###
        #CAVI
        ###
        
        r_b = r_b_init
        r_c = r_c_init
        beta_mu = np.array(beta_mu_init)
        beta_Si = np.array(beta_Si_init)
        
        mu_mu = np.array(mu_mu_init)
        mu_Si = np.array(mu_Si_init)
        a_b = (nu + self.n_rnd) / 2
        a_c = a_b * np.ones((self.n_rnd,))
        Sigma_rho = nu + self.N + self.n_rnd - 1
        Sigma_B = np.array(Sigma_B_init)
        Sigma_BInv = np.linalg.inv(Sigma_B)
        SigmaInv_e = Sigma_rho * Sigma_BInv
        gamma_mu = np.zeros((self.N,self.n_rnd))
        gamma_Si = np.stack([0.01 * np.eye(self.n_rnd) for i in \
                             np.arange(self.N)])
         
        S = expm(tau * self.data.W)
        S2 = S.T @ S
        Omega_tilde = S2 / sigma2 
        phi_mu = np.zeros(self.N,)
        phi_Si = phi_si_init**2 * np.eye(self.N,)
        
        psi_mu_fix = np.zeros((self.N,))
        psi_mu_rnd = np.zeros((self.N,))
        if self.n_fix:
            psi_mu_fix = self.data.x_fix @ beta_mu
        if self.n_rnd:
            psi_mu_rnd = np.sum(self.data.x_rnd * gamma_mu, axis=1)    
        psi_mu = psi_mu_fix + psi_mu_rnd + phi_mu
        psi_sigma = 0.01 * np.ones((self.data.N,))
        
        par_old = np.concatenate((
                np.array(r_b).reshape(1,), np.array(r_c).reshape(1,),
                beta_mu, np.diag(beta_Si_init),
                mu_mu, np.diag(Sigma_B), a_c
                ))
        par_mat = np.zeros((5, par_old.shape[0]))
        
        for i in np.arange(options.max_iter):
            
            ###
            #Updates
            ###
            
            omega_e, Omega_e, z_star_e = self._calc_omega(
                psi_mu, psi_sigma, psi_quad_x, psi_quad_w, r_b, r_c
                )
            
            if self.n_fix:
                beta_mu, beta_Si = self._update_beta(
                    z_star_e, Omega_e, 
                    beta_mu0, beta_Si0Inv, 
                    psi_mu_rnd, phi_mu
                    )   
                psi_mu_fix = self.data.x_fix @ beta_mu
                
            if self.n_rnd:
                gamma_mu, gamma_Si = self._update_gamma(
                    z_star_e, omega_e, 
                    mu_mu, SigmaInv_e,
                    psi_mu_fix, phi_mu
                    )
                mu_mu, mu_Si = self._update_mu(
                    SigmaInv_e, gamma_mu, mu_mu0, mu_Si0Inv
                    )
                Sigma_B, SigmaInv_e = self._update_Sigma(
                    Sigma_rho,
                    gamma_mu, gamma_Si, mu_mu, mu_Si, nu, a_b, a_c
                    )
                a_c = self._update_a(SigmaInv_e, nu, A)
                psi_mu_rnd = np.sum(self.data.x_rnd * gamma_mu, axis=1) 
                
            phi_mu, phi_Si = self._update_phi(
                z_star_e, psi_mu_fix, psi_mu_rnd, 
                omega_e, Omega_e, Omega_tilde
                )
               
            psi_mu, psi_sigma = self._update_psi(
                psi_mu_fix, beta_mu, beta_Si,
                psi_mu_rnd, gamma_mu, gamma_Si,
                phi_mu, phi_Si
                )
                
            h_b, h_c = self._update_h(r_b, r_c, r0, b0, c0)
            log_r_e = digamma(r_b) - np.log(r_c)
            L_e = self._calc_L(self.data.y, log_r_e, self.F)
            r_b, r_c = self._update_r(
                psi_mu, psi_sigma, psi_quad_x, psi_quad_w, 
                r0, L_e, h_b, h_c
                )
            
            ###
            #Check for convergence
            ###
            
            par = np.concatenate((
                np.array(r_b).reshape(1,), np.array(r_c).reshape(1,),
                beta_mu, np.diag(beta_Si),
                mu_mu, np.diag(Sigma_B), a_c
                ))
            par_mat = np.vstack((par_mat[1:,:], par.reshape(1,-1)))
            par_new = par_mat.mean(axis=0)
            par_change = np.max(np.absolute((par_new - par_old)) / 
                                np.absolute(par_old + 1e-8))
            converged = par_change < options.tol
            par_old = np.array(par_new)
            
            ###
            #If converged, break
            ###
            
            if i > 4 and converged:
                break
            
        ###
        #Compute ELBO
        ###
        
        s = 0.5
        rho = nu + self.n_rnd - 1
        
        diff_beta = beta_mu - beta_mu0
        diff_gamma = gamma_mu - mu_mu
        diff_mu = mu_mu - mu_mu0
        
        Sigma_BInv = np.linalg.inv(Sigma_B)
        Sigma_BInv_arr = np.array(Sigma_BInv).reshape(1,self.n_rnd,self.n_rnd)
        
        #Expectation of log joint
        elbo_lnp = self._elbo_lnp_nb(
            psi_mu, psi_sigma, psi_quad_x, psi_quad_w, r_b, r_c
            )
        if self.n_fix:
            elbo_lnp += -0.5 * diff_beta.T @ beta_Si0Inv @ diff_beta \
                -0.5 * np.trace(beta_Si0Inv @ beta_Si)
        if self.n_rnd:
            elbo_lnp += -self.N / 2 * self._log_det(Sigma_B) \
                - 0.5 * Sigma_rho * np.sum(
                    (diff_gamma.reshape(self.N,1,self.n_rnd) @ Sigma_BInv_arr \
                     @ diff_gamma.reshape(self.N,self.n_rnd,1))\
                        .reshape(self.N,)
                    + np.trace(Sigma_BInv_arr @ gamma_Si, axis1=1, axis2=2)
                    + np.trace(Sigma_BInv @ mu_Si)
                    )
            elbo_lnp += -0.5 * diff_mu.T @ mu_Si0Inv @ diff_mu \
                -0.5 * np.trace(mu_Si0Inv @ mu_Si)
            elbo_lnp += -(rho + self.n_rnd + 1) / 2 * self._log_det(Sigma_B) \
                - nu * Sigma_rho * np.sum(a_b / a_c * np.diag(Sigma_BInv))
            elbo_lnp += np.sum((1 - s) * np.log(a_c) - 1/A**2 * a_b / a_c) \
                -0.5 * rho * np.sum(np.log(a_c)) 
        
        elbo_lnp += 0.5 * self._log_det(Omega_tilde) \
            -0.5 * (phi_mu.T @ Omega_tilde @ phi_mu \
                    + np.trace(phi_Si @ Omega_tilde))
        elbo_lnp += -r0 * np.log(h_c) + (r0 - 1) * (digamma(r_b) - np.log(r_c))
        elbo_lnp += -h_b / h_c * r_b / r_c + (1 - b0) * np.log(h_c) \
            - c0 * h_b / h_c
        elbo_lnp += -(sigma2_b0 - 1) * np.log(sigma2) - sigma2_c0 / sigma2
        elbo_lnp += -(tau - tau_mu0)**2 / 2 / tau_si0**2 
        
        #Negative entropy
        elbo_lnq = 0
        if self.n_fix:
            elbo_lnq += -0.5 * self._log_det(beta_Si) 
        if self.n_rnd:
            elbo_lnq += -0.5 * np.sum(self._log_det(gamma_Si), axis=0) 
            elbo_lnq += -0.5 * self._log_det(mu_Si) 
            elbo_lnq += np.sum(np.log(a_c)) 
            elbo_lnq += -(self.n_rnd + 1) / 2 * self._log_det(Sigma_B) 
        elbo_lnq += -0.5 * self._log_det(phi_Si) 
        elbo_lnq += np.log(h_c) 
        elbo_lnq += -r_b + np.log(r_c) - loggamma(r_b) \
            - (1 - r_b) * digamma(r_b) 

        elbo = elbo_lnp - elbo_lnq
        
        print("tau: {:.2f}, sigma: {:.2f}, ELBO: {:.2f}"\
              .format(tau, sigma, elbo))
        sys.stdout.flush()
        
        ###
        #Results
        ###
        
        if self.n_fix == 0:
            beta_mu = None
            beta_Si = None
            
        if self.n_rnd == 0:
            gamma_mu = None
            gamma_Si = None
            mu_mu = None
            mu_Si = None
            Sigma_rho = None 
            Sigma_B = None  
            
        results = ResultsCaviInfvb(
            elbo,
            r_b, r_c,
            psi_mu, psi_sigma,
            beta_mu, beta_Si,
            gamma_mu, gamma_Si, mu_mu, mu_Si, Sigma_rho, Sigma_B,
            phi_mu, phi_Si,
            sigma, tau
            ) 
        
        return results
    
    def estimate_infvb(
            self,
            options, 
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_b_init, r_c_init, 
            beta_mu_init, beta_Si_init, 
            mu_mu_init, mu_Si_init,
            Sigma_B_init,
            phi_si_init,
            tau_grid, sigma_grid
            ):
        """ Performs INFVB estimation of negative binomial model. """
        
        ###
        #Grid evaluation
        ###
        
        psi_quad_x, psi_quad_w = np.polynomial.hermite.hermgauss(
            options.psi_quad_nodes
            )
        
        tau_sigma_grid = [(i,j) for i in tau_grid for j in sigma_grid]
        
        tic = time.time()
        
        grid_results = Parallel(n_jobs=options.infvb_n_jobs)\
            (delayed(self._cavi_infvb)(
            options, 
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_b_init, r_c_init, 
            beta_mu_init, beta_Si_init, 
            mu_mu_init, mu_Si_init,
            Sigma_B_init,
            phi_si_init,
            tau, sigma,
            psi_quad_x, psi_quad_w
            ) 
            for tau, sigma in tau_sigma_grid)
        
        elbo = np.array([r.elbo for r in grid_results])
        grid_weights_numer = np.exp(elbo - elbo.max())
        grid_weights = grid_weights_numer / grid_weights_numer.sum()
    
        toc = time.time() - tic
        
        grid = GridInfvb(
            tau_grid, sigma_grid, tau_sigma_grid, 
            grid_results, grid_weights, 
            toc)
        
        print(' ')
        print('Estimation time [s]: ' + str(toc))
                
        return grid
    
    @staticmethod
    def _simulation_summary(draws, param_name, verbose=True):
        q = (0.025, 0.975)
        df = pd.DataFrame()
        df['mean'] = draws.mean(axis=0).reshape(-1,)
        df['std. dev.'] = draws.std(axis=0).reshape(-1,)
        df['2.5%'] = np.quantile(draws, q[0], axis=0).reshape(-1,)
        df['97.5%'] = np.quantile(draws, q[1], axis=0).reshape(-1,)

        if verbose:
            print(' ')
            print(param_name + ':')
            print(df)
        return df  
    
    def simulate_infvb(self, options, grid):
        
        G = len(grid.tau_sigma)
        
        ###
        #Initialise storage for draws from variational distribution
        ###
        
        beta_store = np.zeros((options.infvb_sim_draws, self.n_fix))
        mu_store = np.zeros((options.infvb_sim_draws, self.n_rnd))
        Sigma_store = np.zeros((options.infvb_sim_draws, 
                                self.n_rnd, self.n_rnd))
        sigma_store = np.zeros((options.infvb_sim_draws, self.n_rnd))
        gamma_store = np.zeros((options.infvb_sim_draws, self.N, self.n_rnd))
        
        r_store = np.zeros((options.infvb_sim_draws,))
        
        tau_store = np.zeros((options.infvb_sim_draws,))
        sigma_mess_store = np.zeros((options.infvb_sim_draws,))
        phi_store = np.zeros((options.infvb_sim_draws, self.N))
         
        psi_store = np.zeros((options.infvb_sim_draws, self.N))
        lam_store = np.zeros((options.infvb_sim_draws, self.N))
        lp_store = np.zeros((options.infvb_sim_draws, self.N))
        
        ls_store = np.zeros((options.infvb_sim_draws,))
        dss_store = np.zeros((options.infvb_sim_draws,))
        rps_store = np.zeros((options.infvb_sim_draws,))
        
        ###
        #Simulate
        ###
        
        for i in np.arange(options.infvb_sim_draws):
            #Draw grid point
            g = int(np.random.choice(np.arange(G), size=1, p=grid.weights))
            rg = grid.results[g]
            
            #Take draws from sampled grid point
            if self.n_fix:
                beta = self._mvn_rnd(rg.beta_mu, rg.beta_Si)
            if self.n_rnd:
                mu = self._mvn_rnd(rg.mu_mu, rg.mu_Si)
                Sigma = invwishart.rvs(rg.Sigma_rho, rg.Sigma_B)
                sigma = np.sqrt(np.diag(Sigma))
                gamma = self._mvn_rnd_arr(rg.gamma_mu, rg.gamma_Si)
            
            r = np.random.gamma(rg.r_b, 1/rg.r_c)
            
            tau, sigma_mess = grid.tau_sigma[g]
            phi = self._mvn_rnd(rg.phi_mu, rg.phi_Si)
            
            psi = rg.psi_mu + rg.psi_sigma * np.random.randn(self.N,)
            lam = np.exp(psi + np.log(r))
            lp = self._nb_lpmf(self.data.y, psi, r)
            
            #Store
            if self.n_fix:
                beta_store[i,:] = beta
            if self.n_rnd:
                mu_store[i,:] = mu
                Sigma_store[i,:,:] = Sigma
                sigma_store[i,:] = sigma
                gamma_store[i,:,:] = gamma
            
            r_store[i] = r
            
            tau_store[i] = tau
            sigma_mess_store[i] = sigma_mess
            phi_store[i,:] = phi
            
            psi_store[i,:] = psi
            lam_store[i,:] = lam
            lp_store[i,:] = lp
            
            ls_store[i] = -np.sum(lp)
            dss_store[i] = self._dss(self.data.y, psi, r, lam)
            rps_store[i] = self._rps(self.data.y, psi, r)
            
        ###
        #Write draws to file
        ###
        
        file_name = '{}_draws_infvb.hdf5'.format(options.model_name)
        if os.path.exists(file_name):
            os.remove(file_name) 
        file = h5py.File(file_name, 'a')  
        
        if self.n_fix:
            file.create_dataset('beta_store', data=beta_store)
        if self.n_rnd:
            file.create_dataset('mu_store', data=mu_store)
            file.create_dataset('sigma_store', data=sigma_store)
            file.create_dataset('Sigma_store', data=Sigma_store)
        
        file.create_dataset('r_store', data=r_store)
        
        file.create_dataset('tau_store', data=tau_store)
        file.create_dataset('sigma_mess_store', data=sigma_mess_store)
        file.create_dataset('phi_store', data=phi_store)
        
        file.create_dataset('psi_store', data=psi_store)
        file.create_dataset('lam_store', data=lam_store)
        file.create_dataset('lp_store', data=lp_store)
        
        file.create_dataset('ls_store', data=ls_store)
        file.create_dataset('dss_store', data=dss_store)
        file.create_dataset('rps_store', data=rps_store)
            
        ###
        #Post-simulation analysis
        ###
        
        post_beta = None
        post_mu = None
        post_sigma = None 
        post_Sigma = None
        post_mean_gamma = None
        
        if self.n_fix:
            post_beta = self._simulation_summary(beta_store, 'beta')
        if self.n_rnd:
            post_mu = self._simulation_summary(mu_store, 'mu')
            post_sigma = self._simulation_summary(sigma_store, 'sigma')
            post_Sigma = self._simulation_summary(Sigma_store, 'Sigma')
            post_mean_gamma = gamma_store.mean(axis=0)
            
        post_r = self._simulation_summary(r_store, 'r')
        
        post_tau = self._simulation_summary(tau_store, 'tau')
        post_sigma_mess = self._simulation_summary(
            sigma_mess_store, 'sigma_mess'
            )
        post_phi = self._simulation_summary(phi_store, 'phi', verbose=False)
        
        lppd = np.log(np.exp(lp_store).mean(axis=0)).sum()
        p_waic = lp_store.var(axis=0).sum()
        waic = -2 * (lppd - p_waic)
        
        post_ls = self._simulation_summary(ls_store, 'ls')
        post_dss = self._simulation_summary(dss_store, 'dss')
        post_rps = self._simulation_summary(rps_store, 'rps')
        
        post_mean_lam = lam_store.mean(axis=0)
            
        rmse = np.sqrt(np.mean((post_mean_lam - self.data.y)**2))
        mae = np.mean(np.abs(post_mean_lam - self.data.y))
        rmsle = np.sqrt(np.mean((np.log1p(post_mean_lam) \
                                - np.log1p(self.data.y))**2))
            
        print(' ')
        print('RMSE: ' + str(rmse))
        print('MAE: ' + str(mae))
        print('RMSLE: ' + str(rmsle))
            
        ### 
        #Store results
        ###
            
        results = ResultsSimulationInfvb(
            options, grid.estimation_time,
            lppd, waic,
            post_ls, post_dss, post_rps,
            post_mean_lam, 
            rmse, mae, rmsle,
            post_r,
            post_beta,
            post_mu, post_sigma, post_Sigma, post_mean_gamma,
            post_sigma_mess, post_tau, post_phi
            )
            
        return results                       
    
class SyntheticData:
    """ Generates synthetic data to test MCMC method. """
    
    def __init__(self, 
                 N=200,
                 nneigh=5,
                 sigma=0.4,
                 tau=-0.6):
        self.N = N
        self.nneigh = nneigh
        self.sigma = sigma
        self.tau = tau
       
    def _draw_W_matrix(self):   
        """ Creates spatial weight matrix. """
        points = [np.random.rand(2,) for i in np.arange(self.N)]
        dist_mat = distance_matrix(points, points)
        nearest_points = np.array(np.argsort(dist_mat, axis=1)[:,1:])
        C = np.zeros((self.N, self.N))
        for i in np.arange(self.N):
            C[i,nearest_points[i,:self.nneigh]] = 1#/ np.arange(1, self.nneigh+1)
        W = C / C.sum(axis=1, keepdims=True)
        """
        C = np.zeros((self.N, self.N))   
        for i in np.arange(self.N-1):
            for j in np.arange(i+1, self.N):
                C[i,j] = np.random.rand() < (self.nneigh / self.N)
        C += C.T
        W = C / C.sum(axis=1, keepdims=True)
        """
        return W
        
    def generate(self, fixed, random, mess):
        """ Generates synthetic data. """
        
        #Fixed parameters
        if fixed:
            beta = np.array([1.0, 0.3, -0.3, 0.3])
            n_fix = beta.shape[0]
            x_fix = 2 * np.random.rand(self.N, n_fix)
            #x_fix[:,0] = 1
            psi_fix = x_fix @ beta        
        else:
            x_fix = None
            psi_fix = 0
            
        #Random parameters
        if random:
            gamma_mu = np.array([0.2, -0.2, 0.2])
            n_rnd = gamma_mu.shape[0]
            gamma_sd = np.sqrt(0.1 * np.abs(gamma_mu)) 
            #np.array([0.15, 0.15, 0.15])
            gamma_corr = 0.2 * np.array(
                [[0, 1, 0],
                 [1, 0, 1],
                 [0, 1, 0]]
                ) + np.eye(n_rnd)
            gamma_cov = np.diag(gamma_sd) @ gamma_corr @ np.diag(gamma_sd)
            gamma_ch = np.linalg.cholesky(gamma_cov)
            gamma = gamma_mu.reshape(1,n_rnd) \
                + (gamma_ch @ np.random.randn(n_rnd, self.N)).T
            """
            gamma_mu = np.array([0.5])
            n_rnd = gamma_mu.shape[0]
            gamma_sd = np.sqrt(0.5 * np.abs(gamma_mu))
            gamma = gamma_mu + gamma_sd * np.random.randn(self.N,n_rnd)
            """
            x_rnd = 1 * np.random.randn(self.N, n_rnd)
            psi_rnd = np.sum(x_rnd * gamma, axis=1)
        else:
            x_rnd = None
            psi_rnd = 0
        
        #Spatial error
        if mess:
            W = self._draw_W_matrix()
            S = expm(self.tau * W)
            eps = self.sigma * np.random.randn(self.N,)
            phi = np.linalg.solve(S, eps)
        else:
            W = None
            phi = 0
        
        #Link function
        psi = psi_fix + psi_rnd + phi
          
        #Success rate
        r = 1.5
        
        #Generate synthetic observations
        p = np.exp(psi) / (1 + np.exp(psi))
        y = np.random.negative_binomial(r, 1-p)
        
        ce = np.exp(psi + np.log(r))
        print('Avg. success prob.: {:.3f}'.format(np.mean(p)))
        print('Avg. cond. exp..: {:.3f}'.format(np.mean(ce)))
        if mess:
            print('Realised std. dev. phi:', phi.std())
        
        #Create data object
        data = Data(y, x_fix, x_rnd, 0, W)
        
        #Signal to noise
        if mess:
            signal = psi_fix + psi_rnd
            noise = phi
            signal_to_noise = np.mean(np.abs(signal / noise))
            print('Signal to noise: {:.3f}'.format(signal_to_noise))
            
        return data

###
#If main: test
###
    
if __name__ == "__main__":
    
    np.random.seed(4725)
    
    ###
    #Generate data
    ###   
    
    data = SyntheticData(N=500).generate(fixed=True, 
                                         random=True, 
                                         mess=True)
    
    ###
    #Estimate model via MCMC
    ###
    
    nb_model = NegativeBinomial(data, data_bart=None)
    
    options = OptionsMcmc(
            model_name='fixed',
            nChain=1, nBurn=4000, nSample=4000, nThin=1, nMem=None, 
            mh_step_initial=0.1, mh_target=0.3, mh_correct=0.01, 
            mh_window=100,
            disp=1000, delete_draws=False, seed=4711
            )
    bart_options = None
    
    r0 = 1e-2
    b0 = 1e-2
    c0 = 1e-2    
    beta_mu0 = np.zeros((data.n_fix,))
    beta_Si0Inv = 1e-2 * np.eye(data.n_fix)
    mu_mu0 = np.zeros((data.n_rnd,))
    mu_Si0Inv = 1e-2 * np.eye(data.n_rnd)
    nu = 2
    A = 1e3 * np.ones(data.n_rnd)
    sigma2_b0 = 1e-2 
    sigma2_c0 = 1e-2
    tau_mu0 = 0 
    tau_si0 = 100
    
    r_init = 5.0
    beta_init = np.zeros((data.n_fix,))
    mu_init = np.zeros((data.n_rnd,))
    Sigma_init = np.eye(data.n_rnd,)
    
    ranking_top_m_list = [100]
    
    results_mcmc = nb_model.estimate_mcmc(
        options, bart_options,
        r0, b0, c0,
        beta_mu0, beta_Si0Inv,
        mu_mu0, mu_Si0Inv, nu, A,
        sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
        r_init, beta_init, mu_init, Sigma_init,
        ranking_top_m_list)

    ###
    #Estimate model via VB
    ###
    
    nb_model = NegativeBinomial(data, data_bart=None)
    
    options = OptionsVb(
            model_name='test',
            max_iter=500, tol=0.005,
            infvb_n_jobs=1,
            psi_quad_nodes=10,
            tau_quad_nodes=10,
            r_sim_draws=1000,
            infvb_sim_draws=10000
            )
    
    r_b_init = 1000
    r_c_init = 500
    
    beta_mu_init = np.zeros(data.n_fix)
    beta_Si_init = 0.01 * np.eye(data.n_fix)
    
    mu_mu_init = np.zeros(data.n_rnd)
    mu_Si_init = 0.1 * np.eye(data.n_rnd)
    
    Sigma_B_init = data.N * np.eye(data.n_rnd)
    
    tau_mu_init = 0
    tau_si_init = 0.01
    sigma2_c_init = 62.5
    phi_si_init = 0.01
    
    """
    results_vb = nb_model.estimate_vb(
        options, 
        r0, b0, c0,
        beta_mu0, beta_Si0Inv,
        mu_mu0, mu_Si0Inv, nu, A,
        sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
        r_b_init, r_c_init, 
        beta_mu_init, beta_Si_init, 
        mu_mu_init, mu_Si_init,
        Sigma_B_init,
        tau_mu_init, tau_si_init,
        sigma2_c_init, phi_si_init
        )
    """
    
    """
    tau_grid = np.concatenate((
        np.linspace(-1,0,20,endpoint=False),
        np.linspace( 0,1,10+1,endpoint=True)
        ))
    """
    
    tau_grid = np.linspace(-2,0,20+1,endpoint=True)
    sigma_grid = np.linspace(0.2,2,20+1,endpoint=True)


    grid = nb_model.estimate_infvb(
        options, 
        r0, b0, c0,
        beta_mu0, beta_Si0Inv,
        mu_mu0, mu_Si0Inv, nu, A,
        sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
        r_b_init, r_c_init, 
        beta_mu_init, beta_Si_init, 
        mu_mu_init, mu_Si_init,
        Sigma_B_init,
        phi_si_init,
        tau_grid, sigma_grid
        )
    
    results = nb_model.simulate_infvb(options, grid)