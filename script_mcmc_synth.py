import negative_binomial as nb
import pandas as pd
import numpy as np
import os
import pickle

###
#Obtain task and load data
###

task = int(os.getenv('TASK'))

infile = open('sample_plan', 'rb')
sp = pickle.load(infile)[task]
infile.close()

sample = 'N{}_sigma{}_tau{}_r{}'.format(sp['N'], sp['sigma'], sp['tau'], sp['r'])
infile = open('synthetic_data_{}'.format(sample), 'rb')
data = pickle.load(infile)
infile.close()

###
#Estimate model via MCMC
###

nb_model = nb.NegativeBinomial(data, data_bart=None)

options = nb.OptionsMcmc(
        model_name='random_mess_{}'.format(sample),
        nChain=2, nBurn=20000, nSample=20000, nThin=5, nMem=None, 
        mh_step_initial=0.1, mh_target=0.44, mh_correct=0.01, 
        mh_window=100,
        disp=1000, delete_draws=True, seed=4711
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
sigma2_b0 = 1e-3
sigma2_c0 = 1e-3
tau_mu0 = 0
tau_si0 = 10

r_init = 2.0
beta_init = np.zeros((data.n_fix,))
mu_init = np.zeros((data.n_rnd,))
Sigma_init = np.eye(data.n_rnd,)

ranking_top_m_list = [100]

results = nb_model.estimate_mcmc(
    options, bart_options,
    r0, b0, c0,
    beta_mu0, beta_Si0Inv,
    mu_mu0, mu_Si0Inv, nu, A,
    sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
    r_init, beta_init, mu_init, Sigma_init,
    ranking_top_m_list)

filename = 'results_mcmc_{}'.format(sample)
outfile = open(filename, 'wb')
pickle.dump(results, outfile)
outfile.close()