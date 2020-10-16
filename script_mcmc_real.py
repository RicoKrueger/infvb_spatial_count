import negative_binomial as nb
import pandas as pd
import numpy as np
import os
import pickle

###
#Load and process data
###

sample = 'real'

df = pd.read_csv('crash_data.csv')

df['intercept'] = 1
df['log_traffic'] = np.log(df['avg_ann_daily_traffic'])
df['log_workers'] = np.log(1 + df['workers_per_km2'])

predictors = [
    'black', 'poor', 'commute_priv_vehicle',
    'fragment_index', 'log_workers', 'log_traffic' 
    ]
scale = lambda x: (x - x.mean()) / x.std() if x.name in predictors else x
df = df.apply(scale, axis=0)

infile = open('weight_matrix', 'rb')
W = pickle.load(infile)
infile.close()

data = nb.Data(y=np.array(df['ped_injury_5to18']), 
               x_fix=np.array(df[[
                   'intercept',
                   'black', 'poor', 'log_workers'
                   ]]), 
               x_rnd=np.array(df[[
                   'fragment_index', 'commute_priv_vehicle', 'log_traffic'
                   ]]), 
               offset=0,
               W=W)

###
#Estimate model via MCMC
###

nb_model = nb.NegativeBinomial(data, data_bart=None)

options = nb.OptionsMcmc(
        model_name='random_mess_{}'.format(sample),
        nChain=2, nBurn=20000, nSample=20000, nThin=5, nMem=None, 
        mh_step_initial=0.1, mh_target=0.44, mh_correct=0.01, 
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