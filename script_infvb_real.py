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
#Estimate model via INFVB
###

nb_model = nb.NegativeBinomial(data, data_bart=None)

options = nb.OptionsVb(
        model_name='random_mess_{}'.format(sample),
        max_iter=500, tol=0.005,
        infvb_n_jobs=1,
        psi_quad_nodes=10,
        tau_quad_nodes=10,
        r_sim_draws=2000,
        infvb_sim_draws=10000
        )

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

r_b_init = 1000
r_c_init = 500

beta_mu_init = np.zeros(data.n_fix)
beta_Si_init = 0.01 * np.eye(data.n_fix)

mu_mu_init = np.zeros(data.n_rnd)
mu_Si_init = 0.1 * np.eye(data.n_rnd)

Sigma_B_init = data.N * np.eye(data.n_rnd)

phi_si_init = 0.1

tau_grid = np.linspace(-1.4,0,14+1,endpoint=True)
sigma_grid = np.linspace(0.05,0.8,10,endpoint=True)

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

filename = 'results_infvb_{}'.format(sample)
outfile = open(filename, 'wb')
pickle.dump(results, outfile)
outfile.close()
