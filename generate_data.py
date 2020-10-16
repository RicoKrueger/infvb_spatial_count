import negative_binomial as nb
import numpy as np
import pickle
import itertools as it

np.random.seed(4711)

R = 10
N_list = [1000, 1500] #1000, 1500
sigma_list = [0.2, 0.4] 
tau_list = [-0.7, 0.7]

sample_plan = {}
t = 0

for n, sigma, tau, r in it.product(N_list, sigma_list, tau_list, np.arange(R)):
    t += 1
    
    data = nb.SyntheticData(N=n, nneigh=8, sigma=sigma, tau=tau).generate(
        fixed=True, 
        random=True, 
        mess=True
        )
    
    filename = 'synthetic_data_N{}_sigma{}_tau{}_r{}'.format(n, sigma, tau, r)
    outfile = open(filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()
    
    sample_plan[t] = {'N': n, 'sigma': sigma, 'tau': tau, 'r': r}

filename = 'sample_plan'
outfile = open(filename, 'wb')
pickle.dump(sample_plan, outfile)
outfile.close()
