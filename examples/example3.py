import os
import sys
cwd = os.getcwd()[:-8]
sys.path.append(cwd)
import bme_reweight as bme

# define name of experimental datafiles
exp_noe='../data/NOE_exp.dat'
exp_unoe='../data/uNOE_exp.dat'
exp_couplings = '../data/couplings_exp.dat'

# define name of experimental quantities # calculated from the trajectory
calc_noe='../data/NOE_calc.dat'
calc_unoe='../data/uNOE_calc.dat'
calc_couplings = '../data/couplings_calc.dat'

n_frames = 20000
n_bins = 5
bin_size = int(n_frames/n_bins)
bins = range(0,n_frames+bin_size,bin_size)

thetas= [0.1,0.5,1,2,3,4,5,7.5,10,15,20,40,50,100,500,5000]


for t in thetas:
    for j in range(len(bins)-1):    
        rows = range(bins[j],bins[j+1])
    
        rew = bme.Reweight()
        rew.load(exp_couplings,calc_couplings,rows=rows)
        rew.optimize(theta=t)

        rew.weight_exp(exp_noe,calc_noe, 'example3_%d_%.1f_noe' % (j,t),rows=rows)
        rew.weight_exp(exp_couplings,calc_couplings, 'example3_%d_%.1f_couplings' % (j,t),rows=rows)

        
