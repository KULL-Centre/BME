# import the reweighting script
import os
import sys
bme_path = os.getcwd()[:-8]
# here append the path to the bme script
sys.path.append(bme_path)
import bme_reweight as bme

# define name of experimental datafiles
exp_couplings = '../data/couplings_exp.dat'
exp_noe='../data/NOE_exp.dat'

# define name of experimental quantities # calculated from the trajectory
calc_couplings = '../data/couplings_calc.dat'
calc_noe='../data/NOE_calc.dat'

# initialize reweighting class 
rew = bme.Reweight()

# load data
rew.load(exp_couplings,calc_couplings)

# do optimization using theta=2
chib,chia, srel = rew.optimize(theta=2)

# compare NOE  before and after optimization .
# output is written to files with prefix 'example1_noe'
chi2_b,chi2_a = rew.weight_exp(exp_noe,calc_noe, 'example1_noe')

# print chisquared
print  srel
print chib, chia
print chi2_a, chi2_b
