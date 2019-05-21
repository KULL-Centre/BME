# import the reweighting script
import os
import sys
bme_path = os.getcwd()[:-8]
# here append the path to the bme script
sys.path.append(bme_path)
import bme_reweight as bme
import numpy as np

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

# compare NOE and jcouplings before and after optimization .
# output is written to files with prefix 'example1_noe'
chi2_b_noe,chi2_a_noe = rew.weight_exp(exp_noe,calc_noe, 'example1_noe')
chi2_b_j3,chi2_a_j3 = rew.weight_exp(exp_couplings,calc_couplings, 'example1_couplings')

# print chisquared
print("Relative entropy: %5.2f" % srel)
print("Fraction of effective frames  : %5.2f" % np.exp(srel))
print("NOE chi2 BEFORE optimization  : %5.2f" % chi2_b_noe)
print("NOE chi2 AFTER optimization   : %5.2f" % chi2_a_noe)

print("3J chi2 BEFORE optimization   : %5.2f" % chi2_b_j3)
print("3J chi2 AFTER optimization    : %5.2f" % chi2_a_j3)

#print(chib, chia)
#print(chi2_a, chi2_b)
