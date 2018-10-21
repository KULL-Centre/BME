# import the reweighting script
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


# initialize reweighting class : use only couplings as input
rew = bme.Reweight()
rew.load(exp_couplings,calc_couplings)

##### HERE SET THE INITIAL WEIGHTS FROM FILE #####
rew.set_w0([float(line.split()[0]) for line in open("../data/weights.dat")])
##### HERE SET THE INITIAL WEIGHTS FROM FILE #####

# do optimization using theta=2
rew.optimize(theta=2.0)

# compare NOE, uNOE and J Couplings before and after optimization .
# output is written to files with prefix 'example1'
rew.weight_exp(exp_noe,calc_noe , 'example4_noe')
rew.weight_exp(exp_couplings,calc_couplings, 'example4_couplings')


