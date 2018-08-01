import os
import sys
cwd = os.getcwd()[:-8]
sys.path.append(cwd)
import reweight as rr

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
bin_size = n_frames/n_bins

thetas= [0.1,0.5,1,2,3,4,5,7.5,10,15,20,40,50,100,500,5000]


for t in thetas:
    for j in range(n_bins):
    
        calc_noe_bin = "%s.%d" % (calc_noe,j)
        calc_unoe_bin = "%s.%d" % (calc_unoe,j)
        calc_couplings_bin = "%s.%d" % (calc_couplings,j)
        
        rew = rr.Reweight([exp_couplings],[calc_couplings_bin])
        rew.optimize(theta=t)
        
        rew.weight_exp([exp_noe,exp_unoe,exp_couplings],[calc_noe_bin, calc_unoe_bin, calc_couplings_bin ] , 'example3_%.1f_%d' % (t,j))
