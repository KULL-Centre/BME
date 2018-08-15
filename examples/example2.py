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

# initalize class, reads experimental and simulated data.
n_frames = 20000
n_bins = 5
bin_size = n_frames/n_bins

theta=2
end = 0
start = 0

# split in 5 blocks using awk
for j in range(n_bins):
    
    end += bin_size
    
    calc_noe_bin = "%s.%d" % (calc_noe,j)
    cmd = "awk '{ if(NR>%d && NR<=%d) print $0}' %s > %s" % (start,end,calc_noe,calc_noe_bin)
    os.system(cmd)
    
    calc_unoe_bin = "%s.%d" % (calc_unoe,j)
    cmd = "awk '{ if(NR>%d && NR<=%d) print $0}' %s > %s" % (start,end,calc_unoe,calc_unoe_bin)
    os.system(cmd)
    
    calc_couplings_bin = "%s.%d" % (calc_couplings,j)
    cmd = "awk '{ if(NR>%d && NR<=%d) print $0}' %s > %s" % (start,end,calc_couplings,calc_couplings_bin)
    os.system(cmd)
    
    rew = rr.Reweight([exp_couplings],[calc_couplings_bin])
    rew.optimize(theta=theta)
    
    rew.weight_exp([exp_noe,exp_unoe,exp_couplings],[calc_noe_bin, calc_unoe_bin, calc_couplings_bin ] , 'example2_%d' % (j))

    start += bin_size
