import os
import sys
cwd = os.getcwd()[:-8]
sys.path.append(cwd)
import bme_reweight as bme

# define name of experimental datafiles
exp_noe='../data/NOE_exp.dat'
exp_couplings = '../data/couplings_exp.dat'

# define name of experimental quantities # calculated from the trajectory
calc_noe='../data/NOE_calc.dat'
calc_couplings = '../data/couplings_calc.dat'

# initalize class, reads experimental and simulated data.
n_frames = 20000
n_bins = 5
bin_size = n_frames/n_bins
bins = range(0,n_frames+bin_size,bin_size)

theta=2

# split in 5 blocks 
for j in range(len(bins)-1):
    
    rows = range(bins[j],bins[j+1])
    rew = bme.Reweight()
    rew.load(exp_couplings,calc_couplings,rows=rows)
    rew.optimize(theta=theta)
    
    rew.weight_exp(exp_noe,calc_noe, 'example2_%d_noe' % (j),rows=rows)
    rew.weight_exp(exp_couplings,calc_couplings, 'example2_%d_couplings' % (j),rows=rows)

    w_opt = rew.get_weights()

    # write weights to file
    w0 = [1./len(w_opt)]*len(w_opt)
    string = "".join([ "%10.4e %10.4e \n " % (w0[k],w_opt[k])for k in range(len(w_opt))])
    fh = open("example2_%d.weights.dat" % j,"w")
    fh.write(string)
    fh.close()
    
    
