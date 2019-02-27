# import the reweighting script
import os
import sys
import numpy as np
bme_path = os.getcwd().split("BME")[0] + "BME/"
#print(bme_path)
# here append the path to the bme script
sys.path.append(bme_path)
import bme_reweight as bme
from comp_mine import comp

outdir = "%s/test/tmp" % bme_path
refdir = "%s/test/reference/" % bme_path
os.system("mkdir -p %s" % (outdir))

num=24
spec=" Train with J3, check that gives the same results as Bayesian ensemble refinement (BER)"

# define training test
exp_train = '%s/data/couplings_exp.dat' % (bme_path)
calc_train = '%s/data/couplings_calc.dat' % (bme_path)



def test():

    print("# TEST %02d: %s" % (num,spec))
    
    # initialize reweighting class
    rew_bme = bme.Reweight()
    rows = range(0,20000,100)
    # load data
    rew_bme.load(exp_train,calc_train,rows=rows)
    
    # do optimization using theta=2, method MAXENT (default)
    chib,chia, srel = rew_bme.optimize(theta=2,method="MAXENT")
    ww_bme = rew_bme.get_weights()
    
    # initialize reweighting class
    rew_eros = bme.Reweight()

    # load data
    rew_eros.load(exp_train,calc_train,rows=rows)
    
    # do optimization using theta=2
    chib,chia, srel = rew_eros.optimize(theta=2,method="BER")
    ww_ber = rew_eros.get_weights()

    diff=np.sqrt(np.sum(((ww_ber-ww_bme)**2))/len(ww_ber))
    print(" rms difference between BER and BME: %10.5e" % diff)


    fh = open("%s/test_%02d_weights_bme.dat" % (outdir,num) ,"w")
    fh.write(" ".join(["%10.5e \n" % x for x in ww_bme]) )
    fh.close()

    fh = open("%s/test_%02d_weights_ber.dat" % (outdir,num) ,"w")
    fh.write(" ".join(["%10.5e \n" % x for x in ww_ber]) )
    fh.close()

    comp("%s/test_%02d_weights_ber.dat" %  (refdir,num))
    comp("%s/test_%02d_weights_bme.dat" %  (refdir,num))
    
    print("# TEST %02d: %s" % (num,spec))
    print("# DONE #")
