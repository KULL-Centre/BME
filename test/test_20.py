# import the reweighting script
import os
import sys

bme_path = os.getcwd().split("BME")[0] + "BME/"

# here append the path to the bme script
sys.path.append(bme_path)
import bme_reweight as bme
from comp_mine import comp

outdir = "%s/test/tmp" % bme_path
refdir = "%s/test/reference/" % bme_path
os.system("mkdir -p %s" % (outdir))

num=20
spec="Train RDC, test NOE, use non-uniform weights"


# define test set 
exp_train='%s/data/RDC_TL.exp.dat' % (bme_path)
calc_train='%s/data/RDC_TL.calc.dat' % (bme_path)

# define training test
exp_test = '%s/data/NOE_TL.exp.dat' % (bme_path)
calc_test = '%s/data/NOE_TL.calc.dat' % (bme_path)
bias_file = '%s/data/w0.dat' % (bme_path)

# kbt
kbt = 0.008314462*280.

def test():

    print("# TEST %02d: %s" % (num,spec))
    
    # initialize reweighting class, use non-uniform initial weights
    bias = [float(line.split()[1]) for line in open(bias_file)]
    
    rew = bme.Reweight(w0=bias,kbt=kbt)

    # load data
    rew.load(exp_train,calc_train)
    
    # do optimization using theta=2
    chib,chia, srel = rew.optimize(theta=1)

    # compare NOE  before and after optimization .
    # output is written to files with prefix 'example1_noe'
    chi2_b,chi2_a = rew.weight_exp(exp_train,calc_train, '%s/test_%02d_train' % (outdir,num))
    chi2_b,chi2_a = rew.weight_exp(exp_test,calc_test, '%s/test_%02d_test' % (outdir,num))

    ww = rew.get_weights()
    fh = open("%s/test_%02d_weights.dat" % (outdir,num) ,"w")
    fh.write(" ".join(["%8.4f " % x for x in ww]) )
    fh.close()

    comp("%s/test_%02d_weights.dat" %  (refdir,num))
    comp("%s/test_%02d_train.stats.dat" % (refdir,num))
    comp("%s/test_%02d_test.stats.dat" % (refdir,num))
    print("# TEST %02d: %s" % (num,spec))
    print("# DONE #")
