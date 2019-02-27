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

num=7
spec="Train RDC, NOE, J3. No test"


# define test set 
exp_train1='%s/data/RDC_TL.exp.dat' % (bme_path)
calc_train1='%s/data/RDC_TL.calc.dat' % (bme_path)

# define test set 2
exp_train2 = '%s/data/NOE_TL.exp.dat' % (bme_path)
calc_train2 = '%s/data/NOE_TL.calc.dat' % (bme_path)

# define test set 3
exp_train3 = '%s/data/J3_TL.exp.dat' % (bme_path)
calc_train3 = '%s/data/J3_TL.calc.dat' % (bme_path)


def test():

    print("# TEST %02d: %s" % (num,spec))
    
    # initialize reweighting class
    rew = bme.Reweight()

    # load data
    rew.load(exp_train1,calc_train1)
    rew.load(exp_train2,calc_train2)
    rew.load(exp_train3,calc_train3)
    
    # do optimization using theta=2
    chib,chia, srel = rew.optimize(theta=50)

    # compare NOE  before and after optimization .
    # output is written to files with prefix 'example1_noe'
    chi2_b,chi2_a = rew.weight_exp(exp_train1,calc_train1, '%s/test_%02d_train1' % (outdir,num))
    chi2_b,chi2_a = rew.weight_exp(exp_train2,calc_train2, '%s/test_%02d_train2' % (outdir,num))
    chi2_b,chi2_a = rew.weight_exp(exp_train3,calc_train3, '%s/test_%02d_train3' % (outdir,num))


    ww = rew.get_weights()
    fh = open("%s/test_%02d_weights.dat" % (outdir,num) ,"w")
    fh.write(" ".join(["%8.4f " % x for x in ww]) )
    fh.close()

    comp("%s/test_%02d_weights.dat" %  (refdir,num))
    comp("%s/test_%02d_train1.stats.dat" % (refdir,num))
    comp("%s/test_%02d_train2.stats.dat" % (refdir,num))
    comp("%s/test_%02d_train3.stats.dat" % (refdir,num))
    print("# TEST %02d: %s" % (num,spec))
    print("# DONE #")
