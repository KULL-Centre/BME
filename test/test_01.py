import sys,os
import numpy as np

bme_dir = os.getcwd().split("test")[0]
sys.path.append(bme_dir)

import BME as BME
tol = 1.E-03
results_one = [[1.1524552410816327, 0.08783301229853915, 0.11401249782116127],\
               [1.1524552410816329, 1.6102870217553371, 7, 0.08783301229853926, 0.44454952218140253, 0],\
               [3.146911169909768, 0.44366636274376653, 16, 1.9907318847951587, 0.3523253434255751, 10]]

results_two = [[1.1532341544399012, 0.0193537697493589, 0.35833792938785797],\
               [3.146911169909768, 0.44366636274376653, 16, 0.026895813535419474, 0.056634715106665866, 0],\
               [1.1524552410816329, 1.6102870217553371, 7, 0.49709905557001793, 1.0575787795869112, 3]]

results_three = [[4.9257716215413625, 0.03270014130179259, 0.040946044724065614],\
                 [14.199966089013843, 0.9213070038949898, 22, 0.03801988712144421, 0.06569120511315368, 0],\
                 [1.1524552410816329, 1.6102870217553371, 7, 0.4126399382362971, 0.9635558421968434, 3]]

results_four = [[2.9766694437353234, 0.022832259699948863, 0.12751965316063574],\
                [8.559809220925167, 0.7209890327714169, 21, 0.02841231336087682, 0.05797616772314339, 0],\
                [1.1524552410816329, 1.6102870217553371, 7, 0.40447816776265333, 0.9539789711864566, 2]]

results_five = [[250.57610578123675, 2.839437835805958e-05, 0.44194271235643084],\
                [4.322245867831557, 0.20790011707143305, 13, 0.0, 0.0, 0]]

exp_file_3j = "%s/data/couplings_exp.dat" % bme_dir
calc_file_3j = "%s/data/couplings_calc.dat" % bme_dir

exp_file_noe = "%s/data/NOE_exp.dat" % bme_dir
calc_file_noe = "%s/data/NOE_calc.dat.zip" % bme_dir

exp_file_unoe = "%s/data/uNOE_exp.dat" % bme_dir
calc_file_unoe = "%s/data/uNOE_calc.dat.zip" % bme_dir

#exp_file_rdc = "%s/data/RDC_TL.exp.dat" % bme_dir
#calc_file_rdc = "%s/data/RDC_TL.calc.dat" % bme_dir

#exp_file_saxs = "%s/data/saxs.exp.dat" % bme_dir
#calc_file_saxs = "%s/data/saxs.calc.dat.zip" % bme_dir


def calc_diff(vref,v):
    return np.abs(np.max(np.array(vref)-np.array(v)))

class TestClass:
    
    def test_one(self):
        # initialize. A name must be specified 
        rew = BME.Reweight("test_01")
        print("")
        print("TEST scalar couplings  ",end="")
        # load the experimental and calculated datasets

        rew.load(exp_file_3j,calc_file_3j)
        
        # fit the data 
        results = rew.fit(theta=0.5)
        stats1 = rew.predict(exp_file_3j,calc_file_3j)
        stats2 = rew.predict(exp_file_noe,calc_file_noe)
        assert(calc_diff(results_one[0],results)<tol)
        assert(calc_diff(results_one[1],stats1)<tol)
        assert(calc_diff(results_one[2],stats2)<tol)
        
        print("..OK")
        
    def test_two(self):
        # initialize. A name must be specified
        print("TEST NOE  ",end="")
        
        rew = BME.Reweight("test_02")

        # load the experimental and calculated datasets
        rew.load(exp_file_noe,calc_file_noe)
        
        # fit the data 
        results = rew.fit(theta=0.5)
        stats1= rew.predict(exp_file_noe,calc_file_noe)
        stats2= rew.predict(exp_file_3j,calc_file_3j)
        
        assert(calc_diff(results_two[0],results)<tol)
        assert(calc_diff(results_two[1],stats1)<tol)
        assert(calc_diff(results_two[2],stats2)<tol)

        print("...OK")
        
    def test_three(self):
        # initialize. A name must be specified
        print("TEST NOE P3 ",end="")
        
        rew = BME.Reweight("test_02")

        # load the experimental and calculated datasets
        rew.load(exp_file_noe,calc_file_noe,averaging="power_3")
        
        # fit the data 
        results = rew.fit(theta=0.5)
        stats1= rew.predict(exp_file_noe,calc_file_noe,averaging="power_3")
        stats2= rew.predict(exp_file_3j,calc_file_3j)

        #print(results,stats1,stats2)
        assert(calc_diff(results_three[0],results)<tol)
        assert(calc_diff(results_three[1],stats1)<tol)
        assert(calc_diff(results_three[2],stats2)<tol)
        print("...OK")

    def test_four(self):
        # initialize. A name must be specified
        print("TEST NOE P4 ",end="")
        
        rew = BME.Reweight("test_02")

        # load the experimental and calculated datasets
        rew.load(exp_file_noe,calc_file_noe,averaging="power_4")
        
        # fit the data 
        results = rew.fit(theta=0.5)
        stats1= rew.predict(exp_file_noe,calc_file_noe,averaging="power_4")
        stats2= rew.predict(exp_file_3j,calc_file_3j)

        #print(results,stats1,stats2)
        assert(calc_diff(results_four[0],results)<tol)
        assert(calc_diff(results_four[1],stats1)<tol)
        assert(calc_diff(results_four[2],stats2)<tol)
        print("...OK")

    def test_five(self):
        # initialize. A name must be specified
        print("TEST uNOE ",end="")
        
        rew = BME.Reweight("test_02")

        # load the experimental and calculated datasets
        rew.load(exp_file_unoe,calc_file_unoe)
        
        # fit the data 
        results = rew.fit(theta=0.5)
        stats1= rew.predict(exp_file_unoe,calc_file_unoe)

        assert(calc_diff(results_five[0],results)<tol)
        assert(calc_diff(results_five[1],stats1)<tol)
        #assert(calc_diff(results_four[2],stats2)<tol)
        print("...OK")

        

# test uNOE

# test RDC

# test RDC w weights

# test SAXS

# test iBME

# test cross-validation

# test weights on datasets



    
