import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


exp_types = ["NOE","JCOUPLINGS","CS","SAXS","RDC"]
bound_types = ["UPPER","LOWER"]
averaging_types = ["linear","power_3","power_6","power_4"]
fit_types = ["scale","scale+offset","no"]


# calculate relative entropy
def srel(w0,w1):
    idxs = np.where(w1>1.0E-50)
    return np.sum(w1[idxs]*np.log(w1[idxs]/w0[idxs]))

# parse files
def parse(exp_file,calc_file,averaging="auto"):

    log = ""
    # read experimental data 
    fh = open(exp_file)
    first = fh.readline()
    assert first[0] == "#", "Error. First line of exp file %s must be in the format \# DATA=[%s] [BOUND=UPPER/LOWER]" % (exp_file,exp_types)

    # read data type
    data_string = (first.split("DATA=")[-1].split()[0]).strip()
    assert data_string in exp_types , "Error. DATA in %s must be one of the following: %s " % (exp_file,exp_types)

    log += "# Reading %s data \n" % (data_string)
        
    # If it is not an average but a boundary it can be specified
    bound_string=None
    if(len(first.split("BOUND="))==2):
        bound_string = (first.split("BOUND=")[-1].split()[0]).strip()
        assert bound_string in bound_types , "Error. %s is not known. BOUND in %s must be one of the following: %s " % (bound_string,exp_file,bound_types)
        log += "# %s-bound data \n" % (bound_string)
        
    df_exp = pd.read_csv(exp_file,sep="\s+",header=None,comment="#")
        
    assert df_exp.shape[1]==3, "Error. Experimental datafile must be in the format LABEL VALUE ERROR"
    df_exp = df_exp.rename(columns={0: "label", 1: "val",2:"sigma"})
        
    # read calculated datafile
    df_calc = pd.read_csv(calc_file,sep="\s+",header=None,comment="#")
    # Drop frame
    df_calc = df_calc.drop(columns=[0])

    assert (df_calc.shape[1])==df_exp.shape[0],\
        "Error: Number of experimental data in %s (%d) must match the calculated data in %s (%d)" % (exp_file,df_exp.shape[0],calc_file,df_calc.shape[1])

    # write to log file
    log += "# Reading %d experimental data from %s \n" % (df_exp.shape[0],exp_file)
    log += "# Reading %d calculated samples from %s \n" % (df_calc.shape[0],calc_file)
        
            
    # determine averaging
    if(averaging=="auto"):
        if(data_string=="NOE"):
            averaging = "power_6"
        else:
            averaging = "linear"
    else:
        assert averaging in averaging_types, "averaging type must be in %s " % (averaging_types)
    log += "# Using %s averaging \n" % (averaging)

    if(averaging.split("_")[0]=="power"):
        noe_power = int(averaging.split("_")[-1])
        df_exp["avg"] = np.power(df_exp["val"], -noe_power)
        df_exp["sigma2"] = (noe_power*df_exp["avg"]*df_exp["sigma"]/(df_exp["val"]))
        df_calc = np.power(df_calc, -noe_power)
            
        # if bound constraints, swap lower and upper
        if(bound_string=="LOWER"):
            bound_string="UPPER"
        elif(bound_string=="UPPER"):
            bound_string="LOWER"
    else:
        df_exp = df_exp.rename(columns={"val": "avg"})
        df_exp = df_exp.rename(columns={"sigma":"sigma2"})

    # define bounds 
    df_exp["bound"] = 0
    if(bound_string=="UPPER"): df_exp["bound"] = 1.0
    if(bound_string=="LOWER"): df_exp["bound"] = -1.0
    #df_exp["tag"] = exp_file

    labels = df_exp["label"].values
    exp = np.array(df_exp[["avg","sigma2","bound"]].values)
    calc = np.array(df_calc.values)
    return labels,exp,calc,log,averaging


# perform linear regression 
def fit_and_scale(exp,calc,sample_weights,fit):

    assert fit in fit_types, "fit type must be in %s " % (fit_types)

    calc_avg = np.sum(calc*sample_weights[:,np.newaxis],axis=0)
    log = "# Using %s scaling \n" % (fit)
    if(fit=="no"):
        return calc_avg,log
    else:
        exp_avg = exp[:,0]

        if(fit=="scale"):
            fit_intercept=False
        else:
            fit_intercept=True

        oversigma = (1./exp[:,1]**2)

        reg = LinearRegression(fit_intercept=fit_intercept).fit(calc_avg.reshape(-1,1),exp_avg.reshape(-1,1),sample_weight=oversigma)
        r_value = reg.score(calc_avg.reshape(-1,1),exp_avg.reshape(-1,1),sample_weight=oversigma)
        slope,intercept = reg.coef_[0],reg.intercept_
               
        calc *=slope
        calc +=intercept

        calc_avg = np.sum(calc*sample_weights[:,np.newaxis],axis=0).reshape(-1,1)
        log = "# Slope=%8.4f; Offset=%8.4f; r2=%8.4f \n" % (slope,intercept,r_value)
        return calc_avg,log


# remove some of the samples
def subsample(label,exp,calc,use_samples,use_data):
    
    log = ""
    if(len(use_samples)!=0):
        calc = calc[use_samples,:]
        log += "# Using a subset of samples (%d) \n" % (calc.shape[0])
    if(len(use_data)!=0):
        label = label[use_data]
        exp = exp[use_data,:]
        calc = calc[:,use_data]
        log += "# Using a subset of datapoints (%d) \n" % (exp.shape[0])
    
    return label,exp, calc,log

# calculate chi2
def calc_chi(exp,calc,sample_weights):
    
    calc_avg = np.sum(calc*sample_weights[:,np.newaxis],axis=0)

    diff = (calc_avg-exp[:,0])    
    ii = np.where(((diff<0) & (exp[:,2]<0)) | ((diff>0) & (exp[:,2]>0)) )[0]
    ff = [1 if (exp[j,2]==0 or j in ii) else 0 for j in range(exp.shape[0])]
    diff *= ff #to_zero
    return  np.average((diff/exp[:,1])**2)

# sanity check 
def check_data(label,exp,calc,sample_weights):

    calc_avg = np.sum(calc*sample_weights[:,np.newaxis],axis=0)

    log = ""

    diff = (calc_avg-exp[:,0])
    ii = np.where(((diff<0) & (exp[:,2]<0)) | ((diff>0) & (exp[:,2]>0)) )[0]
    ff = [1 if (exp[j,2]==0) else 0 for j in range(exp.shape[0])]
    nviol = 0
    if(len(ii)>0):
        log += "# The ensemble violates the following %d boundaries: \n" % (len(ii))
        log += "# %14s %8s %8s \n" % ("label","exp_avg","calc_avg")
        for j in ii:
            log  += "# %14s %8.4f %8.4f \n" % (label[j],exp[j,0],calc_avg[j])
            ff[j] = 1
            nviol += 1
    diff *= ff #to_zero
    chi2 = np.average((diff/exp[:,1])**2)
    rmsd = np.sqrt(np.average(diff**2))
    ii_out =  np.where(diff>1.)[0]
    nviol += len(ii_out)
    log += "CHI2: %.5f \n" % chi2
    log += "RMSD: %.5f \n" % rmsd
    log += "VIOLATIONS: %d \n" % nviol

    m_min = np.min(calc,axis=0)
    m_max = np.max(calc,axis=0)


    diff_min = ff*(m_min-exp[:,0])/exp[:,1]
    ii_min = np.where(diff_min>1.)[0]
    if(len(ii_min)>0):
        log += "##### WARNING ########## \n"
        log += "# The minimum value of the following data is higher than expt range: \n"
        log += "# %14s %8s %8s %8s \n" % ("label","exp_avg","exp_sigma","min_calc")
        for j in ii_min:
            log += "# %15s %8.4f %8.4f %8.4f\n" % (label[j],exp[j,0],exp[j,1],m_min[j])

    diff_max = ff*(exp[:,0]-m_max)/exp[:,1]
    ii_max = np.where(diff_max>1.)[0]
    if(len(ii_max)>0):
        log += "##### WARNING ########## \n"
        log += "# The maximum value of the following data is lower than expt range: \n"
        log += "# %14s %8s %8s %8s \n" % ("label","exp_avg","exp_sigma","max_calc")
        for j in ii_max:
            log += "# %15s %8.4f %8.4f %8.4f\n" % (label[j],exp[j,0],exp[j,1],m_max[j])
                          
    return log


# calculate Chi2, RMSD, number of violations
def calc_diffs(exp_avg,exp_sigma,bounds,calc_avg):

    diff = (calc_avg-exp_avg)/exp_sigma
    # boundaries violations
    ii = np.where(((diff<-1.) & (bounds<0)) | ((diff>1.) & (bounds>0)) )[0]
    ff = [1 if (bounds[j]==0 or j in ii) else 0 for j in range(exp_avg.shape[0])]
    # other violations
    viol = np.where(((diff**2>1) & (bounds==0)) )[0]
    #to_zero = np.zeros(len(diff))
    viol_idx = np.array(list(ii)+list(viol))
    nviol = viol_idx.shape[0]
    diff *= ff #to_zero
    chi2 = np.average((diff)**2)
    rmsd = np.sqrt(np.average((diff*exp_sigma)**2))
    return chi2,rmsd,nviol,viol_idx

# calculate statistics
def calc_stats(label,exp,calc,sample_weights0,sample_weights1,averaging,fit,outfile):
    
    assert averaging in averaging_types, "averaging type must be in %s " % (averaging_types)
    
    calc_avg0 = np.sum(calc*sample_weights0[:,np.newaxis],axis=0)
    calc_avg1 = np.sum(calc*sample_weights1[:,np.newaxis],axis=0)
    
    exp_avg = exp[:,0]
    oversigma = (1./exp[:,1]**2)
    #oversigma = np.ones(len(exp))
    intercept0,intercept1 = 0.,0.
    slope0,slope1 = 1.,1.
    
    if(fit=="scale"):
        slope0 = np.dot(exp_avg,calc_avg0)/np.dot(calc_avg0,calc_avg0)
        slope1 = np.dot(exp_avg,calc_avg1)/np.dot(calc_avg1,calc_avg1)        
    elif(fit=="scale+offset"):
        #slope0, intercept0, r_value0, p_value0, std_err0 = linregress(calc_avg0[:,0],exp[:,0])
        #slope1, intercept1, r_value1, p_value1, std_err1 = linregress(calc_avg1[:,0],exp[:,0])
        reg0 = LinearRegression(fit_intercept=True).fit(calc_avg0.reshape(-1,1),exp_avg.reshape(-1,1),sample_weight=oversigma)
        slope0,intercept0 = reg0.coef_[0],reg0.intercept_
        
        reg1 = LinearRegression(fit_intercept=True).fit(calc_avg1.reshape(-1,1),exp_avg.reshape(-1,1),sample_weight=oversigma)
        slope1,intercept1 = reg1.coef_[0],reg1.intercept_


    log = "# Slope=%8.4f/%8.4f; Offset=%8.4f/%8.4f \n" % (slope0,slope1,intercept0,intercept1)

    if(averaging.split("_")[0]=="power"):
        noe_power = int(averaging.split("_")[-1])

        calc_avg0 = np.power(np.sum((slope0*calc+intercept0)*sample_weights0[:,np.newaxis],axis=0),-1/noe_power)
        calc_avg1 = np.power(np.sum((slope1*calc+intercept1)*sample_weights1[:,np.newaxis],axis=0),-1/noe_power)
        exp_avg = np.power(exp[:,0],-1/noe_power)
        exp_sigma = (exp_avg*exp[:,1])/(noe_power*exp[:,0])
        bounds = -exp[:,2]
    else:
        calc_avg0 = np.sum((slope0*calc+intercept0)*sample_weights0[:,np.newaxis],axis=0)
        calc_avg1 = np.sum((slope1*calc+intercept1)*sample_weights1[:,np.newaxis],axis=0)

        exp_avg = exp[:,0]
        exp_sigma = exp[:,1]
        bounds = exp[:,2]
    log += "# %s averaging \n" % (averaging)
        
    
    chi2_0,rmsd_0,nviol_0, viol_idx_0 = calc_diffs(exp_avg,exp_sigma,bounds,calc_avg0)
    chi2_1,rmsd_1,nviol_1, viol_idx_1 = calc_diffs(exp_avg,exp_sigma,bounds,calc_avg1)

    if(outfile!=None):
        violation0 = [1 if k in viol_idx_0 else 0 for k in range(len(calc_avg0))]
        violation1 = [1 if k in viol_idx_1 else 0 for k in range(len(calc_avg1))] 
        violation = ["%d%d" % (i1,i2) for i1,i2 in zip(violation0,violation1)]
        
        df = pd.DataFrame({'label': label, 'exp_avg': exp_avg,'exp_sigma':exp_sigma,\
                           'calc_avg':calc_avg0,'calc_avg_rew':calc_avg1,"violation":violation})

        with open(outfile, 'w') as fh:
            fh.write('# %s \n' % " ".join(list(df.columns)))
        df['label'] = df['label'].map(lambda x: '%-15s' % x)
        df.to_csv(outfile,index=False,sep="\t",header=False,float_format="%8.3e",mode="a")
        with open(outfile, 'a') as fh:
            fh.write("# CHI2:       %8.4f %8.4f \n"  % (chi2_0,chi2_1))
            fh.write("# RMSD:       %8.4f %8.4f \n"  % (rmsd_0,rmsd_1))
            fh.write("# Violations: %8d %8d \n"  % (nviol_0,nviol_1))
            fh.write("# ndata:      %8d %8d \n"  % (exp.shape[0],calc.shape[0]))

    stats = [chi2_0,rmsd_0,nviol_0,chi2_1,rmsd_1,nviol_1]
    return stats,log


# standardize dataset. Modify array in place
def standardize(exp,calc,sample_weights,normalize="zscore"):

    log = ""
    #normalize="none"
    if(normalize=="zscore"):
        v1 = np.sum(calc*sample_weights[:,np.newaxis],axis=0)
        calc_var = np.average((v1-calc)**2, weights=sample_weights,axis=0)
        #v2 = np.sqrt(np.average(np.array([calc_var,exp[:,1]]),axis=0)) # std
        v2 = np.average(np.array([np.sqrt(calc_var),exp[:,1]]),axis=0)
        exp[:,0] -= v1
        exp[:,0] /= v2
        calc -= v1
        calc /= v2
        exp[:,1] /= v2
        
        log += "# Z-score normalization \n"
        
    elif(normalize=="minmax"):
        mmin = calc.min(axis=0)
        mmax = calc.max(axis=0)
        delta = mmax-mmin
        #exp[:,0] = (exp[:,0]-mmin)/delta
        #calc = (calc-mmin)/delta
        exp[:,0] -= mmin
        exp[:,0] /= delta
        calc -= mmin
        calc /= delta
        exp[:,1] /= delta
        log += "# MinMax normalization \n"
        #print(np.min(np.abs(delta)))

    # do not use this one
    elif(normalize=="bme"):
        #if(np.abs(exp_data[l][0])>1.0E-05):
        print("WHAT???")
        ff = exp[:,0]
        exp[:,1] /= ff
        calc /= ff
        exp[:,0] /= ff
        v1 = 0
        v2 = ff
    return log,v1,v2

