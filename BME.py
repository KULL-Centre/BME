from __future__ import print_function

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize #version >0.13.0 for dogleg optimizer
from scipy.stats import linregress
from scipy.special import logsumexp
from sklearn.linear_model import LinearRegression
#import warnings
import BME_tools as bt
#import csv

known_methods = ["BME","BER","CHI2_L2","CHI1_L1"]

def progress(count, total, suffix=''):
    total -= 1
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()
    
# reweight class. 
class Reweight:

    # initialize
    def __init__(self,name,w0=[]):

        self.name = name

        if(len(w0)!=0):
            self.w0 = w0/np.sum(w0)
        else:
            self.w0= []
        
        self.w_opt = []
        self.lambdas = []
        
        self.log = open("%s.log" % name, "w")
        
        self.labels = []
        self.experiment =  []
        self.calculated =  []
        self.standardized = False
        
    def read_file(self,exp_file,calc_file,averaging="auto",fit='no',use_samples=[],use_data=[]):

        # read file
        log = ""
        label,exp,calc,log,averaging = bt.parse(exp_file,calc_file,averaging=averaging)
        self.log.write(log)
        
        # remove datapoints if use_samples or use_data is not empty
        label,exp, calc, log = bt.subsample(label,exp,calc,use_samples,use_data)
        self.log.write(log)

        if(len(self.w0)==0):
            self.w0 = np.ones(calc.shape[0])/calc.shape[0]
            self.log.write("Initialized uniform weights %d\n" % calc.shape[0])

        # fit/scale
        calc_avg,log = bt.fit_and_scale(exp,calc,self.w0,fit=fit)
        self.log.write(log)

        # do sanity checks
        log  = bt.check_data(label,exp,calc,self.w0)
        self.log.write(log)
        

        return label,exp,calc

    def load(self,exp_file,calc_file,averaging="auto",fit='no',use_samples=[],use_data=[],weight=1):
        
        label,exp, calc = self.read_file(exp_file,calc_file,averaging=averaging,fit=fit,use_samples=use_samples,use_data=use_data)

        if(len(self.experiment)==0):
            self.experiment = exp
            self.calculated = calc
            self.labels = label
            self.weights = np.ones(exp.shape[0])*weight
        else:
            self.experiment = np.vstack([self.experiment,exp])
            self.calculated = np.hstack([self.calculated,calc])
            self.labels = np.hstack([self.labels,label])
            self.weights = np.hstack([self.weights,np.ones(exp.shape[0])*weight])
        
        # note to self: implement weight

    # add data from external array
    def load_array(self,label,exp,calc,weight=1):
        
        if(len(self.experiment)==0):
            self.experiment = exp
            self.calculated = calc
            self.labels = label
            if(len(self.w0)==0):
                self.w0 = np.ones(calc.shape[0])/calc.shape[0]
            self.weights = np.ones(exp.shape[0])*weight
        else:
            self.experiment = np.vstack([self.experiment,exp])
            self.calculated = np.hstack([self.calculated,calc])
            self.labels = np.hstack([self.labels,label])
            self.weights = np.hstack([self.weights,np.ones(exp.shape[0])*weight])
            
    def predict(self,exp_file,calc_file,outfile=None,averaging="auto",fit='no',use_samples=[],use_data=[]):

        label,exp,calc,log,averaging = bt.parse(exp_file,calc_file,averaging=averaging)
        
        self.log.write(log)
        
        # remove datapoints if use_samples or use_data is not empty
        label,exp, calc, log = bt.subsample(label,exp,calc,use_samples,use_data)
        self.log.write(log)

        # do sanity checks
        stats,log  = bt.calc_stats(label,exp,calc,self.w0,self.w_opt,averaging=averaging,fit=fit,outfile=outfile)
        self.log.write(log)
        
        return stats
    
    def predict_array(self,label,exp,calc,outfile=None,averaging="linear",fit="no"):
        stats, log = bt.calc_stats(label,exp,calc,self.w0,self.w_opt,averaging=averaging,outfile=outfile,fit=fit)
        return stats


        # do sanity checks
        #log  = bt.check_data(label,exp,calc,self.w_opt)
        #self.log.write(log)

    def get_lambdas(self):
        return self.lambdas
    
    def get_iterations(self):
        return self.niter

    def get_nsamples(self):
        return self.calculated.shape[0]

    def get_ndata(self):
        return self.experiment.shape[0]

    def get_labels(self):
        return self.labels
    
    def get_experiment(self):
        return np.copy(self.experiment)
    
    def get_calculated(self):
        return np.copy(self.calculated)

    def get_name(self):
        return self.name
    
    def get_weights(self):
        return np.copy(self.w_opt)
    
    def get_w0(self):
        return np.copy(self.w0)

    def set_lambdas(self,lambda0):
        if(len(self.lambdas)==0):
            self.lambdas = lambda0
        else:
            print("# Overriding lambdas is not possible")
            sys.exit(1)


    # optimize
    def fit(self,theta,lambdas_init=True,method="BME"):

        if(self.standardized==False):
            log,v1,v2 = bt.standardize(self.experiment,self.calculated,self.w0,normalize="zscore")
            self.standardized=True

        assert method in known_methods, "method %s not in known methods:" % (method,known_methods)
        
        def maxent(lambdas):
            # weights
            arg = -np.sum(lambdas[np.newaxis,:]*self.calculated,axis=1)-tmax+np.log(self.w0)
            
            logz = logsumexp(arg)
            ww = np.exp(arg-logz)
            
            avg = np.sum(ww[:,np.newaxis]*self.calculated, axis=0)
            
            # gaussian integral
            eps2 = 0.5*np.sum((lambdas*lambdas)*theta_sigma2) 
            
            # experimental value 
            sum1 = np.dot(lambdas,self.experiment[:,0])
            #fun = sum1 + eps2 + np.log(zz)
            fun = sum1 + eps2 + logz
            
            # gradient
            #jac = self.experiment["avg"].values + lambdas*err - avg
            jac = self.experiment[:,0] + lambdas*theta_sigma2 - avg
            

            # divide by theta to avoid numerical problems
            return  fun/theta,jac/theta

    
        def maxent_hess(lambdas):
            arg = -np.sum(lambdas[np.newaxis,:]*self.calculated,axis=1) -tmax
            #arg -= tmax
            
            #########
            ww = (self.w0*np.exp(arg))
            zz = np.sum(ww)
            assert np.isfinite(zz), "# Error. sum of weights is infinite. Use higher theta"
            ww /= zz

            q_w = np.dot(ww,self.calculated)
            hess = np.einsum('k, ki, kj->ij',ww,self.calculated,self.calculated) - np.outer(q_w,q_w) + np.diag(theta_sigma2)

            return  hess/theta

        def func_ber_gauss(w):

            bcalc = np.sum(w[:,np.newaxis]*self.calculated,axis=0)
            diff = bcalc-self.experiment[:,0]
            #print(diff)
            #ii = np.where(((diff<0) & (self.experiment[:,2]<0)) | ((diff>0) & (self.experiment[:,2]>0)) )[0#]
 #           ff = [1 if (self.experiment[j,2]==0 or j in ii) else 0 for j in range(self.experiment.shape[0])]
            #diff *= ff
            
            chi2_half =  0.5*np.sum(((diff/self.experiment[:,1])**2))

            
            idxs = np.where(w>1.0E-50)
            log_div = np.zeros(w.shape[0])
            log_div[idxs] = np.log(w[idxs]/self.w0[idxs])
            srel = theta*np.sum(w*log_div)

            jac = np.sum(diff*self.calculated,axis=1) + theta*(1.+log_div)
            return chi2_half+srel#, jac
    

        
        def func_chi2_L2(w):

            bcalc = np.sum(w[:,np.newaxis]*self.calculated,axis=0)
            diff = (bcalc-self.experiment[:,0])/self.experiment[:,1]
            
            ii = np.where(((diff<0) & (self.experiment[:,2]<0)) | ((diff>0) & (self.experiment[:,2]>0)) )[0]
            ff = [1 if (self.experiment[j,2]==0 or j in ii) else 0 for j in range(self.experiment.shape[0])]

            diff *= ff
            
            chi2_half =  0.5*np.sum(diff**2)

            jac = np.sum(diff*self.calculated,axis=1)
            #idxs = np.where(w>1.0E-50)
                                                                                                                                              #srel = theta*np.sum(w[idxs]*np.log(w[idxs]/self.w0[idxs]))
                                                                                                                                              #jac = 
            return chi2_half
        
        def func_chi2_L1(w):

            bcalc = np.sum(w[:,np.newaxis]*self.calculated,axis=0)
            diff = (bcalc-self.experiment[:,0])/self.experiment[:,1]
            
            ii = np.where(((diff<0) & (self.experiment[:,2]<0)) | ((diff>0) & (self.experiment[:,2]>0)) )[0]
            ff = [1 if (self.experiment[j,2]==0 or j in ii) else 0 for j in range(self.experiment.shape[0])]

            diff *= ff
            
            chi2_half =  0.5*np.sum(diff**2)

            jac = np.sum(diff*self.calculated,axis=1)
            #idxs = np.where(w>1.0E-50)
                                                                                                                                              #srel = theta*np.sum(w[idxs]*np.log(w[idxs]/self.w0[idxs]))
                                                                                                                                              #jac = 
            return chi2_half,jac

        if(lambdas_init==True):
            lambdas=np.zeros(self.experiment.shape[0],dtype=np.longdouble)
            self.log.write("Lagrange multipliers initialized from zero\n")
        else:
            assert(len(self.lambdas)==self.experiment.shape[0])
            lambdas = np.copy(self.lambdas)
            #np.array(lambdas_init)
            self.log.write("Warm start\n")
        #print(lambdas)
            
        bounds = []
        for j in range(self.experiment.shape[0]):
            if(self.experiment[j,2]==0):
                bounds.append([None,None])
            elif(self.experiment[j,2]==-1):
                bounds.append([None,0.0])
            else:
                bounds.append([0.0,None])

        if(method=="BME"):
            opt={'maxiter':50000,'disp':False}
            
            tmax = np.log((sys.float_info.max)/5.)

            theta_sigma2 = theta*self.weights*self.experiment[:,1]**2

            chi2_before  = bt.calc_chi(self.experiment,self.calculated,self.w0)
            self.log.write("Optimizing %d data and %d samples. Theta=%f \n" % (self.experiment.shape[0],self.calculated.shape[0],theta))
            self.log.write("CHI2 before optimization: %8.4f \n" % (chi2_before))
            self.log.flush()
            mini_method = "L-BFGS-B"
            start_time = time.time()
            #if(all(self.experiment[:,2]==0)):
            #    mini_method="trust-constr"
            #    result = minimize(maxent,lambdas,\
            #                               options=opt,method=mini_method,\
            #                               jac=True,hess=maxent_hess)
            
            result = minimize(maxent,lambdas,\
                              options=opt,method=mini_method,\
                              jac=True,bounds=bounds)
            self.log.write("Execution time: %.2f seconds\n" % (time.time() - start_time))
            
            if(result.success):
                self.log.write("Minimization using %s successful (iterations:%d)\n" % (mini_method,result.nit))
                arg = -np.sum(result.x[np.newaxis,:]*self.calculated,axis=1) -tmax
                w_opt = self.w0*np.exp(arg)
                w_opt /= np.sum(w_opt)
                self.lambdas = np.copy(result.x)
                self.w_opt = np.copy(w_opt)
                self.niter = result.nit
                chi2_after  = bt.calc_chi(self.experiment,self.calculated,w_opt)
                phi = np.exp(-bt.srel(self.w0,w_opt))
                
                self.log.write("CHI2 after optimization: %8.4f \n" % (chi2_after))
                self.log.write("Fraction of effective frames: %8.4f \n" % (phi))
                self.log.flush()
                return chi2_before,chi2_after,phi
            
            else:
                self.log.write("Minimization using %s failed\n" % (mini_method))
                self.log.write("Message: %s\n" % (result.message))
                self.niter = -1
                self.log.flush()
                return np.NaN, np.NaN, np.NaN
            
            

        # please check 
        if(method=="BER"):
            
            opt={'maxiter':2000,'disp': True,'ftol':1.0e-20}
            cons = {'type': 'eq', 'fun':lambda x: np.sum(x)-1.0}
            bounds = [(0.,None)]*len(self.w0)
            mini_method = "SLSQP"
            chi2_before  = bt.calc_chi(self.experiment,self.calculated,self.w0)
            self.log.write("CHI2 before optimization: %8.4f \n" % (chi2_before))
            self.log.flush()
                                                                                                                                            
            w0 = np.copy(self.w0)
            start_time = time.time()
            #print(func_ber_gauss(w0))
            #result = minimize(func_ber_gauss,w0,constraints=cons,options=opt,method=mini_method,bounds=bounds,jac=True)
            result = minimize(func_ber_gauss,w0,constraints=cons,options=opt,method=mini_method,bounds=bounds,jac=False)
            self.log.write("Execution time: %.2f seconds\n" % (time.time() - start_time))
            if(result.success):
                self.log.write("Minimization using %s successful (iterations:%d)\n" % (mini_method,result.nit))
                w_opt = np.copy(result.x)
                self.w_opt = w_opt
                chi2_after  = bt.calc_chi(self.experiment,self.calculated,w_opt)
                phi = np.exp(-bt.srel(self.w0,w_opt))
                self.log.write("CHI2 after optimization: %8.4f \n" % (chi2_after))
                self.log.write("Fraction of effective frames: %8.4f \n" % (phi))
                self.log.flush()
                return chi2_before,chi2_after,phi
            
            else:
                self.log.write("Minimization using %s failed\n" % (mini_method))
                self.log.write("Message: %s\n" % (result.message))
                self.log.flush()
                return np.NaN, np.NaN, np.NaN
   
        # please check 
        if(method=="CHI2_L2"):
            
            opt={'maxiter':2000,'disp': True,'ftol':1.0e-20}
            cons = {'type': 'eq', 'fun':lambda x: np.sum(x)-1.0}
            bounds = [(0.,None)]*len(self.w0)
            meth = "SLSQP"
            chi2_before  = bt.calc_chi(self.experiment,self.calculated,self.w0)
            self.log.write("CHI2 before optimization: %8.4f \n" % (chi2_before))
            start_time = time.time()
            result = minimize(func_chi2_L2,self.w0,constraints=cons,options=opt,method=meth,jac=True,bounds=bounds)

            w_opt = np.copy(result.x)
            self.w_opt = w_opt
            chi2_after  = bt.calc_chi(self.experiment,self.calculated,w_opt)
            phi = np.exp(-bt.srel(self.w0,w_opt))
            self.log.write("Execution time: %.2f seconds\n" % (time.time() - start_time))
            self.log.write("CHI2 after optimization: %8.4f \n" % (chi2_after))

        # please check 
        if(method=="CHI2_L1"):
            
            opt={'maxiter':2000,'disp': True,'ftol':1.0e-20}
            cons = {'type': 'eq', 'fun':lambda x: np.sum(x)-1.0}
            bounds = [(0.,None)]*len(self.w0)
            meth = "SLSQP"
            chi2_before  = bt.calc_chi(self.experiment,self.calculated,self.w0)
            self.log.write("CHI2 before optimization: %8.4f \n" % (chi2_before))
            start_time = time.time()
            result = minimize(func_chi2_L2,self.w0,constraints=cons,options=opt,method=meth,jac=True,bounds=bounds)

            w_opt = np.copy(result.x)
            self.w_opt = w_opt
            chi2_after  = bt.calc_chi(self.experiment,self.calculated,w_opt)
            phi = np.exp(-bt.srel(self.w0,w_opt))
            self.log.write("Execution time: %.2f seconds\n" % (time.time() - start_time))
            self.log.write("CHI2 after optimization: %8.4f \n" % (chi2_after))



    def theta_scan(self,thetas=[],train_fraction_data=0.75,nfold=5,train_fraction_samples=0.8):

        np.random.seed(42)
        if(len(thetas)==0): thetas = np.geomspace(0.1,10000,10)
        print("Performing %d-fold cross validation for %d theta values" % (nfold,len(thetas)))
        nsamples = self.get_nsamples()
        ndata =   self.get_ndata()
        
        train_samples =  int(nsamples*train_fraction_samples)
        train_data =  int(ndata*train_fraction_data)

        results = np.zeros((len(thetas),nfold,3))
        for i in range(nfold):
            progress(i,nfold)
            if(i==nfold-1):
                print("\n")
            shuffle_samples = np.arange(nsamples)
            shuffle_data = np.arange(ndata)
            np.random.shuffle(shuffle_samples)
            np.random.shuffle(shuffle_data)
            
            train_idx_data = shuffle_data[:train_data]
            train_idx_samples = shuffle_samples[:train_samples]
            test_idx_data = shuffle_data[train_data:]
            test_idx_samples = shuffle_samples[:train_samples]  # test samples are the same as train!
        

            labels_train = [self.labels[k] for k in train_idx_data]
            labels_test = [self.labels[k] for k in test_idx_data]
            
            exp = self.get_experiment()
            calc = self.get_calculated()
            exp_train = exp[train_idx_data,:]
            calc_train = calc[:,train_idx_data]
        
            exp_test = exp[test_idx_data,:]
            calc_test = calc[:,test_idx_data]
        
            r1 = Reweight("crossval_%s_%d" % (self.name,i),w0=np.copy(self.w0[train_idx_samples]))        
            r1.load_array(labels_train,np.copy(exp_train),np.copy(calc_train[train_idx_samples,:]))
            l_init = True
        
            for j,t in enumerate(thetas):
                c1,c2,phi = r1.fit(t,lambdas_init=l_init)
                l_init = False
                fr = "crossval_%s_t%.2f_f%d" % (self.name,t,i)
                train_stats = r1.predict_array(labels_train,exp_train,calc_train[train_idx_samples,:],\
                                           outfile="%s_train" % (fr))
                test_stats = r1.predict_array(labels_test,exp_test,calc_test[test_idx_samples,:],
                                          outfile="%s_test" % (fr))
                #print("####",i,j)
                #outfiles.append(fr)
                results[j,i,0] = train_stats[3]/train_stats[0]
                results[j,i,1] = test_stats[3]/test_stats[0]
                results[j,i,2] = phi

        fig,ax = plt.subplots(1,1,figsize=(12,8))
        for k in range(nfold):
            plt.scatter(thetas,results[:,k,0],c='k',s=4)
            plt.scatter(thetas,results[:,k,1],c='r',s=4)
            plt.scatter(thetas,results[:,k,2],c='b',s=4)

        avg_test_error = np.nanmean(results[:,:,1],axis=1)
        avg_train_error = np.nanmean(results[:,:,0],axis=1)
        avg_phi = np.nanmean(results[:,:,2],axis=1)
        argmin = np.argmin(avg_test_error)
        
        print("Optimal theta: %.2f" % thetas[argmin])
        print("Validation error reduction %.3f" % avg_test_error[argmin])
        print("Training error reduction %.3f" % avg_train_error[argmin])
        print("Fraction of effective frames  %.3f" % avg_phi[argmin])

        plt.plot(thetas,avg_train_error,"-*",c='k',label="Training error")
        plt.plot(thetas,avg_test_error,"-o",c='r',label="Test error")
        plt.plot(thetas,avg_phi,"-o",c='b',label="Phi")
        plt.legend()
        if(np.max(results[:,:,1])<1.):
            ax.set_ylim(0,1.1)
        else:
            ax.set_ylim(0,np.min([2,1.1*np.max(results[:,:,1])]))
        plt.scatter(thetas[argmin],avg_train_error[argmin],c='r')
        ax.axhline(1,ls="--",color='0.4')
        ax.set_xscale('log')
        ax.set_xlabel("Theta")
        plt.savefig("crossval_%s.png" % self.name)

        return thetas[argmin]
        
        
    def ibme(self,theta,ftol=0.01,iterations=50,lr_weights=True,offset=True):
        
        current_weights = self.get_w0()
        w0 = self.get_w0()
        name = self.get_name()
        labels = self.get_labels()
        exp = self.get_experiment()
        calc = self.get_calculated()

        self.ibme_weights = []
        self.ibme_stats = []
        
        #print("# iterative BME")
        if(lr_weights==True):
            inv_var = 1./exp[:,1]**2
        else:
            inv_var = np.ones(len(exp))

        log = ""
        rr_old = np.NaN
        #self.log.write("acascacasacas\n")
        for it in range(iterations):
            
            calc_avg = np.sum(calc*current_weights[:,np.newaxis],axis=0)

            model = LinearRegression(fit_intercept=offset)
            model.fit(calc_avg.reshape(-1,1),exp[:,0],inv_var)

            #alpha, beta, r_value, p_value, std_err = linregress(calc_avg,exp[:,0])
            alpha = model.coef_[0] #Scale factor
            beta = model.intercept_
            calc = alpha*calc+beta
            
            r1 = Reweight("%s_ibme_%d" % (name,it),w0=np.copy(w0))
            r1.load_array(labels,np.copy(exp),np.copy(calc))
            rr= r1.fit(theta=theta)
            if(it==0): chi2_0 = rr[0]
            if(it==0): calc_0 = np.copy(calc)
                
            current_weights = np.copy(r1.get_weights())

            diff = abs(rr_old-rr[1])
            line = "Iteration:%3d scale: %7.4f offset: %7.4f chi2: %7.4f diff: %7.4e\n" % (it,alpha,beta,rr[1],diff)
            rr_old = rr[1]
            #print(line,end="")
            log += line

            self.ibme_weights.append(current_weights)
            self.ibme_stats.append(rr)
            
            if(diff<ftol):
                line = "Iterative procedure converged below tolerance %.2e after %d iterations\n" % (diff,it)
                print(line,end="")
                log += line
                break
            
        self.log.write(log+ "\n")
        #self.log.close()
        self.log.flush()
        n1 = "%s_%d.calc.dat" % (self.name,it)
        n2 = "%s_%d.weights.dat" % (self.name,it)
        
        df = pd.DataFrame(calc)
        df.to_csv(n1,sep=" ",header=False,float_format="%8.4e")

        df = pd.DataFrame(current_weights)
        df.to_csv(n2,sep=" ",header=False,float_format="%8.4e")
        
        #print("Done. Initial chi2: %8.4f Final chi2:%8.4f" % (chi2_0,rr[1]))
        #print("Done. Writing output files %s %s" % (n1,n2))
        phi = np.exp(-bt.srel(w0,current_weights))
        self.w_opt = current_weights
        return chi2_0,rr[1],phi,calc_0,calc

    def get_ibme_weights(self):

        try:
            return self.ibme_weights
        except:
            print("# iBME weights not available. Call iBME first")
            sys.exit(1)

    def get_ibme_stats(self):

        try:
            return self.ibme_stats
        except:
            print("# iBME stats not available. Call iBME first")
            sys.exit(1)

        
    def iterative_theta_scan(self,thetas=[],ftol=0.01,iterations=50,lr_weights=True,offset=True,
                            train_fraction_data=0.75,nfold=5,train_fraction_samples=0.8):

        np.random.seed(42)
        if(len(thetas)==0): thetas = np.geomspace(0.1,10000,10)
        print("Performing %d-fold cross validation for %d theta values" % (nfold,len(thetas)))
        nsamples = self.get_nsamples()
        ndata =   self.get_ndata()
        
        train_samples =  int(nsamples*train_fraction_samples)
        train_data =  int(ndata*train_fraction_data)

        results = np.zeros((len(thetas),nfold,3))
        for i in range(nfold):
            progress(i,nfold)
            if(i==nfold-1):
                print("\n")
            shuffle_samples = np.arange(nsamples)
            shuffle_data = np.arange(ndata)
            np.random.shuffle(shuffle_samples)
            np.random.shuffle(shuffle_data)
            
            train_idx_data = shuffle_data[:train_data]
            train_idx_samples = shuffle_samples[:train_samples]
            test_idx_data = shuffle_data[train_data:]
            test_idx_samples = shuffle_samples[:train_samples]  # test samples are the same as train!
        

            #labels_train = [self.labels[k] for k in train_idx_data]
            #labels_test = [self.labels[k] for k in test_idx_data]
            
            exp = self.get_experiment()
            calc = self.get_calculated()
            exp_train = exp[train_idx_data,:]
            calc_train = calc[:,train_idx_data]
        
            exp_test = exp[test_idx_data,:]
            calc_test = calc[:,test_idx_data]

            r1 = Reweight("crossval_%s_%d" % (self.name,i),w0=np.copy(self.w0[train_idx_samples]))       
            r1.load_array(train_idx_data,np.copy(exp_train),np.copy(calc_train[train_idx_samples,:]))
        
            for j,t in enumerate(thetas):
                c1,c2,phi,calc_0,calc_1 = r1.ibme(t,ftol=ftol,iterations=iterations,
                                    lr_weights=lr_weights,offset=offset)
                
                #calc = a*calc+b
                #calc_train = calc[:,train_idx_data]
                #calc_test = calc[:,test_idx_data]

                l_init = False
                fr = "crossval_%s_t%.2f_f%d" % (self.name,t,i)
                chi_train_0 = bt.calc_chi(exp[train_idx_data],calc_0, r1.w0)
                chi_train_1 = bt.calc_chi(exp[train_idx_data],calc_1, r1.w_opt)
                #Because the alpha and beta parameters for the scaled observables are not kept in ibme,
                #we fit the final calc arrays to obtain them.                
                model = LinearRegression(fit_intercept=offset)
                model.fit(calc_train[train_idx_samples,0].reshape(-1,1),calc_0[:,0])
                # Initial values
                alpha = model.coef_[0] #Scale factor
                beta = model.intercept_
                calc_test_0 = alpha*calc_test+beta
                # Optimized values
                model.fit(calc_train[train_idx_samples,0].reshape(-1,1),calc_1[:,0])
                alpha = model.coef_[0] #Scale factor
                beta = model.intercept_
                calc_test_1 = alpha*calc_test+beta

                chi_test_0 = bt.calc_chi(exp[test_idx_data],calc_test_0[test_idx_samples,:], r1.w0)
                chi_test_1 = bt.calc_chi(exp[test_idx_data],calc_test_1[test_idx_samples,:], r1.w_opt)
                #print("####",i,j)
                #outfiles.append(fr)
                results[j,i,0] = chi_train_1/chi_train_0
                results[j,i,1] = chi_test_1/chi_test_0
                results[j,i,2] = phi

        fig,ax = plt.subplots(1,1,figsize=(12,8))
        for k in range(nfold):
            plt.scatter(thetas,results[:,k,0],c='k',s=4)
            plt.scatter(thetas,results[:,k,1],c='r',s=4)
            plt.scatter(thetas,results[:,k,2],c='b',s=4)

        avg_test_error = np.nanmean(results[:,:,1],axis=1)
        avg_train_error = np.nanmean(results[:,:,0],axis=1)
        avg_phi = np.nanmean(results[:,:,2],axis=1)
        argmin = np.argmin(avg_test_error)
        
        print("Optimal theta: %.2f" % thetas[argmin])
        print("Validation error reduction %.3f" % avg_test_error[argmin])
        print("Training error reduction %.3f" % avg_train_error[argmin])
        print("Fraction of effective frames  %.3f" % avg_phi[argmin])

        plt.plot(thetas,avg_train_error,"-*",c='k',label="Training error")
        plt.plot(thetas,avg_test_error,"-o",c='r',label="Test error")
        plt.plot(thetas,avg_phi,"-o",c='b',label="Phi")
        plt.legend()
        if(np.max(results[:,:,1])<1.):
            ax.set_ylim(0,1.1)
        else:
            ax.set_ylim(0,np.min([2,1.1*np.max(results[:,:,1])]))
        plt.scatter(thetas[argmin],avg_train_error[argmin],c='r')
        ax.axhline(1,ls="--",color='0.4')
        ax.set_xscale('log')
        ax.set_xlabel("Theta")
        plt.savefig("crossval_%s.png" % self.name)

        return thetas[argmin]
        

