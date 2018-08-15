#   This is a small script for reweighting molecular simulations using
#   the Bayesian/MaxEnt (BME) approach
#   Copyright (C) 2018 Sandro Bottaro (sandro . bottaro @ bio . ku . dk)
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License V3 as published by
#   the Free Software Foundation, 
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy import optimize
import argparse
import sys
import os.path


class Reweight:

    def __init__(self,exp_files,sim_files):

        self.exp_data = []
        self.labels = []
        self.bounds = []
        self.norm_factor = []
        self.result = None
        
        print "###### INITIALIZATION ########"
        
        # these are known data types and error priors
        self.exp_types = ["NOE","JCOUPLINGS","CS","SAXS","DIST","RDC"]
        self.prior_types = ["GAUSS"]

        assert len(exp_files)==len(sim_files), "# Error. Number of experimental (%d) and simulation (%d) files must be equal" % (len(exp_files),len(sim_files))
        for k in range(len(exp_files)):

            # read experimental data 
            labels, bounds, exp_data, data_type,noe_power =  self.read_exp(exp_files[k])

            # read simulation data
            sim_data = self.read_sim(sim_files[k])
            if(data_type=="NOE"):
                sim_data = np.power(sim_data,-noe_power)
            
            assert len(labels) == sim_data.shape[1], "# Number of rows in exp (%d) not equal to number of columns in calc (%d)" % (len(labels),sim_data.shape[1])

            # now do some sanity checks on data 
            mins = np.min(sim_data,axis=0)
            maxs = np.max(sim_data,axis=0)
            norm_factor = []
            for l in range(len(exp_data)):

                if(bounds[l][0] == 0.0  and mins[l]> exp_data[l][0]):
                    print "# Warning: expt boundary %s=%-10.4f is smaller than minimum value in simulation %-10.4f"\
                        % (labels[l],exp_data[l][0],mins[l])
                if(bounds[l][1] == 0.0  and maxs[l] < exp_data[l][0]):
                    print "# Warning: expt boundary %s=%-10.4f is larger than maximum value in simulation %-10.4f"\
                        % (labels[l],exp_data[l][0],mins[l])
                
                if(exp_data[l][0] > maxs[l]):
                    print "# Warning: expt average %s=%-10.4f is larger than maximum value in simulation %-10.4f"\
                    % (labels[l],exp_data[l][0],maxs[l])
                if(exp_data[l][0] < mins[l]):
                    print "# Warning: expt average %s=%-10.4f is smaller than minimum value in simulation %-10.4f"\
                        % (labels[l],exp_data[l][0],mins[l])

                # if exp data is non-zero, normalize
                ff = 1.0            
                if(np.abs(exp_data[l][0])>1.0E-07):
                    ff = exp_data[l][0]
                    exp_data[l][1] /= ff**2
                    sim_data[:,l] /= ff
                    exp_data[l][0] /= ff
                norm_factor.append(ff)
                
            self.exp_data.extend(exp_data)
            if(k==0):
                self.sim_data = np.array(sim_data)
            else:
                self.sim_data = np.concatenate((self.sim_data,np.array(sim_data)),axis=1)
            self.labels.extend(labels)
            self.bounds.extend(bounds)
            self.norm_factor.extend(norm_factor)
            #self.data_source.extend([data_type]*len(labels))
            
        self.exp_data = np.array(self.exp_data)
        
        print "# Exp data shape ",self.exp_data.shape
        print "# Simulation data shape ",self.sim_data.shape

            # normalize by experimental average if 
        print "###### Initialization OK ########"
        self.w0 = np.ones(self.sim_data.shape[0])/self.sim_data.shape[0]

    # read experimental data
    def read_exp(self,filename):

        fh = open(filename)
        first = fh.readline().split()
        
        # do some checks on first line
        assert first[0] == "#", "Error. First line of exp file %s must be in the format \# DATA=XXX PRIOR=XXX" % filename

        # find data type and data prior
        data_type = (first[1].split("=")[1]).strip()
        prior_type = (first[2].split("=")[1]).strip()
        assert data_type in self.exp_types , "Error. DATA must be one of the following: %s " % (self.exp_types)
        assert prior_type in self.prior_types , "Error. PRIOR must be one of the following: %s " % (self.prior_types)
        
        # noe power is 6 by default, but it can be changed in file 
        noe_power=6.0
        if(len(first)==4 and first[3]=="POWER"):
            noe_power = float((first[3].split("=")[1]).strip())
            assert(noe_power>0.)
            
        ln = 0
        labels = []
        bounds = []
        exp_data = []
        
        for line in fh:
            # skip commented out data
            if("#" in line): continue

            ln += 1
            lsplit = line.split()
            if(len(lsplit) != 3 and len(lsplit) != 4):
                print "# ERROR: experimental line in %s not valid" % filename
                print line,
                sys.exit(1)
                
            labels.append( "%s"  % (lsplit[0]))
                
            # if lenght is 4, third and fourth columns are  r_avg and sigma
            avg0 = float(lsplit[1])
            sigma0 = float(lsplit[2])

            if(len(lsplit)==3):
                bounds.append([None,None])

            # if lenght is 4
            elif(len(lsplit)==4):

                # upper/lower bound
                if(lsplit[3] == "UPPER"):
                    # upper/lower is inverted because of 1/r^n dependence of NOE
                    if(data_type=="NOE"):
                        bounds.append([None, 0.0])
                    else:
                        bounds.append([0.0,None])
                elif(lsplit[3] == "LOWER"):
                    if(data_type=="NOE"):
                        bounds.append([0.0,None])
                    else:
                        bounds.append([None, 0.0])


            if(data_type=="NOE"):
                avg = np.power(avg0,-noe_power)
                sigma = (noe_power*avg*sigma0/(avg0))**2
                exp_data.append([avg,sigma])
            else:
                exp_data.append([avg0,sigma0*sigma0])
                

                
        print "# Read %d %s experimental datapoints" % (ln,data_type)
        assert(len(labels) == len(bounds))
        assert(len(labels) == len(exp_data))
        
        return labels, bounds, exp_data, data_type, noe_power

    def srel(self,w):
        idxs = np.where(w>1.0E-50)
        return np.sum(w[idxs]*np.log(w[idxs]/self.w0[idxs]))

    def read_sim(self,filename):
        
        data = np.array([[float(x) for x in line.split()[1:]] for line in open(filename) if("#" not in line)])
        print "# Read %d simulated datapoints from %d frames" % (data.shape[1],data.shape[0])
        return data


    def set_w0(self,filename,kbt=None):
        w0 = np.array([float(line.split()[0]) for line in open(filename) if("#" not in line)])
        
        if(kbt!=None):
            print "# Assuming weights given as minus free energies. w=exp(bias/kbt) kbt=%8.4f "  % kbt
            w0 = np.exp(w0/kbt)
        print "# Set non-uniform initial weights from file. Sum=", np.sum(w0), w0.shape
        print "# ATT!! call this function BEFORE optimization!!"
        self.w0 = (w0/np.sum(w0))
        #self.w0 = w0


    def chi_squared(self,ww):

        bcalc = np.sum(ww[:,np.newaxis]*self.sim_data,axis=0)
        diff = bcalc-self.exp_data[:,0]
        idxs = [0.0 if(( diff[i] >= 0 and self.bounds[i][1] == 0) or (diff[i] <= 0 and self.bounds[i][0] == 0)) else 1.0 for i in range(len(self.bounds))]
        diff *= idxs                
        return np.sum((diff*diff)/self.exp_data[:,1])

                
    def optimize(self,theta,method="MAXENT"):

        def func_maxent_gauss(lambdas):
            
            # weights
            ww  = self.w0*np.exp(-np.sum(lambdas[np.newaxis,:]*self.sim_data,axis=1))
            # normalization 
            zz = np.sum(ww)
            ww /= zz
            # new averages
            avg = np.sum((ww[:,np.newaxis]*self.sim_data), axis=0)
            
            # errors are rescaled by factor theta
            #err = (self.theta)*(self.exp_data[:,1])
            err = (self.theta)*(self.exp_data[:,1])
            
            # gaussian integral
            eps2 = 0.5*np.sum((lambdas*lambdas)*err)
            # experimental value 
            sum1 = np.dot(lambdas,self.exp_data[:,0])
            fun = sum1 + eps2+ np.log(zz)
            # gradient
            jac = self.exp_data[:,0] + lambdas*err - avg

            return  fun,jac

        
        
        def func_ber_gauss(w):
            
            bcalc = np.sum(w[:,np.newaxis]*self.sim_data,axis=0)
            diff = bcalc-self.exp_data[:,0]
            idxs = [0.0 if(( diff[i] >= 0 and self.bounds[i][1] == 0) or (diff[i] <= 0 and self.bounds[i][0] == 0)) else 1.0 for i in range(len(self.bounds))]
            diff *= idxs
            chi2_half =  0.5*np.sum(((diff**2)/(self.exp_data[:,1])))

            idxs = np.where(w>1.0E-50)
            srel = self.theta*np.sum(w[idxs]*np.log(w[idxs]/self.w0[idxs]))
            return chi2_half+srel

            
        self.theta = theta
        # first, calculate initial chi squared and RMSD
        chi_sq0 = self.chi_squared(self.w0)/len(self.exp_data)
        print "Initial average chi squared %10.4f, srel %10.4f " % (chi_sq0, self.srel(self.w0))
        
        if(method=="MAXENT"):
            #opt={'maxiter':10000,'disp': False,'ftol':1.0e-10}
            opt={'maxiter':50000,'disp': False,'ftol':1.0e-50}
            meth = "L-BFGS-B"
            #meth = "SLSQP"
            lambdas=np.zeros(self.exp_data.shape[0])

            result = optimize.minimize(func_maxent_gauss,lambdas,options=opt,method=meth,jac=True,bounds=self.bounds)
            #self.result = np.copy(result.x)            
            w_opt = self.w0*np.exp(-np.sum(result.x[np.newaxis,:]*self.sim_data,axis=1))
            w_opt /= np.sum(w_opt)
            
        if(method=="BER"):
            opt={'maxiter':2000,'disp': True,'ftol':1.0e-20}
            cons = {'type': 'eq', 'fun':lambda x: np.sum(x)-1.0}
            bounds = [(0.,1.)]*len(self.w0)  
            meth = "SLSQP"
            print "# Bayesian Ensemble Refinement. Useful for testing purposes and "
            print "# When the number of experimental data is larger than the number of samples."
            #w_init = np.ones(self.sim_data.shape[0])/self.sim_data.shape[0]
            
            result = optimize.minimize(func_ber_gauss,self.w0,constraints=cons,options=opt,method=meth,bounds=bounds)
            #self.result = np.copy(result.x)

            w_opt = result.x


        if(result.success):
            print "# Minimization successful"
            chi_sq1 = self.chi_squared(w_opt)/len(self.exp_data)
            print "Final average chi squared   %10.4f, srel %10.4f " % (chi_sq1, self.srel(w_opt))
            self.w_opt = w_opt

            if(method=="MAXENT"):
                fname_lambda = "lambda.dat"
                fh_lambda  = open(fname_lambda,'w')
                stri = "# %-14s %9s\n" % ("Label","Lambda")
                stri += "".join(["   %-15s %9.3e \n" % (self.labels[l],result.x[l]) for l in range(len(self.labels))] )
                fh_lambda.write(stri)
                fh_lambda.close()

            return 0
        
        else:
            print "# ERROR - Minimization failed. Perhaps theta is too small, or there is some bug."
            print "# exit message:", result.message
            return 1
            
        
    def weight_exp(self,exp_files,sim_files,outfile,plot=False):
        
        assert len(exp_files)==len(sim_files), "# Error. Number of experimental (%d) and simulation (%d) files must be equal" % (len(exp_files),len(sim_files))
        
        # write stats to file
        fname_stats = "%s.stats.dat" % (outfile)
        if(os.path.isfile(fname_stats)):
            print "# File %s already exists. " % fname_stats
            ov = raw_input("Overwrite? (y/n) ")
            if(ov!="y"):
                print "# Quit"
                sys.exit(1)

        fh_stats  = open(fname_stats,'w')
        fh_stats.write("# Reweighting. n_data=%d, n_samples=%d \n" % (self.exp_data.shape[0],self.sim_data.shape[0]))
        neff1 = np.exp(np.sum(-self.w_opt*np.log(self.w_opt/self.w0)))
        fh_stats.write("# %-5s %8.4f  \n" % ("Theta",self.theta))        
        fh_stats.write("# %-5s %8.4f  \n" % ("neff",neff1))

        if(plot):
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            try:
                import seaborn as sns
                sns.set_style("white")
                sns.set_context("notebook")
                c1 = sns.color_palette()[0]
                c2 = sns.color_palette()[1]
                c3 = sns.color_palette()[2]
            except:
                print "# not using Seaborn. Nevermind."
                c1 = (0.29803921568627451, 0.44705882352941179, 0.69019607843137254)
                c2 = (0.33333333333333331, 0.6588235294117647, 0.40784313725490196)
                c3 = (0.7686274509803922, 0.30588235294117649, 0.32156862745098042)
            pdf = PdfPages('%s.pdf' % outfile)
            
        # now calculate rmsd, chi and print everything to file
        all_labels = []
        for k in range(len(exp_files)):
            
            labels, bounds, exp_data, data_type,noe_power =  self.read_exp(exp_files[k])
            all_labels.extend(labels)
            
            sim_data = self.read_sim(sim_files[k])
            if(data_type=="NOE"):
                sim_data = np.power(sim_data,-noe_power)

            assert len(labels) == len(sim_data[0]), \
                "# Number of rows in %s not equal to number of columns in %s" % (len(labels),len(labels))
            assert len(self.w_opt) ==  len(sim_data),\
                "# Number of weights %d not equal to number of data from simulation (%d) %" (len(self.w_opt),len(sim_data))

            exp_data = np.array(exp_data)
            sim_data = np.array(sim_data)
            
            # before reweighting
            bcalc0 = np.sum(self.w0[:,np.newaxis]*sim_data,axis=0)
            diff0 = bcalc0-exp_data[:,0]
            #print bcalc0
            #print exp_data[:,0]
            #print diff0
            idxs0 = [0.0 if(( diff0[i] >= 0 and bounds[i][1] == 0) or (diff0[i] <= 0 and bounds[i][0] == 0)) else 1.0 for i in range(len(bounds))]

            diff0 *= idxs0
            rmsd0 = np.sqrt(np.average(diff0*diff0))
            chi_sq0 = np.sum(diff0*diff0/exp_data[:,1])/exp_data.shape[0]
            
            # after reweighting
            bcalc1 = np.sum(self.w_opt[:,np.newaxis]*sim_data,axis=0)
            diff1 = bcalc1-exp_data[:,0]
            idxs1 = [0.0 if(( diff1[i] >= 0 and bounds[i][1] == 0) or (diff1[i] <= 0 and bounds[i][0] == 0)) else 1.0 for i in range(len(bounds))]

            diff1 *= idxs1
            rmsd1 = np.sqrt(np.average(diff1*diff1))
            chi_sq1 = np.sum(diff1*diff1/exp_data[:,1])/exp_data.shape[0]

                
            # write to file.
            fh_stats.write("# %s vs. %s \n" % (exp_files[k].split("/")[-1],sim_files[k].split("/")[-1]))
            fh_stats.write("# %5s %8s %8s \n" % ("","before","after"))
            fh_stats.write("#  %-5s %8.4f %8.4f \n" % ("Srel",self.srel(self.w0),self.srel(self.w_opt)))
            fh_stats.write("#  %-5s %8.4f %8.4f \n" % ("RMSD",rmsd0,rmsd1))
            fh_stats.write("#  %-5s %8.4f %8.4f \n" % ("chi2",chi_sq0,chi_sq1))
            fh_stats.write("#  %-15s %9s %9s %9s %9s \n" % ("Label","avg_exp","sigma_exp","avg_before","avg_after"))
            for l in range(exp_data.shape[0]):
                # convert to distances if is NOE
                if(data_type == "NOE"):
                    exp_avg  = np.power(exp_data[l,0],-1./noe_power)
                    sim_b  = np.power(bcalc0[l],-1./noe_power)
                    sim_a  = np.power(bcalc1[l],-1./noe_power)
                    sigma = np.sqrt(exp_data[l,1])*(exp_avg/(noe_power*exp_data[l,0]))
                    fh_stats.write("   %-15s %9.3e %9.3e %9.3e %9.3e \n" % (labels[l],exp_avg,sigma,sim_b,sim_a))
                else:
                    fh_stats.write("   %-15s %9.3e %9.3e %9.3e %9.3e \n" % (labels[l],exp_data[l,0],np.sqrt(exp_data[l,1]),bcalc0[l],bcalc1[l]))

                if(plot):
                    bins = int(np.sqrt(sim_data.shape[0]*neff1*2))
                    if(bins<10):
                        bins =10
                        
                    h1,e1 = np.histogram(sim_data[:,l],weights=self.w0,bins=bins)
                    h2,e2 = np.histogram(sim_data[:,l],weights=self.w_opt,bins=bins)
                    ee = 0.5*(e1[1:]+e1[:-1])
                    plt.plot(ee,h1,c=c1,label="Before")
                    plt.plot(ee,h2,c=c2,label="After")
                    plt.axvline(bcalc0[l],c=c1,ls='--')
                    plt.axvline(bcalc1[l],c=c2,ls='--')
                    plt.axvline(exp_data[l,0],c='k',ls='--',label="exp")
                    plt.axvspan(exp_data[l,0]-exp_data[l,1], exp_data[l,0]+exp_data[l,1], alpha=0.2, color='0.4')
                    plt.title("%s" % (exp_files[k].split("/")[-1]))
                    plt.xlabel("%s %s (some units) " % (data_type,labels[l]))
                    plt.legend()
                    pdf.savefig()
                    plt.close()
        fh_stats.close()
        
        # write weights to file
        fname_weights = "%s.weights.dat" % (outfile)
        fh_weights  = open(fname_weights,'w')
        fh_weights.write("# Reweighting. n_data=%d, n_samples=%d \n" % (self.exp_data.shape[0],self.sim_data.shape[0]))
        fh_weights.write("# %-5s %8.4f  \n" % ("Theta",self.theta))        
        fh_weights.write("# %-5s %8.4f  \n" % ("neff",neff1))
        fh_weights.write("# %-5s %8.4e %8.4e \n" % ("sum weights",np.sum(self.w0),np.sum(self.w_opt)))
        fh_weights.write("# %8s %8s \n" % ("before","after"))
        for l in range(len(self.w0)):
            fh_weights.write("  %8.4e %8.4e \n" % (self.w0[l],self.w_opt[l]))
        fh_weights.close()
            

        if(plot):
            pdf.close()


    # read data with no experimental values
    
    def weight(self,sim_files,outfile,plot=False):
        

        fname_stats = "%s.stats.dat" % (outfile)
        if(os.path.isfile(fname_stats)):
            print "# File %s already exists. " % fname_stats
            ov = raw_input("Overwrite? (y/n) ")
            if(ov!="y"):
                print "# Quit"
                sys.exit(1)

        fh_stats  = open(fname_stats,'w')
        fh_stats.write("# Reweighting. n_data=%d, n_samples=%d \n" % (self.exp_data.shape[0],self.sim_data.shape[0]))
        neff1 = np.exp(np.sum(-self.w_opt*np.log(self.w_opt)))/len(self.w_opt)
        fh_stats.write("# %-5s %8.4f  \n" % ("Theta",self.theta))        
        fh_stats.write("# %-5s %8.4f  \n" % ("neff",neff1))

        if(plot):
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            try:
                import seaborn as sns
                sns.set_style("white")
                sns.set_context("notebook")
                c1 = sns.color_palette()[0]
                c2 = sns.color_palette()[1]
                c3 = sns.color_palette()[2]
            except:
                print "# not using Seaborn. Nevermind."
                c1 = (0.29803921568627451, 0.44705882352941179, 0.69019607843137254)
                c2 = (0.33333333333333331, 0.6588235294117647, 0.40784313725490196)
                c3 = (0.7686274509803922, 0.30588235294117649, 0.32156862745098042)
            pdf = PdfPages('%s.pdf' % outfile)

        # read files
        for k in range(len(sim_files)):
            
            sim_data = np.array(self.read_sim(sim_files[k]))
            fh_stats.write("#  %s \n" % (sim_files[k].split("/")[-1]))
            fh_stats.write("# %5s %8s %8s \n" % ("","before","after"))
            for l in range(sim_data.shape[1]):
                if(np.isnan(sim_data[0,l])):
                    print "Skipping column %d" % (l+1)
                    continue
                fh_stats.write(" %-5d %8.4f %8.4f \n" % (l,np.average(sim_data[:,l],weights=self.w0),np.average(sim_data[:,l],weights=self.w_opt)))
                if(plot):
                    bins = int(np.sqrt(sim_data.shape[0]*neff1*2))
                    if(bins<5):
                        print "#  Using %d bins (?)", bins
                    h1,e1 = np.histogram(sim_data[:,l],weights=self.w0,bins=bins)
                    h2,e2 = np.histogram(sim_data[:,l],weights=self.w_opt,bins=bins)
                    ee = 0.5*(e1[1:]+e1[:-1])
                    plt.plot(ee,h1,c=c1,label="Before")
                    plt.plot(ee,h2,c=c2,label="After")
                    plt.axvline(np.average(sim_data[:,l],weights=self.w0),c=c1,ls='--')
                    plt.axvline(np.average(sim_data[:,l],weights=self.w_opt),c=c2,ls='--')
                    plt.title("%s" % (sim_files[k].split("/")[-1]))
                    plt.legend()
                    pdf.savefig()
                    plt.close()

            fh_stats.close()
            
        if(plot):
            pdf.close()


'''        
        def func_maxent_laplace(lambdas):

            # weights
            ww  = self.w0*np.exp(-np.sum(lambdas[np.newaxis,:]*self.sim_data,axis=1))
            # normalization 
            zz = np.sum(ww)
            ww /= zz
            # new averages
            avg = np.sum((ww[:,np.newaxis]*self.sim_data), axis=0)
            # errors are rescaled by factor theta
            err = theta*self.exp_data[:,1]
            
            # integral error
            eps2 = np.sqrt(2.*err)/(1.-0.5*(err*lambdas**2))
            eps2 = np.sum(np.log(eps2))
    
            # experimental value 
            sum1 = np.dot(lambdas,self.exp_data[:,0])
            fun = sum1 + eps2+ np.log(zz)
            # gradient
            lap0 = -lambdas*err
            lap1 = lap0/(1.+0.5*(lambdas*lap0))
            jac = exp_avg[:,0] - lap1 - avg

            return  fun,jac
'''
