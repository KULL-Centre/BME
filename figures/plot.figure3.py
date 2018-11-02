import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import sys

sns.set_style("white")
sns.set_context("paper")


c1 = sns.xkcd_rgb["grey"]
c2 = sns.xkcd_rgb["pale red"]
c3 = sns.xkcd_rgb["denim blue"]
t=40


data_noe = []
data_j3 = []
data_unoe = []

n_bins = 5
nframes = 20000
sqrt_nbins = 1./np.sqrt(n_bins)

datat = ["couplings","noe"]
data = [[],[],[]]
for i,d in enumerate(datat):
    for j in range(n_bins):
        data_tmp = []
        fh = open("../examples/example2_%d_%s.stats.dat" % (j,d))
        for line in fh:
            if("#" not in line):
                if(i==0):
                    l = float(line.split()[1])
                    s = float(line.split()[2])
                    b = float(line.split()[3])
                    a = float(line.split()[4])
                else:
                    l = float(line.split()[5])
                    s = float(line.split()[6])
                    b = float(line.split()[7])
                    a = float(line.split()[8])
                    
                data_tmp.append([l,s,b,a])
        fh.close()
        data[i].append(data_tmp)
        
data_j3 = np.array(data[0])
data_noe = np.array(data[1])
data_unoe = np.array(data[2])


# scalar couplings
xx1 = np.arange(data_noe.shape[1])
ss = np.argsort(data_noe[0,:,0])
ms = 3.0
ff = 's'
fig,ax = plt.subplots(1,1,figsize=(3.2,2.5))
plt.subplots_adjust(left=0.16, bottom=0.15, right=0.98, top=0.92)
ax.set_ylabel("NOE ($\AA$)")
ax.set_xlabel("Index")

ax.errorbar(xx1,data_noe[0,ss,0],yerr=data_noe[0,ss,1],c=c1,fmt=ff,markersize=ms,label="EXP")

ax.errorbar(xx1,np.average(data_noe[:,ss,2],axis=0),yerr=sqrt_nbins*np.std(data_noe[:,ss,2],axis=0,ddof=1),\
            fmt=ff,c=c2,markersize=ms,alpha=0.8,label="no reweight")
#ax.errorbar(xx1,np.average(data_noe[:,ss,3],axis=0),yerr=sqrt_nbins*np.std(data_noe[:,ss,3],axis=0,ddof=1),\
#            fmt=ff,c=c3,markersize=ms,alpha=0.8,label=r"\theta=40")

plt.savefig("fig03B.pdf",dpi=600)
plt.close()

xx1 = np.arange(data_j3.shape[1])
ss = np.argsort(data_j3[0,:,0])
ms = 3.0
ff = 's'
fig,ax = plt.subplots(1,1,figsize=(3.2,2.5))
plt.subplots_adjust(left=0.16, bottom=0.15, right=0.98, top=0.92)
ax.set_ylabel("$^3J$ (Hz)")
ax.set_xlabel("Index")
ax.set_title("Scalar couplings")

ax.errorbar(xx1,data_j3[0,ss,0],yerr=data_j3[0,ss,1],c=c1,fmt=ff,markersize=ms,label="EXP")

ax.errorbar(xx1,np.average(data_j3[:,ss,2],axis=0),yerr=sqrt_nbins*np.std(data_j3[:,ss,2],axis=0,ddof=1),\
            fmt=ff,c=c2,markersize=ms,alpha=0.8,label="no reweight")
plt.savefig("fig03A.pdf",dpi=600)
plt.close()

### # scalar couplings
xx1 = np.arange(data_noe.shape[1])
ss = np.argsort(data_noe[0,:,0])
ms = 3.0
ff = 's'
fig,ax = plt.subplots(1,1,figsize=(3.2,2.5))
plt.subplots_adjust(left=0.16, bottom=0.15, right=0.98, top=0.92)
ax.set_ylabel("NOE ($\AA$)")
ax.set_xlabel("Index")

ax.errorbar(xx1,data_noe[0,ss,0],yerr=data_noe[0,ss,1],c=c1,fmt=ff,markersize=ms,label="EXP")
ax.errorbar(xx1,np.average(data_noe[:,ss,2],axis=0),yerr=sqrt_nbins*np.std(data_noe[:,ss,2],axis=0,ddof=1),\
            fmt=ff,c=c2,markersize=ms,alpha=0.8,label="no reweight")

ax.errorbar(xx1,np.average(data_noe[:,ss,3],axis=0),yerr=sqrt_nbins*np.std(data_noe[:,ss,3],axis=0,ddof=1),\
            fmt=ff,c=c3,markersize=ms+1,alpha=0.8,label=r"\theta=40")

plt.savefig("fig03E.pdf",dpi=600)
plt.close()

xx1 = np.arange(data_j3.shape[1])
ss = np.argsort(data_j3[0,:,0])
ms = 3.0
ff = 's'
fig,ax = plt.subplots(1,1,figsize=(3.2,2.5))
plt.subplots_adjust(left=0.16, bottom=0.15, right=0.98, top=0.92)
ax.set_ylabel("$^3J$ (Hz)")
ax.set_xlabel("Index")
ax.set_title("Scalar couplings")

ax.errorbar(xx1,data_j3[0,ss,0],yerr=data_j3[0,ss,1],c=c1,fmt=ff,markersize=ms,label="EXP")
ax.errorbar(xx1,np.average(data_j3[:,ss,2],axis=0),yerr=sqrt_nbins*np.std(data_j3[:,ss,2],axis=0,ddof=1),\
            fmt=ff,c=c2,markersize=ms,alpha=0.8,label=r"\theta=40")

ax.errorbar(xx1,np.average(data_j3[:,ss,3],axis=0),yerr=sqrt_nbins*np.std(data_j3[:,ss,3],axis=0,ddof=1),\
            fmt=ff,c=c3,markersize=ms+1,alpha=0.8,label=r"\theta=40")

plt.savefig("fig03D.pdf",dpi=600)
plt.close()


# now do ermsd plot
data = np.array([float(line.split()[1]) for line in open("../data/ermsd.opc.dat") if "#" not in line])
fig,ax = plt.subplots(1,1,figsize=(4,3))
plt.subplots_adjust(left=0.16, bottom=0.15, right=0.98, top=0.92)
ax.set_ylabel("probability")
ax.set_xlabel("eRMSD from A-form")
hist_0 = []
hist_post = []
binsize = nframes/n_bins
bins = np.linspace(0,2,75)
for j in range(n_bins):
    # read weights
    weights = np.array([float(line.split()[1]) for line in open("../examples/example2_%d.weights.dat" % j)
                        if (("#" not in line) and (len(line.split())==2))])
    
    hh_0, ee_0 = np.histogram(data[j*binsize:(j+1)*binsize],density=True,bins=bins)
    hist_0.append(hh_0)
    hh_post, ee_1 = np.histogram(data[j*binsize:(j+1)*binsize],density=True,bins=bins,weights=weights)
    print np.sum(hh_0*(ee_0[1]-ee_0[0]))
    print np.sum(hh_post*(ee_1[1]-ee_1[0]))
    hist_post.append(hh_post)

xx = 0.5*(ee_0[1:]+ee_0[:-1])
avg_0 = np.average(hist_0,axis=0)
sem_0 = sqrt_nbins*np.std(hist_0,axis=0,ddof=1)
ax.plot(xx,avg_0,color=c2)
ax.fill_between(xx,avg_0-sem_0,avg_0+sem_0,color=c2,alpha=0.3)
avg_post = np.average(hist_post,axis=0)
sem_post = sqrt_nbins*np.std(hist_post,axis=0,ddof=1)
ax.plot(xx,avg_post,color=c3)
ax.fill_between(xx,avg_post-sem_post,avg_post+sem_post,color=c3,alpha=0.3)
plt.savefig("fig03F.pdf",dpi=600)
