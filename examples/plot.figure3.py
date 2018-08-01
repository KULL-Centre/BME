import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import sys

#sns.set_style("white",{'axes.linewidth': 0.5})
sns.set_style("white")
sns.set_context("paper")


#sugars = ["H1H2","H2H3","H3H4"]
#colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
#cols = sns.palplot(sns.xkcd_palette(colors))
#c3 = sns.xkcd_rgb["grey"]
#c1 = sns.xkcd_rgb["windows blue"]
#c2 = sns.xkcd_rgb["green"]
#c4 = sns.xkcd_rgb["pale red"]
c1 = sns.xkcd_rgb["grey"]
c2 = sns.xkcd_rgb["pale red"]
c3 = sns.xkcd_rgb["denim blue"]
t=40


ty = {"NOE":0,"UNOE":1,"J3":2}
data_noe = []
data_j3 = []
data_unoe = []

n_bins = 5
nframes = 20000
sqrt_nbins = 1./np.sqrt(n_bins)

for j in range(n_bins):
    fh = open("example2_%d.stats.dat" % (j))
    data = [[],[],[]]
    for line in fh:
        if("NOE_exp.dat" in line):
            dt = "NOE"
        if("uNOE_exp.dat" in line):
            dt = "UNOE"
        if("couplings_exp.dat" in line):
            dt = "J3"
        if("#" not in line):
            l = float(line.split()[1])
            s = float(line.split()[2])
            b = float(line.split()[3])
            a = float(line.split()[4])
            data[ty[dt]].append([l,s,b,a])
    fh.close()
    
    data_noe.append(data[0])
    data_unoe.append(data[1])
    data_j3.append(data[2])
data_noe = np.array(data_noe)
data_unoe = np.array(data_unoe)
data_j3 = np.array(data_j3)



# scalar couplings
xx1 = np.arange(data_noe.shape[1])
ss = np.argsort(data_noe[0,:,0])
ms = 4.5
ff = 's'
fig,ax = plt.subplots(1,1,figsize=(4,4))
plt.subplots_adjust(left=0.16, bottom=0.15, right=0.98, top=0.92)
ax.set_ylabel("NOE ($\AA$)")
ax.set_xlabel("Index")
#ax.set_title("Scalar couplings")

ax.errorbar(xx1,data_noe[0,ss,0],yerr=data_noe[0,ss,1],c=c1,fmt=ff,markersize=ms,label="EXP")

ax.errorbar(xx1,np.average(data_noe[:,ss,2],axis=0),yerr=sqrt_nbins*np.std(data_noe[:,ss,2],axis=0,ddof=1),\
            fmt=ff,c=c2,markersize=ms,alpha=0.8,label="no reweight")
ax.errorbar(xx1,np.average(data_noe[:,ss,3],axis=0),yerr=sqrt_nbins*np.std(data_noe[:,ss,3],axis=0,ddof=1),\
            fmt=ff,c=c3,markersize=ms,alpha=0.8,label=r"\theta=40")

plt.savefig("fig03B.pdf",dpi=600)
plt.close()

xx1 = np.arange(data_j3.shape[1])
ss = np.argsort(data_j3[0,:,0])
ms = 4.5
ff = 's'
fig,ax = plt.subplots(1,1,figsize=(4,4))
plt.subplots_adjust(left=0.16, bottom=0.15, right=0.98, top=0.92)
ax.set_ylabel("$^3J$ (Hz)")
ax.set_xlabel("Index")
ax.set_title("Scalar couplings")

ax.errorbar(xx1,data_j3[0,ss,0],yerr=data_j3[0,ss,1],c=c1,fmt=ff,markersize=ms,label="EXP")

ax.errorbar(xx1,np.average(data_j3[:,ss,2],axis=0),yerr=sqrt_nbins*np.std(data_j3[:,ss,2],axis=0,ddof=1),\
            fmt=ff,c=c2,markersize=ms,alpha=0.8,label="no reweight")
ax.errorbar(xx1,np.average(data_j3[:,ss,3],axis=0),yerr=sqrt_nbins*np.std(data_j3[:,ss,3],axis=0,ddof=1),\
            fmt=ff,c=c3,markersize=ms,alpha=0.8,label=r"\theta=40")

plt.savefig("fig03A.pdf",dpi=600)
plt.close()

#####################
#xx1 = np.arange(data_unoe.shape[1])
#ss = np.argsort(data_unoe[0,:,0])
#ms = 3
#ff = 's'
#fig,ax = plt.subplots(1,1,figsize=(6,4))
#plt.subplots_adjust(left=0.16, bottom=0.15, right=0.98, top=0.92)
#ax.set_ylabel("UNOE ($\AA$)")
#ax.set_xlabel("Index")
##ax.set_title("Scalar couplings")
#
#ax.errorbar(xx1,data_unoe[0,ss,0],yerr=data_unoe[0,ss,1],c=c1,fmt='.',markersize=0.001,label="EXP")
#
#ax.errorbar(xx1,np.average(data_unoe[:,ss,2],axis=0),yerr=sqrt_nbins*np.std(data_unoe[:,ss,2],axis=0,ddof=1),\
#            fmt=ff,c=c2,markersize=ms,alpha=0.8,label="no reweight")
#ax.errorbar(xx1,np.average(data_unoe[:,ss,3],axis=0),yerr=sqrt_nbins*np.std(data_unoe[:,ss,3],axis=0,ddof=1),\
#            fmt=ff,c=c3,markersize=ms,alpha=0.8,label=r"\theta=40")
#ax.set_ylim([2,7])
#
#plt.savefig("fig03_unoe.pdf",dpi=600)
#
#plt.close()

# now do ermsd plot
data = np.array([float(line.split()[1]) for line in open("../data/ermsd.opc.dat") if "#" not in line])
fig,ax = plt.subplots(1,1,figsize=(4,4))
plt.subplots_adjust(left=0.16, bottom=0.15, right=0.98, top=0.92)
ax.set_ylabel("probability")
ax.set_xlabel("eRMSD from A-form")
hist_0 = []
hist_post = []
binsize = nframes/n_bins
bins = np.linspace(0,2,75)
for j in range(n_bins):
    # read weights
    weights = np.array([float(line.split()[1]) for line in open("example2_%d.weights.dat" % j) if "#" not in line])
    hh_0, ee_0 = np.histogram(data[j*binsize:(j+1)*binsize],normed=True,bins=bins)
    hist_0.append(hh_0)
    hh_post, ee_1 = np.histogram(data[j*binsize:(j+1)*binsize],normed=True,bins=bins,weights=weights)
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
plt.savefig("fig03C.pdf",dpi=600)
