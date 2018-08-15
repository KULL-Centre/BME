# run  toy_model with different theta
import os
import numpy as np

thetas = [0.0,0.1,0.5,1,2.0,3.0,4,5,6,7,8,9,10,15,20,30,40,100,1000000]
os.system("echo ! toy model theta scan > scan_theta.dat")
for t in thetas:
    cmd = "python ../toy_model/toy_model.py %.1f noplot | grep theta >> scan_theta.dat" % t 
    os.system(cmd)
data = np.array([[float(line.split()[2]),float(line.split()[4]),float(line.split()[6]),float(line.split()[8])] for line in open("scan_theta.dat") if ("!" not in line)])


import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import ticker
from scipy import optimize
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

# plot solid lines, all data
chi2 = 0.5*(data[:,2]**2+data[:,3]**2)
fig, ax1 = plt.subplots(figsize=(4, 4))
ax1.plot(data[:,1],chi2,c='#878787',ls="-",label="y coordinate")


# plot selected scatter points
idxs = [0,3,4,5,12,18]
for i in idxs:
    ax1.scatter(data[i,1],chi2[i],c='#878787')

# add labels for selected values of theta
idxs1 = [0,3,4,5,12,18]
xmi,xma= ax1.get_ylim()
lev = xma +0.2
for i in idxs1:
    xx= (-xmi + chi2[i])/(xma-xmi)
    ax1.axvline(data[i,1],ymin=xx,ls='--',c='gray',lw=0.3,zorder=0)
    if(data[i,0]>0.99):
        if(i==18):
            ax1.text(data[i,1],lev,"Prior",va='bottom',fontsize=8,ha='center',rotation=90)
        else:
            ax1.text(data[i,1],lev,r"$\theta=%.0f$" % data[i,0],va='bottom',fontsize=8,ha='center',rotation=90)
    else:
        ax1.text(data[i,1],lev,r"$\theta=%.1f$" % data[i,0],va='bottom',fontsize=8,ha='center',rotation=90)
ax1.set_ylabel(r'$\chi^2$')
ax1.set_xlabel(r'$N_{eff}$')
ax1.set_xlim(0.25,1.05)
ax1.set_xscale('log')
ax1.set_xticks([0.3,0.4,0.6,0.8,1.0])
ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%3.1f"))
ax1.xaxis.set_minor_formatter(ticker.NullFormatter())

plt.subplots_adjust(left=0.12, right=0.95, top=0.83, bottom=0.1)
plt.savefig("fig02B.pdf",dpi=600)
                
