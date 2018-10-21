import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import ticker 

thetas= [0.1,0.5,1,2,3,4,5,7.5,10,15,20,40,50,100,500,5000]

nbins = 5
sb = np.sqrt(nbins)
data = []

for t in thetas:
    neff = []
    chi_noe_b = []
    chi_j3_b = []
    chi_noe_a = []
    chi_unoe_a = []
    chi_j3_a = []    
    
    for j in range(nbins):
        fh = open("../examples/example3_%d_%.1f_noe.stats.dat" % (j,t))
        for line in fh:
            if("srel" in line):
                neff.append(np.exp(-float(line.split("=")[1])))
                continue
            if("distance" in line):
                chi_noe_b.append(float(line.split()[4]))
                chi_noe_a.append(float(line.split()[5]))
        fh.close()
            
        fh = open("../examples/example3_%d_%.1f_couplings.stats.dat" % (j,t))
        for line in fh:
            if("srel" in line):
                neff.append(np.exp(-float(line.split("=")[1])))
                continue
            if("chi2" in line):
                chi_j3_b.append(float(line.split()[2]))
                chi_j3_a.append(float(line.split()[3]))
        fh.close()
        

    data.append([t, np.average(neff),\
                 np.average(chi_noe_b),np.average(chi_noe_a),\
                 np.average(chi_j3_b),np.average(chi_j3_a),\
                 np.std(chi_noe_b,ddof=1)/sb,np.std(chi_noe_a,ddof=1)/sb,\
                 np.std(chi_j3_b,ddof=1)/sb,np.std(chi_j3_a,ddof=1)/sb,\
                 np.std(neff,ddof=1)/sb])

data = np.array(data)
fig, ax1 = plt.subplots(figsize=(3.8, 3.5))
plt.subplots_adjust(left=0.15, right=0.95, top=0.83, bottom=0.12)
ax1.plot(data[:,1],data[:,3],label="NOE",c='#C1CDCD',ls="--",lw=0.75)
ax1.plot(data[:,1],data[:,5],label="J-couplings",c='k',lw=0.75)
ax1.set_ylim(-0.1,4.0)

#          0   1  2 3 4 5 6  7  8   9 10 11 12  13  14  15
#thetas= [0.1,0.5,1,2,3,4,5,7.5,10,15,20,40,50,100,500,5000]

idxs = [0,2,3,5,8,15]
for i in idxs:
    #ax1.text(np.sqrt(0.5*data[i,1]),data[i,2],"%d" % data[i,0],fontsize=10,ha='center',va='center')                                                          
    #ax1.text(np.sqrt(data[i,1]),data[i,3],"%d" % data[i,0],fontsize=10,ha='center',va='center')                                                              
    ax1.errorbar(data[i,1],data[i,3],c='#C1CDCD',yerr=data[i,7],xerr=data[i,10],ms=5,fmt="o")
    #ax1.errorbar(data[i,1],data[i,5],c='#878787',yerr=data[i,11],xerr=data[i,14],ms=5,fmt="o")
    ax1.errorbar(data[i,1],data[i,5],c='k',yerr=data[i,9],xerr=data[i,10],ms=5,fmt="o")
    

idxs1 = [0,3,8,15]
xmi,xma= ax1.get_ylim()
print xmi, xma
for i in idxs1:
    xx= (-xmi + data[i,5])/(xma-xmi)
    ax1.axvline(data[i,1],ymin=xx,ls='--',c='gray',lw=0.3,zorder=0)
    if(data[i,0]>0.99):
        if(i==15):
            ax1.text(data[i,1],4.1,"prior",va='bottom',fontsize=8,ha='center',rotation=90)
        else:
            ax1.text(data[i,1],4.1,r"$\theta=%.0f$" % data[i,0],va='bottom',fontsize=8,ha='center',rotation=90)
    else:
        ax1.text(data[i,1],4.1,r"$\theta=%.1f$" % data[i,0],va='bottom',fontsize=8,ha='center',rotation=90)
ax1.legend()

ax1.set_ylabel(r'$\chi^2$')
ax1.set_xlabel(r'$N_{eff}$')
ax1.set_xscale('log')
ax1.set_xticks([0.01,0.1,0.2,0.5,1.0])
ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%3.1f"))
ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
plt.savefig("fig03C.pdf")
