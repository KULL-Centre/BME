import sys
import numpy as np
from scipy import optimize

def myfunc(samples):
    return samples

def gamma(lambdas):
    avg = np.sum(lambdas[:,np.newaxis]*myfunc(samples),axis=0)
    logz = np.log(np.sum(np.exp(-avg)))
    eps =  np.sum(0.5*theta*((lambdas**2)*(exp_sigma**2)))
    sum1 = np.dot(lambdas,[avgx_exp,avgy_exp])
    return logz+sum1 + eps



# set theta
if(len(sys.argv)>=2):
    theta = float(sys.argv[1])
else:
    theta = 3.0

# If noplot == True, no plot is produced
noplot=False
if(len(sys.argv)==3 and sys.argv[2]=="noplot"):
    noplot=True

# First, construct the true probability distribution as a sum of
# three two-dimensional Gaussian distributions.

# number of samples
nn_true = 1E06
# populations of the three states
f1_true = int(0.2*nn_true)
f2_true = int(0.3*nn_true)
f3_true = int(0.5*nn_true)
# define mean and convariance of the Gaussians
mean1_true = [2,2]
cov1_true = [[0.1,0],[0,0.05]]
mean2_true = [3,7.5]
cov2_true = [[0.12,0],[0,0.2]]
mean3_true = [7,5.5]
cov3_true = [[0.1,0.1],[0.1,0.8]]
# Construct Gaussians
x1, y1 = np.random.multivariate_normal(mean1_true, cov1_true, f1_true).T  
x2, y2 = np.random.multivariate_normal(mean2_true, cov2_true, f2_true).T 
x3, y3 = np.random.multivariate_normal(mean3_true, cov3_true, f3_true).T
# concatenate samples
x_true = np.array(list(x1)+list(x2)+list(x3))
y_true = np.array(list(y1)+list(y2)+list(y3))

# Here, generate samples from a prior distribution p0. The code is the same as above.

# number of samples
nn_0 = 1E05
# populations
f1_0 = int(0.55*nn_0)
f2_0 = int(0.4*nn_0)
f3_0 = int(0.05*nn_0)
# Define mean and covariance
mean1_0 = [2.,2.]
cov1_0 = [[0.05,0],[0,0.09]]
mean2_0 = [3,7.5]
cov2_0 = [[0.15,0],[0,0.3]]
mean3_0 = [7,5.5]
cov3_0 = [[0.25,0.1],[0.08,1.25]]
# Create Gaussians
x1, y1 = np.random.multivariate_normal(mean1_0, cov1_0, f1_0).T  
x2, y2 = np.random.multivariate_normal(mean2_0, cov2_0, f2_0).T 
x3, y3 = np.random.multivariate_normal(mean3_0, cov3_0, f3_0).T
# concatenate samples
x_0 = list(x1)+list(x2)+list(x3)
y_0 = list(y1)+list(y2)+list(y3)


# Calculate the average position in Ptrue and P0
avgx_true = np.average(x_true)
avgy_true = np.average(y_true)
# set an arbitrary experimental uncetainty

avgx_exp = np.average(x_true) + 0.6
avgy_exp = np.average(y_true) - 0.5
exp_sigma = np.array([0.8,0.6])

avgx_0 = np.average(x_0)
avgy_0 = np.average(y_0)

print("# True average  x=%5.3f y=%5.3f" % (avgx_true,avgy_true))
print("# Experimental average  x=%5.3f y=%5.3f" % (avgx_exp,avgy_exp))
print("# Sigma x=%5.3f sigma y=%5.3f" % (exp_sigma[0],exp_sigma[1]))
print("# Prior average x=%5.3f y=%5.3f" % (avgx_0,avgy_0))

########################
# Here starts the optimization

# initialize lambdas to zero
lambdas = np.zeros(2)
# set some optimization variables. 
opt={'maxiter':1000,'disp': False,'ftol':1.0e-10}
meth="L-BFGS-B"

# do optimization
samples = np.array([x_0,y_0])
print("# Start minimization. Theta is set to %4.1f" % theta)
result = optimize.minimize(gamma,lambdas,options=opt,method=meth)
print("# Minimization was succesful? %s" % result.success)
# calculate weights
w_post = np.exp(-np.sum(result.x[:,np.newaxis]*myfunc(samples),axis=0))
w_post /= np.sum(w_post)
avg_post = np.sum(w_post[np.newaxis,:]*myfunc(samples),axis=1)
print("# Posterior average x=%5.3f y=%5.3f" % (avg_post[0],avg_post[1]))
print("# theta %4.1f Neff %5.3f " % (theta, np.exp(-np.sum(w_post*np.log(len(w_post)*w_post)))))
print("chix %5.3f chiy %5.3f" % (np.abs((avg_post[0]-avgx_exp))/exp_sigma[0],np.abs((avg_post[1]-avgy_exp))/exp_sigma[1]))


########################################
# Now do some plotting. 
########################################
if(noplot): sys.exit(0)

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")


# Define bins
xmin,xmax= (0,10)
ymin,ymax= (0,10)
binsx = np.linspace(xmin,xmax,100)
binsy = np.linspace(xmin,xmax,100)
bins2d = [np.linspace(xmin,xmax,60),np.linspace(ymin,ymax,60)]
lw=1.6

# here define boundaries for states
xb=5
yb=4
idx_yl = np.where(bins2d[1]<yb)[0]
idx_xl = np.where(bins2d[0]<xb)[0]
idx_yu = np.where(bins2d[1]>=yb)[0][:-1]
idx_xu = np.where(bins2d[0]>=xb)[0][:-1]
# definitions for the axes
nullfmt = NullFormatter()         # no labels
left, width = 0.12, 0.58
bottom, height = 0.12, 0.58
bottom_h = left_h = left + width + 0.07

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(4, 3.6))
axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# set limits and levels
axScatter.set_ylim(ymin,ymax)
axScatter.set_xlim(xmin,xmax)
axHistx.set_ylim(0,1)
axHisty.set_xlim(0,1)
lv = [1.e0-10,1.0e-04,1.0e-03,1.0e-2,1.0e-1,1.0e-0]

cols1 = sns.color_palette("Greys")
cols2 = sns.color_palette("YlOrRd")
cols3 = sns.color_palette("Blues")

# plot reference distribution
hhr,xer,yer = np.histogram2d(x_true,y_true,bins=bins2d,density=True)
CS = axScatter.contourf(0.5*(xer[1:] + xer[:-1]), 0.5*(yer[1:] + yer[:-1]),hhr.T,inline=1, fontsize=10,levels=lv,colors=cols1)

hh2r,ee2r = np.histogram(x_true,bins=binsx,density=True)
axHistx.plot(0.5*(ee2r[1:]+ee2r[:-1]),hh2r,lw=lw,c=cols1[-1],label="Sample")

hh2r,ee2r = np.histogram(y_true,bins=binsy,density=True)
axHisty.plot(hh2r,0.5*(ee2r[1:]+ee2r[:-1]),lw=lw,c=cols1[-1])

axScatter.errorbar(avgx_exp,avgy_exp,xerr=exp_sigma[0],yerr=exp_sigma[1],fmt="s",capthick=0.5,capsize=2,color="m",zorder=10)
axScatter.scatter(avgx_true,avgy_true,marker="o",color="k")

# calculate population of the states
pop = np.sum(hhr)
pop3_true = np.sum(hhr[idx_xu,:])/pop
pop1_true = np.sum(hhr[idx_xl,:][:,idx_yl])/pop
pop2_true = 1.-pop3_true-pop1_true

######################################################
# Now plot P0
#######################################################

hh,xe,ye = np.histogram2d(x_0,y_0,bins=bins2d,density=True)
CS = axScatter.contour(0.5*(xe[1:] + xe[:-1]), 0.5*(ye[1:] + ye[:-1]),hh.T,inline=1, fontsize=10,levels=lv,colors=cols2)
hh2,ee2 = np.histogram(x_0,bins=binsx,density=True)
axHistx.plot(0.5*(ee2[1:]+ee2[:-1]),hh2,lw=lw,c=cols2[-2],label="Sample")

hh2,ee2 = np.histogram(y_0,bins=binsy,density=True)
axHisty.plot(hh2,0.5*(ee2[1:]+ee2[:-1]),lw=lw,c=cols2[-2])
axScatter.scatter(avgx_0,avgy_0,marker="*",facecolors='none',edgecolors=cols2[-2],s=100,linewidth=1.5)
pop = np.sum(hh)
pop3_0 = np.sum(hh[idx_xu,:])/pop
pop1_0 = np.sum(hh[idx_xl,:][:,idx_yl])/pop
pop2_0 = 1.-pop3_0-pop1_0

# Now plot lines and set labels
axScatter.set_xlabel("x",fontsize=12)
axScatter.set_ylabel("y",fontsize=12)
axScatter.axvline(5,ls="--",c='k',lw=0.5,dashes=(5, 10))
axScatter.axhline(4,ls="--",c='k',lw=0.5,xmax=0.5,dashes=(5, 10))
axScatter.text(0.2,9,"S2",fontweight='bold')
axScatter.text(0.2,3.2,"S1",fontweight='bold')
axScatter.text(9,9,"S3",fontweight='bold')
axHistx.set_ylabel('p(x)')
axHisty.set_xlabel('p(y)')

plt.savefig("fig02A.pdf",dpi=600)
plt.close()

################################################################
# Plot reference state again
###############################################################

left, width = 0.12, 0.58
bottom, height = 0.12, 0.58
bottom_h = left_h = left + width + 0.07

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(4.0, 3.6))
axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# set limits and levels
axHistx.set_ylim(0,1)
axHisty.set_xlim(0,1)
axScatter.set_ylim(ymin,ymax)
axScatter.set_xlim(xmin,xmax)


CS = axScatter.contourf(0.5*(xer[1:] + xer[:-1]), 0.5*(yer[1:] + yer[:-1]),hhr.T,inline=1, fontsize=10,levels=lv,colors=cols1)

hh2r,ee2r = np.histogram(x_true,bins=binsx,density=True)
axHistx.plot(0.5*(ee2r[1:]+ee2r[:-1]),hh2r,lw=lw,c='k',label="Sample")
hh2r,ee2r = np.histogram(y_true,bins=binsy,density=True)
axHisty.plot(hh2r,0.5*(ee2r[1:]+ee2r[:-1]),lw=lw,c='k')

###############################
# Plot posterior
###############################

hh,xe,ye = np.histogram2d(x_0,y_0,bins=bins2d,density=True,weights=w_post)
pop = np.sum(hh)
pop3_post = np.sum(hh[idx_xu,:])/pop
pop1_post = np.sum(hh[idx_xl,:][:,idx_yl])/pop
pop2_post = 1.-pop3_post-pop1_post

CS = axScatter.contour(0.5*(xe[1:] + xe[:-1]), 0.5*(ye[1:] + ye[:-1]),hh.T,inline=1, fontsize=10,levels=lv,colors=cols3)

hh2,ee2 = np.histogram(x_0,bins=binsx,density=True,weights=w_post)
axHistx.plot(0.5*(ee2[1:]+ee2[:-1]),hh2,lw=lw,c=cols3[-2],label="Sample")

hh2,ee2 = np.histogram(y_0,bins=binsy,density=True,weights=w_post)
axHisty.plot(hh2,0.5*(ee2[1:]+ee2[:-1]),lw=lw,c=cols3[-2])
#axScatter.errorbar(avgx_true,avgy_true,xerr=exp_sigma[0],yerr=exp_sigma[1],fmt="s",capthick=0.5,capsize=2,color="k")
axScatter.errorbar(avgx_exp,avgy_exp,xerr=exp_sigma[0],yerr=exp_sigma[1],fmt="s",capthick=0.5,capsize=2,color="m",zorder=10)
axScatter.scatter(avgx_true,avgy_true,marker="o",color="k")

axScatter.scatter(avg_post[0],avg_post[1],marker="*",facecolors='none',edgecolors=cols3[-2],s=100,linewidth=1.5,zorder=10)

# 
axScatter.set_xlabel("x",fontsize=12)
axScatter.set_ylabel("y",fontsize=12)
axScatter.axvline(5,ls="--",c='k',lw=0.5,dashes=(5, 10))
axScatter.axhline(4,ls="--",c='k',lw=0.5,xmax=0.5,dashes=(5, 10))
axScatter.text(0.2,9,"S2",fontweight='bold')
axScatter.text(0.2,3.2,"S1",fontweight='bold')
axScatter.text(9,9,"S3",fontweight='bold')
axHistx.set_ylabel('p(x)')
axHisty.set_xlabel('p(y)')

plt.savefig("fig02C.pdf",dpi=600)
plt.close()

###############################################
## Print populations of the different states ##
###############################################
print("# Populations")
print(" True      %5.1f %5.1f %5.1f " % ((pop1_true*100.0),(pop2_true*100),(pop3_true*100)))
print(" Prior     %5.1f %5.1f %5.1f " % ((pop1_0*100.0),(pop2_0*100),(pop3_0*100)))
print(" Posterior %5.1f %5.1f %5.1f " % ((pop1_post*100.0),(pop2_post*100),(pop3_post*100)))




fig, axs =plt.subplots(2,1,figsize=(4.1, 3.6))
ii1 = [x for x in range(len(x1))]
axs[1].scatter(ii1,w_post[0:len(x1)],s=0.2,color=cols3[-2])
axs[1].text(ii1[int(len(ii1)/2)],0.15e-05,"S1",ha='center')

off= len(ii1)+0
ii2 = [x+off for x in range(len(x2))]
axs[1].scatter(ii2,w_post[len(x1):len(x1)+len(x2)],s=0.2,color=cols3[-2])
axs[1].text(ii2[int(len(ii2)/2)],0.15e-05,"S2",ha='center')

off += len(ii2)+0
ii3 = [x+off for x in range(len(x3))]
axs[1].scatter(ii3,w_post[len(x1)+len(x2):],s=0.2,color=cols3[-2])
axs[1].text(ii3[int(len(ii3)/2)],0.15e-05,"S3",ha='center')

axs[1].set_yscale('log')

axs[1].axhline(1./len(w_post),ls='--',c=cols2[-2],lw=0.75)
axs[1].text(len(w_post)+2000,1./len(w_post),r'$w_0$',va='center')

axs[1].set_ylim(0.000001,5.e-04)
#axs[1].set_ylabel(r"$w^*$")
axs[1].set_xlabel("sample index")
axs[1].set_xlim(-1000,ii3[-1]+1000)
#axs[1].ticklabel_format(style='sci')
axs[1].set_xticks([1,25000,50000,75000,100000])
#plt.ticklabel_format(style='sci', axis='x', scilimits=(6,8))
axs[0].axis('off')
#
rows = ["True","Exp",'Prior',r"$\theta=%d$" % theta ,r"$\theta=0$"]
columns= ["<x>","<y>","S1 (%)","S2 (%)","S3 (%)"]
cell_text = [["%4.1f" % avgx_true,"%4.1f" % avgy_true, "%4.0f" % (pop1_true*100.),"%4.0f" % (pop2_true*100.),"%4.0f" % (pop3_true*100.)],\
             ["%4.1f" % avgx_exp,"%4.1f" % avgy_exp, "--" ,"--","--"],\
             ["%4.1f" % avgx_0,"%4.1f" % avgy_0, "%4.0f" % (pop1_0*100.),"%4.0f" % (pop2_0*100.),"%4.0f" % (pop3_0*100.)],\
             ["%4.1f" % avg_post[0],"%4.1f" % avg_post[1], "%4.0f" % (pop1_post*100.),"%4.0f" % (pop2_post*100.), "%4.0f" % (pop3_post*100.)],\
             ["%4.1f" % 5.4,"%4.1f" % 4.9, "%4.0f" % 22.8,"%4.0f" % 15.0, "%4.0f" % 62.2]]

colors = ["#cec7c7","m",cols2[-3],cols3[-4],"#ffffff"]
rowColours=colors
the_table = axs[0].table(cellText=cell_text,\
                         rowLabels=rows,\
                         colLabels=columns,\
                         loc='best',rowColours=colors,colWidths=[0.16,0.16,0.16,0.16,0.16,0.16])
#
##pp=1.
#data = [real_avg[0],avg_b[0],avg_a[0],real_avg[1],avg_b[1],avg_a[1],\
#        (pop1r*pp),(pop1b*pp),(pop1a*pp),\
#        (pop2r*pp),(pop2b*pp),(pop2a*pp),\
#        (pop3r*pp),(pop3b*pp),(pop3a*pp)]
#xx = [1,2,3,5,6,7, 10,11,12,14,15,16,18,19,20]
#plt.bar(xx,data)
plt.savefig("fig02DE.png",dpi=600)
