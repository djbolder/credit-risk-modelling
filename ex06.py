import numpy as np
import importlib 
import matplotlib.pyplot as plt
import pylab
pylab.ion()
pylab.show()
import time
import seaborn as sns
sns.set()
'''
BRIEF DESCRIPTION:
This is an example file associated with the code library for Chapter 6. In 
this example, we compute the IRB, ASRF, and one-factor Gaussian models. We 
also use Gordy and Luetkebohmert's granularity adjustment to reconcile the IRB
and one-factor Gaussian model for concentration risk. The approximation, given
its calibration, work much better at the higher quantiles than for the lower
ones, although all are presented. As a final, foreshadowing note, we also compute
and illustrate the 99.99th quantile VaR contributions associated with the IRB
model.
-----------------
David Jamieson Bolder, February 2018
'''
# This is the base location for your code implementation
# You'll need to change this to reflect your own personal location
myHome = "/home/djb/Work/cmBook/GitHub/"
# These are the exposure and default-probability files
dpFile = myHome+"defaultProbabilties.npy"
expFile = myHome+"exposures.npy"
tenorFile = myHome+"tenors.npy"
# Loading the necessary libraries
import cmUtilities as util
import binomialPoissonModels as bp
import mixtureModels as mix
import thresholdModels as th
import mertonModel as mert
import irbModel as irb
importlib.reload(util)
importlib.reload(bp)
importlib.reload(mix)
importlib.reload(th)
importlib.reload(mert)
importlib.reload(irb)
plt.close('all')
# Key inputs and parameters
c = np.load(expFile)  
p = np.load(dpFile)
N = len(c)
tenor = np.load(tenorFile)
alpha = np.array([0.95,0.97,0.99,0.995,0.999,0.9997,0.9999])
myP = np.mean(p)
myRho = irb.getRho(myP)
nu = 9
portfolioSize = 1000
N = 100
rhoTarget = 0.0225
tProbTarget = 0.01
pTarget = np.mean(p)
M=1000000
numberOfModels = 4
# Set aside some memory
el = np.zeros([numberOfModels])
ul = np.zeros([numberOfModels])
var = np.zeros([len(alpha),numberOfModels])
es = np.zeros([len(alpha),numberOfModels])
cTime = np.zeros(numberOfModels)
# Compute weighted average IRB asset correlation
irbP = np.zeros(N)
for n in range(0,N):
    irbP[n] = irb.getRho(p[n])
rhoG = np.dot(irbP,c)/np.sum(c)
print("Running IRB MODEL")
el[0],ul[0],var[:,0],es[:,0],cTime[0] = irb.runModelSuite(N,M,p,c,alpha,nu,rhoG,rhoTarget,tenor,4)
print("Running ASRF MODEL")
el[1],ul[1],var[:,1],es[:,1],cTime[1] = irb.runModelSuite(N,M,p,c,alpha,nu,rhoG,rhoTarget,tenor,5)
print("Running GAUSSIAN THRESHOLD MODEL")
startTime = time.time() 
el[2],ul[2],var[:,2],es[:,2] = th.oneFactorGaussianModel(N,M,p,c,rhoG,alpha)
cTime[2] = (time.time() - startTime)
print("Running GRANULARITY ADJUSTMENT")
myA = 0.25
xi = 0.25
gBar = 1
GA = np.zeros(len(alpha))
for n in range(0,len(alpha)):    
    GA[n]=irb.granularityAdjustmentCR(myA,
            irb.getW(myP,myA,rhoG,alpha[n]),gBar,xi,p,c,alpha[n])
# =====================
# TABLE: Key VaR Model results
# =====================
print("Alpha\t VaR_i\t VaR_a\t VaR_g\t IRB+GA")
for n in range(0,len(alpha)):
    print("%0.2fth\t %0.1f\t %0.1f\t %0.1f\t %0.1f" % (1e2*alpha[n],var[n,0],var[n,1],var[n,2],var[n,0]+GA[n]))
print("Expected loss: %0.1f vs. %0.1f vs. %0.1f" % (el[0],el[1],el[2]))
print("Loss volatility: %0.1f vs. %0.1f vs. %0.1f" % (ul[0],ul[1],ul[2]))
print("CPU Time: %0.1f vs. %0.1f vs. %0.1f" % (cTime[0],cTime[1],cTime[2]))
# =====================
plt.figure(1) # Plot the independent default simulation results
# =====================
plt.plot(var[:,0],alpha,color='red',linestyle='-',label='IRB')
plt.plot(var[:,1],alpha,color='blue',linestyle='--',label='ASRF')
plt.plot(var[:,2],alpha,color='green',linestyle='-.',label='Gaussian')
plt.plot(var[:,0]+GA,alpha,color='gray',linestyle=':',label='IRB + GA',linewidth=3)
plt.xlabel('USD')
plt.ylabel('Quantile')
plt.legend(loc=4)
print("(TOP TEN) Obligor VaR CONTRIBUTIONS at 99.99th quantile.")
myContributions = np.zeros([N])
for n in range(0,N):
    irbPart = irb.getBaselRiskCapital(p[n],tenor[n],c[n],np.array([0.9999]))
    gaPart = irb.gaContribution(myA,irb.getW(myP,myA,rhoG,0.9999),
                            gBar,xi,p,c,n,0.9999)
    myContributions[n] = irbPart + gaPart
Q=np.vstack([np.linspace(1,N,N),myContributions])
Q=Q[:,np.argsort(-Q[1, :])]
loc = Q[0,:].astype(int)-1
nView=10
print("#\t Rank\t p\t c\t VaR_n\t %")
for n in range(0,nView):
    print("%d\t %d\t %0.1f%%\t $%0.1f\t $%0.1f\t %0.1f%%" % 
    (n+1,
     Q[0,n],
     1e2*p[loc[n]],
     c[loc[n]],
     myContributions[loc[n]],
     1e2*myContributions[loc[n]]/np.sum(myContributions)))
print("Total is $%0.1f" % (np.sum(myContributions)))

