import numpy as np
import importlib 
import time
import matplotlib.pyplot as plt
import pylab
pylab.ion()
pylab.show()
import scipy
import seaborn as sns
sns.set()
'''
BRIEF DESCRIPTION:
This is an example file associated with the code library for Chapter 4. In 
this example, we simulate the Gaussian threshold, and t-threshold, 
and variance-gamma threshold models. We then print out a subset of their
risk measures and graph the associated tail probabilities.
-----------------
David Jamieson Bolder, February 2018
'''
# This is the base location for your code implementation
# You'll need to change this to reflect your own personal location
myHome = "/home/djb/Work/cmBook/GitHub/"
# These are the exposure and default-probability files
dpFile = myHome+"defaultProbabilties.npy"
expFile = myHome+"exposures.npy"
# Loading the necessary libraries
import cmUtilities as util
import binomialPoissonModels as bp
import mixtureModels as mix
import thresholdModels as th
import assetCorrelation as ac
importlib.reload(util)
importlib.reload(bp)
importlib.reload(mix)
importlib.reload(th)
importlib.reload(ac)
plt.close('all')
# Key inputs and parameters
c = np.load(expFile)  
p = np.load(dpFile)
N = len(c)
portfolioSize = np.sum(c)
myC = portfolioSize/N
myP = np.mean(p)
M = 1000000
alpha = np.array([0.95,0.97,0.99,0.995,0.999,0.9997,0.9999])
startRho = 0.20
startNu = 12
rhoTarget = 0.05  
tDependenceTarget = 0.02
numberOfModels = 3
# Set aside some memory
el = np.zeros([numberOfModels])
ul = np.zeros([numberOfModels])
var = np.zeros([len(alpha),numberOfModels])
es = np.zeros([len(alpha),numberOfModels])
cTime = np.zeros(numberOfModels)
print("Running GAUSSIAN THRESHOLD MODEL")
# (a) Calibrate
resultG = scipy.optimize.minimize(th.calibrateGaussian,startRho,args=(myP,rhoTarget))
rhoG = resultG.x
# (b) Simulate
startTime = time.time() 
el[0],ul[0],var[:,0],es[:,0] = th.oneFactorGaussianModel(N,M,p,c,rhoG,alpha)
cTime[0] = (time.time() - startTime)
print("Running t THRESHOLD MODEL")
# (a) Calibrate
tModel = scipy.optimize.fsolve(th.tCalibrate, np.array([startRho,startNu]), 
                            args=(myP,rhoTarget,tDependenceTarget))  
rhoT = tModel[0]
nu = tModel[1]
# (b) Simulate
startTime = time.time() 
el[1],ul[1],var[:,1],es[:,1] = th.oneFactorThresholdModel(N,M,p,c,rhoT,nu,alpha,1)
cTime[1] = (time.time() - startTime)
print("Running VARIANCE-GAMMA MODEL")
# (a) Calibrate
kTarget = 6       
vgModel = scipy.optimize.fsolve(th.nvmCalibrate, np.array([0.2,1]), 
                                        args=(myP,rhoTarget,kTarget,0)) 
rhoVG = vgModel[0]
vgA = vgModel[1]
nvmOutcome = th.nvmCalibrate(np.array([rhoVG,vgA]),myP,0,0,0)                            
# (b) Simulate
startTime = time.time()
el[2],ul[2],var[:,2],es[:,2] = th.oneFactorNVMModel(N,M,p,c,rhoVG,vgA,alpha,0)
cTime[2] = time.time() - startTime
# =====================
# TABLE: Key VaR Model results
# =====================
print("Alpha\t VaR_g\t VaR_t\t VaR_vg")
for n in range(0,len(alpha)):
    print("%0.2fth\t %0.1f\t %0.1f\t %0.1f" % (1e2*alpha[n],var[n,0],var[n,1],var[n,2]))
print("Expected loss: %0.1f vs. %0.1f vs. %0.1f" % (el[0],el[1],el[2]))
print("Loss volatility: %0.1f vs. %0.1f vs. %0.1f" % (ul[0],ul[1],ul[2]))
print("CPU Time: %0.1f vs. %0.1f vs. %0.1f" % (cTime[0],cTime[1],cTime[2]))
# =====================
plt.figure(1) # Plot the independent default simulation results
# =====================
plt.plot(var[:,0],alpha,color='red',linestyle='-',label='Gaussian')
plt.plot(var[:,1],alpha,color='blue',linestyle='--',label=r'$t$')
plt.plot(var[:,2],alpha,color='green',linestyle='-.',label='Variance-gamma')
plt.xlabel('USD')
plt.ylabel('Quantile')
plt.legend(loc=4)