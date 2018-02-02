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
This is an example file associated with the code library for Chapter 3. In 
this example, we simulate the binomial independent-default, beta-binomial 
mixture, and logit-normal mixture models. We then print out a subset of their
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
importlib.reload(util)
importlib.reload(bp)
importlib.reload(mix)
plt.close('all')
# Key inputs and parameters
c = np.load(expFile)  
p = np.load(dpFile)
N = len(p)
myRho = 0.05  
portfolioSize = np.sum(p)
myC = portfolioSize/N
myP = np.mean(p)
M = 1000000
numberOfModels = 3
alpha = np.array([0.95,0.97,0.99,0.995,0.999,0.9997,0.9999])
# Set aside some memory
el = np.zeros([numberOfModels])
ul = np.zeros([numberOfModels])
var = np.zeros([len(alpha),numberOfModels])
es = np.zeros([len(alpha),numberOfModels])
cTime = np.zeros(numberOfModels)
a = np.zeros(numberOfModels)
b = np.zeros(numberOfModels)
M1 = np.zeros(numberOfModels)
M2 = np.zeros(numberOfModels)
print("Running BINOMIAL MODEL")
# (a) Calibrate
M1[0],M2[0]=mix.calibrateVerify(a[0],b[0],myP,myRho,0)
# (b) Simulate
startTime = time.clock() 
el[0],ul[0],var[:,0],es[:,0] = bp.independentBinomialSimulation(N,M,p,c,alpha)
cTime[0] = (time.clock() - startTime)
print("Running BETA-BINOMIAL MODEL")
# (a) Calibrate
a[1],b[1] = mix.betaCalibrate(myP,myRho)
M1[1],M2[1]=mix.calibrateVerify(a[1],b[1],myP,myRho,1)
# (b) Simulate
startTime = time.clock() 
el[1],ul[1],var[:,1],es[:,1] = mix.betaBinomialSimulation(N,M,c,a[1],b[1],alpha)
cTime[1] = (time.clock() - startTime)
print("Running LOGIT-NORMAL MODEL")
# (a) Calibrate
logit = scipy.optimize.fsolve(mix.logitProbitCalibrate, np.array([1,1]), \
                              args=(myP,myRho,1))  
a[2] = logit[0]
b[2] = logit[1]
M1[2],M2[2]=mix.calibrateVerify(a[2],b[2],myP,myRho,2)
# (b) Simulate
startTime = time.clock() 
el[2],ul[2],var[:,2],es[:,2] = \
           mix.logitProbitBinomialSimulation(N,M,c,a[2],b[2],alpha,1)
cTime[2] = (time.clock() - startTime)
# =====================
# TABLE: Key VaR Model results
# =====================
print("Alpha\t VaR_b\t VaR_bb\t VaR_ln")
for n in range(0,len(alpha)):
    print("%0.2fth\t %0.1f\t %0.1f\t %0.1f" % (1e2*alpha[n],var[n,0],var[n,1],var[n,2]))
print("Expected loss: %0.1f vs. %0.1f vs. %0.1f" % (el[0],el[1],el[2]))
print("Loss volatility: %0.1f vs. %0.1f vs. %0.1f" % (ul[0],ul[1],ul[2]))
print("CPU Time: %0.1f vs. %0.1f vs. %0.1f" % (cTime[0],cTime[1],cTime[2]))
# =====================
plt.figure(1) # Plot the independent default simulation results
# =====================
plt.plot(var[:,0],alpha,color='red',linestyle='-',label='Binomial')
plt.plot(var[:,1],alpha,color='blue',linestyle='--',label='Beta binomial')
plt.plot(var[:,2],alpha,color='green',linestyle='-.',label='Logit normal')
plt.xlabel('USD')
plt.ylabel('Quantile')
plt.legend(loc=4)