import numpy as np
import importlib 
import matplotlib.pyplot as plt
import pylab
pylab.ion()
pylab.show()
import time
import scipy
import seaborn as sns
sns.set()
'''
BRIEF DESCRIPTION:
This is an example file associated with the code library for Chapter 10. In 
this example, we estimate the one-factor Gaussian threshold model using a pre-
simulated dataset. The likelihood function and standard-error bounds are 
presented in detail. Finally, we examing the method-of-moments approach for a
collection of six different mixture models. The fit and parameter estimates
are summarized in tabular form.
-----------------
David Jamieson Bolder, February 2018
'''
# This is the base location for your code implementation
# You'll need to change this to reflect your own personal location
myHome = "/home/djb/Work/cmBook/GitHub/"
# This is the simulated transition-data file
expFile = myHome+"exposures.npy"
XFile = myHome+"X.npy"
YFile = myHome+"Y.npy"
pMapFile = myHome+"pMap.npy"
allPFile = myHome+"allP.npy"
X0File = myHome+"X0.npy"
# Loading the necessary libraries
import cmUtilities as util
import binomialPoissonModels as bp
import mixtureModels as mix
import thresholdModels as th
import mertonModel as mert
import irbModel as irb
import varContributions as vc
import varianceReduction as vr
import markovChain as mc
import assetCorrelation as ac
importlib.reload(util)
importlib.reload(bp)
importlib.reload(mix)
importlib.reload(th)
importlib.reload(mert)
importlib.reload(irb)
importlib.reload(vc)
importlib.reload(vr)
importlib.reload(mc)
importlib.reload(ac)
plt.close('all')
# Key inputs and parameters
P = mc.inventedTransitionMatrix()
portfolioSize = 1000
N = 100 
T = 30  
K = 4   
myRho = 0.20
nu = 120 
isT = 0 
c = np.load(expFile)  
# Some quantiles, not sure we'll need them
alpha = np.array([0.99,0.999,0.9997,0.9999])
# Rating definitions, weights in each rating (at inception)
cRatings = np.array(['A','B','C','D'])
wStart = np.array([0.40,0.30,0.30,0.00])
# =============================================
# 1/ Simple (1-parameter) MLE Example
# =============================================
# Load pre-simulated data
X = np.load(XFile)
Y = np.load(YFile)
pMap = np.load(pMapFile)
allP = np.load(allPFile)
X0 = np.load(X0File)
# To generate your own data, merely use the following function call
# X,Y,pMap,allP,X0 = ac.createRatingData(K,N,T,P,wStart,myRho,nu,isT)  
# Extract key elements from the data
pVec,nVec,kVec = ac.getSimpleEstimationData(T,X,allP)      
Phat = mc.estimateTransitionMatrix(K,T,N,X,1,0)
print("====  True Transition Matrix ==========")
util.printMatrix(P,'%0.4f')
print("====  Estimated Transition Matrix ==========")
util.printMatrix(Phat,'%0.4f')
print("====  Difference ==========")
util.printMatrix(P-Phat,'%0.4f')
# Grid search
gridSize = 100
rhoRange = np.linspace(0.001,0.999,gridSize)
logL = np.zeros(gridSize)
cTime  = np.zeros(gridSize)
for m in range(0,gridSize):
    if np.mod(m+1,25)==0:
        print("Running ITERATION %d" % (m+1))        
    startTime = time.time() 
    logL[m] = -ac.logLSimple(rhoRange[m],T,pVec,nVec,kVec)       
    cTime[m] = time.time() - startTime
print("Each iteration requires about %0.1f seconds" % (np.mean(cTime)))
print("Total time is %0.1f minutes" % (np.sum(cTime)/60))
print("Examining LOG-LIKELIHOOD FUNCTION.")
plt.figure(1)
rhoMax = rhoRange[logL==max(logL)]
normLogL = logL/np.abs(np.max(logL))
plt.plot(rhoRange,normLogL,color='gray',label=r'$\ell(\rho_G)$')
plt.plot(rhoMax,np.max(normLogL),marker='o',
             linestyle='none',color='red',label=r'$\rho_G^*$')
plt.plot(rhoRange,-1*np.ones(gridSize),color='black',
         linestyle=':',linewidth=2,label='Maximum')
plt.plot(myRho,-1,marker='^',color='darkorange',
         linestyle='none',label=r'True $\rho_G$')
plt.ylim([-1.5,-0.95])
plt.xlim([0,1])
plt.legend(loc=3,numpoints=1)
# Optimization approach
rhoHat,successFlag = ac.maxSimpleLogL(T,pVec,nVec,kVec)
myScore = ac.computeSimpleScore(rhoHat,T,pVec,nVec,kVec)
fInf = ac.simpleFisherInformation(rhoHat,T,pVec,nVec,kVec)
se = np.sqrt(-np.reciprocal(fInf))
print("Printing ONE-FACTOR MLE RESULTS")
print("Quantity\t\t\t Estimate")
print("--------\t\t\t --------")
print("True parameter value\t\t %0.5f" % (myRho))
print("Grid search optimum\t\t %0.5f" % (rhoMax))
print("Hill-climber optimum\t\t  %0.5f" % (rhoHat))
print("Score function\t\t\t %0.5f" % (myScore))
print("Observed Fisher information\t %0.2f" % (-fInf))
print("Standard error\t\t\t %0.2f" % (se))
print("95\%% CI\t\t\t\t [%0.2f,%0.2f]" % (np.maximum(rhoHat-1.96*se,0),np.minimum(rhoHat+1.96*se,1)))
# =============================================
# 2/ Method-of-Moments Example
# =============================================
# Observed moments
dHat = np.mean(kVec/nVec)
vHat = np.var(kVec/nVec)
mList = np.array(['Beta','Logit','Probit',
                  'P-gamma','P-LN','P-WB'])
numModels = 6
x0 = np.array([[0.1,0.2],
               [-1,0.7],
               [-1,0.7],
               [0.48,0.005],
               [-1,0.7],
               [0.48,0.005]])
M1 = np.zeros(numModels)
M2 = np.zeros(numModels)
dCorr = np.zeros(numModels)
mOrder = np.array([1,2,3,5,6,7])
mixPara = np.zeros([numModels,2])
for n in range(0,numModels):
    mixPara[n,:] = scipy.optimize.fsolve(ac.mixtureMethodOfMoment,x0[n,:],args=(dHat,vHat,n))  
    M1[n],M2[n]= mix.calibrateVerify(mixPara[n,0],mixPara[n,1],dHat,0.015,mOrder[n])
    dCorr[n] = mix.defCorr(M1[n],M2[n])
print("Printing METHOD-OF-MOMENTS RESULTS")
print("Model\t M1\t M1Hat\t M2\t M2Hat\t p1\t p2\t RHO")
print("-----\t --\t -----\t --\t -----\t --\t --\t ---")

for n in range(0,numModels):
    print("%s\t %0.2f\t %0.2f\t %0.2f\t %0.2f\t %0.2f\t %0.2f\t %0.3f" % (mList[n],1e4*dHat,
                        1e4*M1[n],1e4*np.sqrt(vHat),
                        1e4*np.sqrt(M2[n]-M1[n]**2),
                        mixPara[n,0],mixPara[n,1],dCorr[n]))


