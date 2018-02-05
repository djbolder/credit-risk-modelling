import numpy as np
import importlib 
import matplotlib.pyplot as plt
import pylab
pylab.ion()
pylab.show()
from scipy.stats import t as myT
import seaborn as sns
sns.set()
'''
BRIEF DESCRIPTION:
This is an example file associated with the code library for Chapter 8. In 
this example, we compute the VaR contributions from the t-threshold model
using the saddlepoint, brute-force MC, and importance-sampling Monte-Carlo
methods. Even with the importance-sampling estimator, quite a lot of computation
is required to reproduce the saddlepoint estimates. We also produce a 95% 
confidence interval for the VaR and expected-shortfall estimates.
-----------------
David Jamieson Bolder, February 2018
'''
# This is the base location for your code implementation
# You'll need to change this to reflect your own personal location
myHome = "/home/djb/Work/cmBook/GitHub/"
# These are the exposure and default-probability files
dpFile = myHome+"defaultProbabilties.npy"
expFile = myHome+"exposures.npy"
regionFile = myHome+"regions.npy"
# Loading the necessary libraries
import cmUtilities as util
import binomialPoissonModels as bp
import mixtureModels as mix
import thresholdModels as th
import mertonModel as mert
import irbModel as irb
import varContributions as vc
import varianceReduction as vr
importlib.reload(util)
importlib.reload(bp)
importlib.reload(mix)
importlib.reload(th)
importlib.reload(mert)
importlib.reload(irb)
importlib.reload(vc)
importlib.reload(vr)
plt.close('all')
# Model input and parameters
c = np.load(expFile)  
p = np.load(dpFile)
N = len(c)
alpha = np.array([0.95,0.97,0.99,0.995,0.999,0.9997,0.9999])
myP =np.mean(p)
# See ex04.py for the calibration of the t-threshold model
rhoT = 0.06339367516538337
nu = 8.0269146874964417
M = 1000000
S = 10 # Caution, not very many.
print("Running t-THRESHOLD MODEL. Caution: it's a bit slow!")
el,ul,var,es = th.oneFactorThresholdModel(N,M,p,c,rhoT,nu,alpha,1)
VaRC = vc.myVaRCYT(var[-1],p,c,rhoT,nu,6)
print("Printing RAW MONTE-CARLO RESULTS")
C,V,E = vc.mcThresholdTDecomposition(N,M,S,p,c,rhoT,nu,1,alpha[-1])
mcVaRC = np.mean(C,1)[:,0]
print("Printing IMPORTANCE-SAMPLING RESULTS")
Mis = 24000
eps = 5
tailProb = np.zeros([S])
varTarget = np.mean(V)
isContributions = np.zeros([N,S])
for s in range(0,S):
    print("Running IS Estimator: %d" % (s+1))
    testIS,thetaZStar,pZ,qZ,cgf,rnDerivative = \
           vr.isThresholdContr(N,Mis,p,c,varTarget,rhoT,nu,-0.2)
    L_is = np.dot(c,testIS)
    thresholdIndicator = L_is>varTarget
    tailProb[s] = np.mean(np.multiply(thresholdIndicator,rnDerivative))
    # Compute the contributions with importance-sampling technique
    keyPoint = varTarget
    varIndicator = (np.abs(L_is-keyPoint)<eps)
    while np.sum(varIndicator)==0:
        eps += eps
        varIndicator = (np.abs(L_is-keyPoint)<eps)
    eps = 5
    for n in range(0,N):
        num = np.dot(testIS[n,varIndicator],
                     np.multiply(rnDerivative[varIndicator],
                     c[n]))
        den = np.sum(rnDerivative[varIndicator])
        isContributions[n,s] = num/den 
isVaRC = np.mean(isContributions,1)
# Ordering results
Q=np.vstack([np.linspace(1,N,N),VaRC])
Q=Q[:,np.argsort(-Q[1, :])]
loc = Q[0,:].astype(int)-1
print("Printing MONTE-CARLO RESULTS")
nView = 10
print("#\t Rank\t VaR_sp\t VaR_mc\t VaR_is")
for n in range(0,nView):
    print("%d\t %d\t %0.1f\t %0.1f\t %0.1f" % (n+1,loc[n],VaRC[loc[n]],
                                       mcVaRC[loc[n]],
                                       isVaRC[loc[n]]))
print("--------------------------------------------------------")
print("Total\t\t %0.1f\t %0.1f\t %0.1f" % (np.sum(VaRC),
                                           np.sum(mcVaRC),
                                           np.sum(isVaRC)))
print("Generating 99% CONFIDENCE INTERVALS")
varCheck = np.zeros([len(alpha),S])
esCheck = np.zeros([len(alpha),S])
for s in range(0,S):
    elCheck,ulCheck,varCheck[:,s],esCheck[:,s] = \
    th.oneFactorThresholdModel(N,M,p,c,rhoT,nu,alpha,1)
print("Alpha\t Mean\t CI")
print("-----\t ----\t --")
print("VaR")
print("-----------------------------")
for n in range(0,len(alpha)):
    gamma = myT.ppf(0.95,S-1)*np.std(varCheck[n,:])/np.sqrt(S-1)
    print("%0.2fth\t %0.1f\t [%0.1f,%0.1f]" % (1e2*alpha[n],np.mean(varCheck[n,:]),
                                       np.mean(varCheck[n,:])-gamma,
                                       np.mean(varCheck[n,:])+gamma))
print("-----------------------------")
print("ES")
print("-----------------------------")
for n in range(0,len(alpha)):
    gamma = myT.ppf(0.95,S-1)*np.std(esCheck[n,:])/np.sqrt(S-1)
    print("%0.2fth\t %0.1f\t [%0.1f,%0.1f]" % (1e2*alpha[n],np.mean(esCheck[n,:]),
                                       np.mean(esCheck[n,:])-gamma,
                                       np.mean(esCheck[n,:])+gamma))

