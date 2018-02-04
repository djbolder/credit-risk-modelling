import numpy as np
import importlib 
import matplotlib.pyplot as plt
import pylab
pylab.ion()
pylab.show()
import seaborn as sns
sns.set()
import scipy
'''
BRIEF DESCRIPTION:
This is an example file associated with the code library for Chapter 7. In 
this example, we use the saddlepoint approximation to determine the VaR
contributions of four models: the independent-default binomial, Gaussian threshold
Weibull-Poisson mixture, and t-threshold models. The results are presented by
both obligor and regional distinctions. Finally, we use the Monte-Carlo approach
to estimate the VaR and expected-shortfall contributions for the one-factor
Gaussian threshold model. It shows clearly the relative noisiness of the VaR
contributions relative to the "true" analytic values. We thus also include
VaR-matched expected-shortfall idea from Bluhm, Overbeck, and Wagner (2003).
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
importlib.reload(util)
importlib.reload(bp)
importlib.reload(mix)
importlib.reload(th)
importlib.reload(mert)
importlib.reload(irb)
importlib.reload(vc)
plt.close('all')
# Key inputs and parameters
c = np.load(expFile)  
p = np.load(dpFile)
N = len(c)
alpha = np.array([0.95,0.97,0.99,0.995,0.999,0.9997,0.9999])
myP =np.mean(p)
rhoT = 0.173
rhoTarget = 0.05  
tDependenceTarget = 0.02
nu = 9.6
rId = (np.load(regionFile)-1).astype(int)
numberOfModels = 4
M = 1000000
rhoTarget = 0.02
# Set aside some memory
el = np.zeros([numberOfModels])
ul = np.zeros([numberOfModels])
var = np.zeros([len(alpha),numberOfModels])
es = np.zeros([len(alpha),numberOfModels])
print("Running INDEPENDENT-DEFAULT BINOMIAL MODEL")
el[0],ul[0],var[:,0],es[:,0] = bp.independentBinomialSimulation(N,M,p,c,alpha)
indVaRC = vc.getVaRC(var[-1,0],p,c)
print("Running GAUSSIAN-THRESHOLD MODEL")
resultG = scipy.optimize.minimize(th.calibrateGaussian,0.2,args=(myP,rhoTarget))
rhoG = resultG.x

el[1],ul[1],var[:,1],es[:,1] = th.oneFactorThresholdModel(N,M,p,c,rhoG,nu,alpha,0)
gVaRC = vc.myVaRCY(var[-1,1],p,c,rhoG,0,0)
print("Running WEIBULL-POISSON MIXTURE MODEL")
weibull = scipy.optimize.fsolve(mix.poissonMixtureCalibrate, \
                                np.array([0.5,0.01]),args=(myP,rhoTarget,1))
a = weibull[0]
b = weibull[1]
M1,M2=mix.calibrateVerify(a,b,myP,rhoTarget,7)
el[2],ul[2],var[:,2],es[:,2] = mix.poissonMixtureSimulation(N,M,c,a,b,alpha,1)
wVaRC = vc.myVaRCY(var[-1,2],p,c,a,b,5)
print("Running t-THRESHOLD MODEL. Caution: it's a bit slow!")
tModel = scipy.optimize.fsolve(th.tCalibrate, np.array([0.2,10]), 
                            args=(myP,rhoTarget,tDependenceTarget))  
rhoT = tModel[0]
nu = tModel[1]
el[3],ul[3],var[:,3],es[:,3] = th.oneFactorThresholdModel(N,M,p,c,rhoT,nu,alpha,1)
tVaRC = vc.myVaRCYT(var[-1,3],p,c,rhoT,nu,6)
# Print the results
Q=np.vstack([np.linspace(1,N,N),indVaRC])
Q=Q[:,np.argsort(-Q[1, :])]
loc = Q[0,:].astype(int)-1
nView = 10
print("Printing VaR CONTRIBUTIONS")
print("#\t Rank\t B\t G\t W\t T")
for n in range(0,nView):
    print("%d\t %d\t %0.1f\t %0.1f\t %0.1f\t %0.1f" % (n+1,loc[n],
                                       indVaRC[loc[n]],
                                       gVaRC[loc[n]],
                                       wVaRC[loc[n]],
                                       tVaRC[loc[n]]))
print("---------------------------")
print("Total\t\t %0.1f\t %0.1f\t %0.1f\t %0.1f" % (np.sum(indVaRC),
                                           np.sum(gVaRC),
                                           np.sum(wVaRC),
                                           np.sum(tVaRC)))
# Compute and graph the regional contributions
numberOfRegions = len(np.unique(rId))
rVaRC = np.zeros([numberOfRegions,numberOfModels])
for n in range(0,numberOfRegions):
    rVaRC[n,0] = np.sum(indVaRC[rId==n])
    rVaRC[n,1] = np.sum(gVaRC[rId==n])
    rVaRC[n,2] = np.sum(wVaRC[rId==n])
    rVaRC[n,3] = np.sum(tVaRC[rId==n])
howMany = 3
ind = np.linspace(1,howMany,howMany)    
num_items = 3
margin = 0.15
width = (1.-2.*margin)/num_items
plt.figure(1)
plt.bar(ind+margin+(0*width),rVaRC[:,0],color='aqua',width=width,label=r'Binomial')
plt.bar(ind+margin+(1*width),rVaRC[:,1],color='gold',width=width,label=r'Gaussian')
plt.bar(ind+margin+(2*width),rVaRC[:,2],color='firebrick',width=width,label=r'Weibull')
plt.bar(ind+margin+(3*width),rVaRC[:,3],color='blue',width=width,label=r'$t$')
plt.xlim([1.0,howMany+1.25])
plt.ylim([0,110])
plt.xticks(np.array([1,2,3])+0.65,np.array(['Alpha','Bravo','Charlie']))
plt.xlabel('Region')
plt.ylabel('USD')
plt.legend(loc=2)
# Gaussian threshold Monte-Carlo VaR and ES contributions
myAlpha = 0.9999
M = 1000000
S = 5 # For VaR, this should be much bigger
# Brute-force Monte-Carlo estimator
mcRisk,varG,esG =  vc.mcThresholdGDecomposition(N,M,S,p,c,rhoG,0,0,alpha[-1])
varMC = np.mean(mcRisk[:,:,0],1)
esMC = np.mean(mcRisk[:,:,1],1)
# Analytical expected-shortfall contributions
esCG = vc.myESCY(var[-1,1],p,c,rhoG,0,0)
# VaR-matched Monte-Carlo estimator 
# With only one-million iterations, there will be a bit of simulation noise
ahat = scipy.optimize.minimize_scalar(vc.findAlphaGaussian,0.9999,
                    args=(N,M,p,c,var[-1,1],rhoG), 
                    method='bounded',bounds=(0.0001,0.9999))  
myAlpha = ahat.x    
mcMatchedRisk,varMatched,esMatched =  vc.mcThresholdGDecomposition(N,M,S,p,c,
                                                rhoG,0,0,myAlpha)
varMatchedC = np.mean(mcMatchedRisk[:,:,1],1)
# Print out the results
print("Printing MONTE-CARLO RESULTS")
nView = 10
print("#\t Rank\t VaR_sp\t VaR_mc\t VaR_mt\t ES_sp\t ES_mc")
for n in range(0,nView):
    print("%d\t %d\t %0.1f\t %0.1f\t %0.1f\t %0.1f\t %0.1f" % (n+1,loc[n],
                                       gVaRC[loc[n]],
                                       varMC[loc[n]],
                                       varMatchedC[loc[n]],
                                       esCG[loc[n]],
                                       esMC[loc[n]]))
print("--------------------------------------------------------")
print("Total\t\t %0.1f\t %0.1f\t %0.1f\t %0.1f\t %0.1f" % (np.sum(gVaRC),
                                           np.sum(varMC),
                                           np.sum(varMatchedC),
                                           np.sum(esCG),
                                           np.sum(esMC)))