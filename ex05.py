import numpy as np
import importlib 
import time
import matplotlib.pyplot as plt
import pylab
pylab.ion()
pylab.show()
from scipy.optimize import minimize
from scipy.stats import norm
import seaborn as sns
sns.set()
'''
BRIEF DESCRIPTION:
This is an example file associated with the code library for Chapter 5. It
uses the simulation approach for key firm inputs to construct a direct
implementation of the Merton model. The base portfolio has only six firms, but
this can be changed as desired. Asset and default correlations, summary risk
measures, and tail probabilities are also provided.
-----------------
David Jamieson Bolder, February 2018
'''
# This is the base location for your code implementation
# You'll need to change this to reflect your own personal location
myHome = "/home/djb/Work/cmBook/GitHub/"
# Loading the necessary libraries
import cmUtilities as util
import binomialPoissonModels as bp
import mixtureModels as mix
import thresholdModels as th
import mertonModel as mert
importlib.reload(util)
importlib.reload(bp)
importlib.reload(mix)
importlib.reload(th)
importlib.reload(mert)
plt.close('all')
# Key inputs and parameters
M = 1000000  
r= 0.01    
dt = 1     
myRho = 0.20
alpha = np.array([0.95,0.97,0.99,0.995,0.999,0.9997,0.9999])
numberOfModels = 1
# Set aside some memory
el = np.zeros([numberOfModels])
ul = np.zeros([numberOfModels])
var = np.zeros([len(alpha),numberOfModels])
es = np.zeros([len(alpha),numberOfModels])
cTime = np.ones(numberOfModels)
# High-dimensional simulated dataset (initialized in underlying loop)
N = 6  # Can play with this parameter to see how it influences the results
portfolioSize = N*10
c6 = (portfolioSize/N)*np.ones(N)     
rateList = np.array(['AAA','AA','A','BBB'])
phat = np.zeros([N])
daRatio = np.zeros([N])
E = np.zeros([N])
sigmaE = np.zeros([N])
sharpeRatio = np.zeros([N])
muE = np.zeros([N])
A = np.zeros([N])
sigma = np.zeros([N])
K = np.zeros([N])
optionDelta = np.zeros([N])
mu = np.zeros([N])
Etest = np.zeros([N])
EsigmaTest = np.zeros([N])
myDelta = np.zeros([N])
defaultProb = np.zeros([N])
rnDefaultProb = np.zeros([N])
D2D = np.zeros([N])
sigmaA = np.zeros([N])
varA = np.zeros([N])
rating = []
for n in range(0,N):
    # Randomly assign a rating
    rating.append(mert.assignRating(rateList))
    # Use rating to randomly assign a default probability
    daRatio[n] = mert.assignDebtToAssetRatio(rating[n])
    # Randomly assign an equity value
    E[n] = np.random.uniform(50,150)
    # Compute a default barrier from the debt-to-equity ratio
    K[n] = (daRatio[n]/(1-daRatio[n]))*E[n]
    # Use rating to randomly assign an equity volatility 
    sigmaE[n] = mert.assignEquityVolatility(rating[n])
    # Randomly assign Sharpe ratio--to create consistency between
    # expected equity returns and equity volatility
    sharpeRatio[n] = np.random.uniform(0.05,0.10)
    # Compute expected equity returns (invert Sharpe ratio)
    muE[n] = r + sigmaE[n]*sharpeRatio[n]
    # Use the default probabilities, equity volatility and equity
    # values to solve for the asset values and asset volatilities
    x = np.array([E[n]+K[n],sigmaE[n]])
    res = minimize(mert.minimizeG,x,args=(r,sigmaE[n],dt,E[n],K[n]),method='Powell',options={'xtol': 1e-8, 'disp': False})
    # Assign optimized asset values and volatilities
    A[n] = res.x[0]
    sigma[n] = res.x[1]
    # Compute the option delta (to get asset returns)
    optionDelta[n] = mert.getOptionDelta(r,sigma[n],dt,A[n],K[n])
    # Compute the asset returns (under the physical measure)
    #mu[n] = (E[n]/A[n])*optionDelta[n]*muE[n]
    mu[n] = r + sigma[n]*sharpeRatio[n]
    # Use Black-Scholes formula to solve for the theoretical price
    Etest[n] = mert.getE(r,sigma[n],dt,A[n],K[n])
    # Check the relationship between instantaneous asset and equity volatility
    EsigmaTest[n] = optionDelta[n]*(A[n]/E[n])*sigma[n]
    # Compute the default probabilties under physical measure    
    myDelta[n] = mert.getDelta(mu[n],sigma[n],dt,A[n],K[n])
    defaultProb[n] = norm.cdf(myDelta[n])
    rnDefaultProb[n] = norm.cdf(mert.getDelta(r,sigma[n],dt,A[n],K[n]))
    # Compute the asset-value variance and volatility    
    sigmaA[n] = mert.getSigmaA(mu[n],sigma[n],dt,A[n],K[n])
    varA[n] = mert.getVarA(mu[n],sigma[n],dt,A[n],K[n])
# Print the key inputs
print("#\t R\t Asset\t K\t p\t sigmaA\t muA")
for n in range(0,N):
    print("%d\t %s\t %0.1f\t %0.1f\t %0.1f\t %0.1f\t %0.1f\t" 
    % (n+1, rating[n], A[n], K[n], 1e4*defaultProb[n], 1e2*sigma[n], 1e2*mu[n]))
# Asset moments and other important inputs
EA = mert.getExpA(mu,dt,A)
VA = mert.getVarA(mu,sigma,dt,A,K)
SA = np.sqrt(VA)
C = mert.generateCorrelationMatrix(N,myRho)    
CA = np.zeros([N,N])
myOmega = np.zeros([N,N])
for n in range(0,N):
    for m in range(0,N):
        CA[n,m] = mert.getCorAB(sigma[n],sigma[m],C[n,m],dt)
        myOmega[n,m] = mert.getCovAB(A[n],A[m],mu[n],mu[m],
                       sigma[n],sigma[m],CA[n,m],dt)
print("ASSET correlation matrix")
util.printMatrix(CA,'%0.2f')
DC = np.zeros([N,N])
for n in range(0,N):
    for m in range(0,N):
        DC[n,m] = mert.getDefaultCorAB(A[n],A[m],
                  mu[n],mu[m],sigma[n],sigma[m],CA[n,m],dt,K[n],K[m])
print("DEFAUL correlation matrix")
util.printMatrix(DC,'%0.2f')
print("Running MERTON MODEL")
V = np.diag(sigmaA)
OmegaA = np.dot(np.dot(V,CA),V)
hatA,Omega = mert.computeAssetValueMoments(N,A,mu,sigma,CA,dt)  
startTime = time.clock() 
el[0],ul[0],var[:,0],es[:,0] = mert.mertonDirectSimulation(N,M,K,hatA,OmegaA,c6,alpha)
cTime[0] = (time.clock() - startTime)   
# =====================
# TABLE: Key VaR Model results
# =====================
print("Alpha\t VaR\t ES")
for n in range(0,len(alpha)):
    print("%0.2fth\t %0.1f\t %0.1f" % (1e2*alpha[n],var[n],es[n]))
print("Expected loss: %0.1f" % (el))
print("Loss volatility: %0.1f" % (ul))
print("CPU Time: %0.1f" % (cTime))
# =====================
plt.figure(1) # Plot the independent default simulation results
# =====================
plt.plot(var,alpha,color='red',linestyle='-',label='VaR')
plt.plot(es,alpha,color='blue',linestyle='--',label='Expected shortfall')
plt.xlabel('USD')
plt.ylabel('Quantile')
plt.legend(loc=4)


