import numpy as np
import importlib 
import matplotlib.pyplot as plt
import pylab
pylab.ion()
pylab.show()
from scipy.stats import t as myT
import numpy.linalg as anp
import seaborn as sns
sns.set()
'''
BRIEF DESCRIPTION:
This is an example file associated with the code library for Chapter 9. In 
this example, we load a pre-simulated transition dataset, estimate the 
transition matrix with the cohort approach and then compute and compare the
uncertainty of these estimates using the standard MLE Fisher-information approach
and the bootstrapping methodology. Results are provided both in graphic and 
tabular format.
-----------------
David Jamieson Bolder, February 2018
'''
# This is the base location for your code implementation
# You'll need to change this to reflect your own personal location
myHome = "/home/djb/Work/cmBook/GitHub/"
# This is the transition-data file
transitionFile = myHome+"transitionData.npy"
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
importlib.reload(util)
importlib.reload(bp)
importlib.reload(mix)
importlib.reload(th)
importlib.reload(mert)
importlib.reload(irb)
importlib.reload(vc)
importlib.reload(vr)
importlib.reload(mc)
plt.close('all')
# Key parameters and inputs
wStart = np.array([0.25,0.50,0.25,0.00])
N = 100
T = 30 
K = 4              
cRatings = np.array(['A','B','C','D'])
P = mc.inventedTransitionMatrix()
# Pre-simulated transition data
D = np.load(transitionFile)
# If you wish to generate new data, then use:
# D = mc.simulateRatingData(N,T,P,wStart)
plt.figure(1)
cnt = np.abs(np.diff(D))
numberOfJumps = np.zeros(K)
for n in range(0,K):
    numberOfJumps[n] = np.sum(cnt==n)
plt.bar(np.linspace(0,K-1,K)-0.35,
                    1e2*numberOfJumps/(N*(T+1)),
                    color='cornflowerblue')    
plt.xticks(np.linspace(0,K-1,K).astype(int),
           np.linspace(0,K-1,K).astype(int))
plt.xlabel('Rating Notch Change')
plt.ylabel('Proportion (Per cent)')
# Compute the empirical number of transitions
M = mc.getTransitionCount(K,T,N,D)
print("We expect to observe %d transitions." % (N*(T-1)))
print("We observe, however, %d transitions." % (np.sum(M)))
print("And, %d default(s)." % (np.sum(D[:,-1]==4)))
print("Counted default(s) are %d." % (M[-1,-1]))
print("Default span %d periods." % (np.sum(D==4)-np.sum(D[:,-1]==4)))
print("+++++  Counts +++++++")
# Estimate the transition matrix with cohort approach
Phat = mc.estimateTransitionMatrix(K,T,N,D,1,0)
print("+++++  Estimated P +++++++")
util.printMatrix(Phat,"%0.4f")  
print("+++++  True P +++++++")
util.printMatrix(P,"%0.4f")  
print("+++++  Error +++++++")
util.printMatrix(np.abs(P-Phat),"%0.4f")  
print("The average error is %0.2f" % (1e2*np.mean(np.abs(P[0:-1,:]-Phat[0:-1,:]))))
# Compute SE by the observed Fisher information
xTrue = np.reshape(P, K*K)
x = np.reshape(Phat, K*K)
mVector = np.reshape(M,len(M)**2)
H1 = -np.diag(np.divide(mVector,x**2))
seAnl = np.sqrt(-np.diag(anp.inv(H1[0:-K,0:-K])))
# Examine analytic results
print("Printing FISHER-INFORMATION CONFIDENCE BOUNDS")
print("p\t pHat\t SE\t CI")
print("-\t ----\t --\t --")
for n in range(0,K**2-K):
    print("p_%d\t %0.2f\t %0.2f\t [%0.2f,%0.2f]" % 
                         (n+1,
                          x[n],
                          seAnl[n],
                          np.maximum(x[n]-myT.ppf(1-0.05/2,T-1)*seAnl[n],0),
                          np.minimum(x[n]+myT.ppf(1-0.05/2,T-1)*seAnl[n],1)))

plt.figure(2)
myLabels = np.array([r'$p_{11}$',r'$p_{12}$',r'$p_{13}$',r'$p_{14}$',
                     r'$p_{21}$',r'$p_{22}$',r'$p_{23}$',r'$p_{24}$',
                     r'$p_{31}$',r'$p_{32}$',r'$p_{33}$',r'$p_{34}$'])
plt.bar(np.linspace(1,K**2-K,K**2-K)-0.40,x[0:-K],yerr=myT.ppf(1-0.05/2,T-1)*seAnl,
        color='grey',label='ML estimates')
plt.plot(np.linspace(1,K**2-K,K**2-K),xTrue[0:-K],marker='*',
         color='red',linestyle='None',label='True value')
plt.xticks(np.linspace(1,K**2-K,K**2-K).astype(int),
           myLabels)
plt.xlim([0.25,K**2-K+0.5])
plt.ylim([0,1])
plt.xlabel('Parameter')
plt.ylabel('Value')
plt.legend(loc=0,numpoints=1)
print("Printing BOOTSTRAP CONFIDENCE BOUNDS")
S = 1000 # Probably should be larger, but in the interests of time and illustration   
Ptry =  mc.bootstrapDistribution(K,N,T,Phat,wStart,S)   
print("p\t pHat\t SE\t CI")
print("-\t ----\t --\t --")
cnt = 0
for n in range(0,K):
    for m in range(0,K):
        trySort = 1e2*np.sort(Ptry[n,m,:],axis=None)
        print("p_%d\t %0.1f\t %0.1f\t [%0.1f,%0.1f]" % 
                         (cnt+1,
                          np.mean(trySort),
                          np.std(trySort),
                          trySort[np.ceil(0.025*(S-1)).astype(int)],
                          trySort[np.ceil(0.975*(S-1)).astype(int)]))
        cnt += 1
# Plot the results    
xTry = np.zeros(K**2)
xBounds = np.zeros([2,K**2])
cnt = 0
for i in range(0,K):
    for j in range(0,K):
        trySort = np.sort(Ptry[i,j,:],axis=None)
        xTry[cnt] = np.mean(trySort)
        xBounds[0,cnt] = xTry[cnt]-trySort[np.ceil(0.025*(S-1)).astype(int)]
        xBounds[1,cnt] = trySort[np.ceil(0.975*(S-1)).astype(int)]-xTry[cnt]
        cnt += 1
plt.figure(3)
plt.bar(np.linspace(1,K**2-K,K**2-K)-0.40,xTry[0:-K],
        yerr=xBounds[:,0:-K],color='lightsage',label='Bootstrap')
plt.plot(np.linspace(1,K**2-K,K**2-K),xTrue[0:-K],marker='*',
         color='red',linestyle='None',label='True value')
plt.xticks(np.linspace(1,K**2-K,K**2-K).astype(int),
           myLabels)
plt.xlim([0.25,K**2-K+0.5])
plt.xlabel('Parameter')
plt.ylabel('Value')
plt.legend(loc=0,numpoints=1)