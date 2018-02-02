import numpy as np
import math
import cmUtilities as util
import numpy.linalg as anp
import importlib 
from scipy.stats import norm
import scipy.integrate as nInt
from scipy.stats import t as myT
from scipy.stats import mvn
from scipy.optimize import minimize

import thresholdModels as th


def generateCorrelationMatrix(N,rho):
    lowerDiagonal = np.zeros([N,N])
    for n in range(0,N):
        for m in range(0,N):
            if n==m:
                lowerDiagonal[n][m] = 0.5
            elif n>m:
                lowerDiagonal[n][m] = rho
            else:
                continue                
    C = np.transpose(lowerDiagonal)+lowerDiagonal
    return C
        
def getK(mu,sigma,dt,A,myP):
    t1 = (mu-0.5*(np.power(sigma,2)))*dt
    t2 = np.multiply(np.multiply(norm.ppf(myP),sigma),np.sqrt(dt))
    return np.multiply(A,np.exp(t1+t2))   

def mertonIndirectSimulation(N,M,p,Omega,c,alpha):
    Z = np.random.normal(0,1,[M,N])
    w,v = anp.eigh(Omega)
    H = np.dot(v,np.sqrt(np.diag(w)))
    xi = np.dot(Z,np.transpose(H)) 
    K = norm.ppf(p)*np.ones((M,1))        
    lossIndicator = 1*np.less(xi,K)
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)    
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es             

def mertonDirectSimulation(N,M,K,hatA,Omega,c,alpha):
    Z = np.random.normal(0,1,[M,N])
    w,v = anp.eigh(Omega)
    H = np.dot(v,np.sqrt(np.diag(w)))
    A = np.tile(hatA,(M,1)) + np.dot(Z,np.transpose(H)) 
    lossIndicator = 1*np.less(A,K)
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)    
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es         

def assignRating(rateList):
    rv = np.random.uniform()
    if rv>0 and rv <= 0.25: # AAA
        myRating = rateList[0]         
    elif rv>0.25 and rv <= 0.50: # AA
        myRating = rateList[1] 
    elif rv>0.50 and rv <= 0.75: # A
        myRating = rateList[2] 
    else:  # BBB
        myRating = rateList[3] 
    return myRating 

def assignDebtToAssetRatio(rating):
    if rating=='AAA': 
        debtToAssetRatio = np.random.uniform(0.40,0.45)         
    elif rating=='AA': 
        debtToAssetRatio = np.random.uniform(0.45,0.50)         
    elif rating=='A': 
        debtToAssetRatio = np.random.uniform(0.50,0.55)         
    else:  # BBB
        debtToAssetRatio = np.random.uniform(0.55,0.6)         
    return debtToAssetRatio 

def assignEquityVolatility(rating):
    if rating=='AAA': 
        equityVolatility = np.random.uniform(0.40,0.45)         
    elif rating=='AA': 
        equityVolatility = np.random.uniform(0.45,0.50)         
    elif rating=='A': 
        equityVolatility = np.random.uniform(0.50,0.55)         
    else:  # BBB
        equityVolatility = np.random.uniform(0.55,0.60)         
    return equityVolatility     

def getDelta(mu,sigma,dt,A,K):
    t1 = np.log(K/A)
    t2 = (mu-0.5*(sigma**2))*dt
    return np.divide(t1-t2,sigma*np.sqrt(dt))

def getD1(r,sigma,dt,A,K):
    t1 = math.log(A/K)
    t2 = (r+0.5*(sigma**2))*dt
    t3 = sigma*np.sqrt(dt)
    return np.divide(t1+t2,t3)   

def getE(r,sigma,dt,A,K):
    d1 = getD1(r,sigma,dt,A,K)
    d2 = d1 - sigma*math.sqrt(dt)
    return A*norm.cdf(d1)-np.exp(-r*dt)*K*norm.cdf(d2)

def getOptionDelta(r,sigma,dt,A,K):
    d1 = getD1(r,sigma,dt,A,K)
    optionDelta = norm.cdf(d1)
    return optionDelta 

def minimizeG(x,r,sigmaE,dt,E,K):
    A = x[0]
    if A<=0:
        return np.inf
    else:
        sigmaA = x[1]
        G1 = getE(r,sigmaA,dt,A,K)-E
        G2 = A*getOptionDelta(r,sigmaA,dt,A,K)*(sigmaA/sigmaE)-E
        return G1**2 + G2**2
    
def getVarA(mu,sigma,dt,A,K):
    t1 = np.exp((sigma**2)*dt)
    t2 = (A**2)*np.exp(2*mu*dt)
    return t2*(t1-1)

def getExpA(mu,dt,A):
    return np.multiply(A,np.exp(mu*dt))

def getCovAB(A,B,muA,muB,sigmaA,sigmaB,rhoAB,dt):
    t1 = (muA+muB)*dt
    t2 = sigmaA*sigmaB*rhoAB*dt    
    return A*B*np.exp(t1)*(np.exp(t2)-1)

def getCorAB(sigmaA,sigmaB,rhoAB,dt):
    num = np.exp(rhoAB*sigmaA*sigmaB*dt)-1
    tA = np.sqrt(np.exp((sigmaA**2)*dt)-1)
    tB = np.sqrt(np.exp((sigmaB**2)*dt)-1)
    return np.divide(num,tA*tB)   

def getDefaultCorAB(A,B,muA,muB,sigmaA,sigmaB,rhoAB,dt,KA,KB):
    dA = getDelta(muA,sigmaA,dt,A,KA)
    dB = getDelta(muB,sigmaB,dt,B,KB)     
    pA = norm.cdf(dA)
    pB = norm.cdf(dB)   
    pAB,err = mvn.mvnun(np.array([-100, -100]),np.array([dA, dB]),
                      np.array([0, 0]),np.array([[1,rhoAB],[rhoAB,1]]))   
    return np.divide(pAB-pA*pB,np.sqrt(pA*pB*(1-pA)*(1-pB)))


def getSigmaA(mu,sigma,dt,A,K):
    t1 = math.exp((sigma**2)*dt)
    t2 = A*math.exp(mu*dt)
    return t2*np.sqrt(t1-1)

def computeAssetValueMoments(N,A,mu,sigma,C,dt):
    hatA = np.zeros(N)
    Omega = np.zeros([N,N])
    for n in range(0,N):
        hatA[n] = getExpA(mu[n],dt,A[n])
        for m in range(0,N):
            Omega[n,m] = getCovAB(A[n],A[m],mu[n],mu[m],
                                  sigma[n],sigma[m],C[n,m],dt)
    return hatA,Omega

        