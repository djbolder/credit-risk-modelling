import numpy as np
import cmUtilities as util
import importlib 
from scipy.stats import norm
from scipy.stats import t as myT
import scipy

import thresholdModels as th
import varContributions as vc

importlib.reload(th)
importlib.reload(vc)
importlib.reload(util)

def myFunction(x):
    return 7*np.sin(5*x) + 5*np.power(x,3) + 3*np.exp(x/2)

def mcIntegration(M):
    U = np.random.uniform(0,1,M)
    return np.mean(myFunction(U))

def naiveNumericalIntegration(K,a,b):
    myGrid = np.linspace(a,b,K)
    myIntegral = 0
    for n in range(0,K-1):
        dx = myGrid[n+1]-myGrid[n]
        myIntegral += myFunction(myGrid[n+1])*dx
    return myIntegral    

def getQ(theta,c,p):
    return np.divide(p*np.exp(c*theta),vc.computeMGF(theta,p,c))
    
def meanShiftOF(mu,c,p,l,myRho):
    pZ = th.computeP(p,myRho,mu)
    theta = vc.getSaddlePoint(pZ,c,l,0.0)
    f_l = -theta*l + vc.computeCGF(theta,pZ,c)
    return -(f_l - 0.5*np.dot(mu,mu))    

def getOptimalMeanShift(c,p,l,myRho):
    r = scipy.optimize.minimize(meanShiftOF,-1.0,args=(c,p,l,myRho), 
                            method='SLSQP',jac=None,bounds=[(-4.0,4.0)]) 
    return r.x

def isIndDefault(N,M,p,c,l):
    U = np.random.uniform(0,1,[M,N])
    theta = vc.getSaddlePoint(p,c,l,0.0)
    qZ = getQ(theta,c,p)
    cgf = vc.computeCGF(theta,p,c)
    I = np.transpose(1*np.less(U,qZ))
    L = np.dot(c,I)
    rn = computeRND(theta,L,cgf)        
    tailProb = np.mean(np.multiply(L>l,rn)) 
    eShortfall =  np.mean(np.multiply(L*(L>l),rn))/tailProb        
    return tailProb,eShortfall    

def isThreshold(N,M,p,c,l,myRho,nu,shiftMean,isT,invVector=0):
    mu = 0.0
    gamma = 0.0
    if shiftMean==1:
        mu = getOptimalMeanShift(c,p,l,myRho)
    theta = np.zeros(M)
    cgf = np.zeros(M)
    qZ = np.zeros([M,N])
    G = np.transpose(np.tile(np.random.normal(mu,1,M),(N,1)))
    e = np.random.normal(0,1,[M,N])
    if isT==1:
        gamma = -2
        W = np.random.chisquare(nu,M)
        myV = W/(1-2*gamma)
        V = np.transpose(np.sqrt(np.tile(myV,(N,1))/nu))
        num = (1/V)*myT.ppf(p,nu)*np.ones((M,1))-np.multiply(np.sqrt(myRho),G)
        pZ = norm.cdf(np.divide(num,np.sqrt(1-myRho)))
    elif isT==2:
        V = np.transpose(np.sqrt(np.tile(np.random.gamma(nu,1/nu,M),(N,1))))
        num = (1/V)*invVector*np.ones((M,1))-np.multiply(np.sqrt(myRho),G)
        pZ = norm.cdf(np.divide(num,np.sqrt(1-myRho)))
    else:
        pZ = th.computeP(p,myRho,G)
    for n in range(0,M):
        theta[n] = vc.getSaddlePoint(pZ[n,:],c,l,0.0)
        qZ[n,:] = getQ(theta[n],c,pZ[n,:])
        cgf[n] = vc.computeCGF(theta[n],pZ[n,:],c)
    I = np.transpose(1*np.less(e,norm.ppf(qZ)))
    L = np.dot(c,I)
    if isT==1:
        rnChi = np.exp(-gamma*myV-(nu/2)*np.log(1-2*gamma))
    else:
        rnChi = np.ones(M)
    if shiftMean==1:
        rn = computeRND(theta,L,cgf)*np.exp(-mu*G[:,0]+0.5*(mu**2))*rnChi
    else:
        rn = computeRND(theta,L,cgf)*rnChi
    tailProb = np.mean(np.multiply(L>l,rn)) 
    eShortfall =  np.mean(np.multiply(L*(L>l),rn))/tailProb        
    return tailProb,eShortfall    
              
def isMixture(N,M,p,c,l,p1,p2):
    theta = np.zeros(M)
    cgf = np.zeros(M)
    qS = np.zeros([M,N])
    S = np.random.gamma(p1, 1/p1, [M]) 
    wS =  np.transpose(np.tile(1-p2 + p2*S,[N,1]))
    pS = np.tile(p,[M,1])*wS
    for n in range(0,M):
        theta[n] = vc.getSaddlePoint(pS[n,:],c,l,-0.2)
        qS[n,:] = getQ(theta[n],c,pS[n,:])
        cgf[n] = vc.computeCGF(theta[n],pS[n,:],c)
    I = np.transpose(1*np.greater_equal(np.random.poisson(qS,[M,N]),1))
    L = np.dot(c,I)
    rn = computeRND(theta,L,cgf)        
    tailProb = np.mean(np.multiply(L>l,rn)) 
    eShortfall =  np.mean(np.multiply(L*(L>l),rn))/tailProb        
    return tailProb,eShortfall    

def computeRND(theta,L,cgf):
    return np.exp(-np.multiply(theta,L)+cgf)

def isThresholdSimple(N,M,p,c,l,myRho):
    mu = getOptimalMeanShift(c,p,l,myRho)
    theta = np.zeros(M)
    cgf = np.zeros(M)
    qZ = np.zeros([M,N])
    e = np.random.normal(0,1,[M,N])
    G = np.transpose(np.tile(np.random.normal(mu,1,M),(N,1)))
    num = (norm.ppf(p)*np.ones((M,1)))-np.sqrt(myRho)*G
    pZ = norm.cdf(np.divide(num,np.sqrt(1-myRho)))
    for n in range(0,M):
        theta[n] = vc.getSaddlePoint(pZ[n,:],c,l,0.0)
        qZ[n,:] = getQ(theta[n],c,pZ[n,:])
        cgf[n] = vc.computeCGF(theta[n],pZ[n,:],c)
    I = np.transpose(1*np.less(e,norm.ppf(qZ)))
    L = np.dot(c,I)
    rn = np.exp(-mu*G[:,0]+0.5*(mu**2))*computeRND(theta,L,cgf)
    tailProb = np.mean(np.multiply(L>l,rn)) 
    eShortfall =  np.mean(np.multiply(L*(L>l),rn))/tailProb        
    return tailProb,eShortfall    

def isThresholdT(N,M,p,c,l,myRho,nu,cm=0):
    myShift = (1-2*cm)
    mu = getOptimalMeanShift(c,p,l,myRho)
    W = np.random.chisquare(nu,M)
    myV = W/myShift
    theta = np.zeros(M)
    cgf = np.zeros(M)
    qZ = np.zeros([M,N])
    V = np.transpose(np.sqrt(np.tile(myV,(N,1))/nu))
    e = np.random.normal(0,1,[M,N])
    G = np.transpose(np.tile(np.random.normal(mu,1,M),(N,1)))
    num = V*(myT.ppf(p,nu)*np.ones((M,1)))-np.sqrt(myRho)*G
    pZ = norm.cdf(np.divide(num,np.sqrt(1-myRho)))
    for n in range(0,M):
        theta[n] = vc.getSaddlePoint(pZ[n,:],c,l,0.0)
        qZ[n,:] = getQ(theta[n],c,pZ[n,:])
        cgf[n] = vc.computeCGF(theta[n],pZ[n,:],c)
    I = np.transpose(1*np.less(e,norm.ppf(qZ)))
    L = np.dot(c,I)
    rnChi = np.exp(-cm*myV-(nu/2)*np.log(myShift))
    rnMu=np.exp(-mu*G[:,0]+0.5*(mu**2))
    rnTwist = computeRND(theta,L,cgf)
    rn = rnChi*rnMu*rnTwist
    tailProb = np.mean(np.multiply(L>l,rn)) 
    eShortfall =  np.mean(np.multiply(L*(L>l),rn))/tailProb        
    return tailProb,eShortfall    

