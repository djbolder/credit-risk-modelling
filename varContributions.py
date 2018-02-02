import numpy as np
import math
import scipy
import cmUtilities as util
import binomialPoissonModels as bp
import importlib 
from scipy.stats import norm
import scipy.integrate as nInt
import thresholdModels as th
from scipy.stats import t as myT

importlib.reload(util)
importlib.reload(th)
importlib.reload(bp)

def mcThresholdTDecomposition(N,M,S,p,c,rho,nu,isT,myAlpha):
    contributions = np.zeros([N,S,2])
    var = np.zeros(S)
    es = np.zeros(S)
    K = myT.ppf(p,nu)*np.ones((M,1))        
    for s in range(0,S):
        print("Iteration: %d" % (s+1))
        Y = th.getY(N,M,p,rho,nu,isT)
        myD = 1*np.less(Y,K)     
        myLoss = np.sort(np.dot(myD,c),axis=None)
        el,ul,var[s],es[s]=util.computeRiskMeasures(M,myLoss,np.array([myAlpha]))
        varVector = c*myD[np.dot(myD,c)==var[s],:]
        esVector = c*myD[np.dot(myD,c)>=var[s],:]
        contributions[:,s,0] = np.sum(varVector,0)/varVector.shape[0]
        contributions[:,s,1] = np.sum(esVector,0)/esVector.shape[0]
    return contributions,var,es

def mcThresholdIndDecomposition(N,M,S,p,c,myAlpha):
    contributions = np.zeros([N,S,2])
    var = np.zeros(S)
    es = np.zeros(S)
    for s in range(0,S):
        print("Iteration: %d" % (s+1))
        myLoss,myD = bp.independentBinomialLossDistribution(N,M,p,c,myAlpha,1)
        el,ul,var[s],es[s]=util.computeRiskMeasures(M,myLoss,np.array([myAlpha]))
        varVector = c*myD[np.dot(myD,c)==var[s],:]
        esVector = c*myD[np.dot(myD,c)>=var[s],:]
        contributions[:,s,0] = np.sum(varVector,0)/varVector.shape[0]
        contributions[:,s,1] = np.sum(esVector,0)/esVector.shape[0]
    return contributions,var,es

def computeMGF(t,p,c): 
    return 1-p+p*np.exp(c*t)
    
def computeCGF(t,p,c):
    return np.sum(np.log(computeMGF(t,p,c)))
 
def computeCGF_1(t,p,c):
    num = c*p*np.exp(c*t)
    den = computeMGF(t,p,c)
    return np.sum(np.divide(num,den))

def computeCGF_2(t,p,c,asVector=0):
    num = (1-p)*(c**2)*p*np.exp(c*t)
    den = np.power(computeMGF(t,p,c),2)
    if asVector==1:
        return np.divide(num,den)
    else:
        return np.sum(np.divide(num,den))

def computeCGF_3(t,p,c):
    num1 = (1-p)*(c**3)*p*np.exp(c*t)
    num2 = 2*(1-p)*(c**3)*(p**2)*np.exp(2*c*t)
    den1 = np.power(computeMGF(t,p,c),2)
    den2 = np.power(computeMGF(t,p,c),3)
    return np.sum(np.divide(num1,den1)-np.divide(num2,den2))

def getSaddlePoint(p,c,l,startPoint=0.00025):
    r = scipy.optimize.root(computeCGFRoot,startPoint,args=(p,c,l),method='hybr')
    #if r.success==False:
    #    print("Did not converge!" % (np.min(p),np.max(p),np.mean(p)))
    return r.x    

def step(x):
    return 1 * (x > 0)

def computeCGFRoot(t,p,c,l):
    return computeCGF_1(t,p,c)-l

def getJ(l,p,c,t_l,myOrder):
    K2 = computeCGF_2(t_l,p,c)
    if myOrder==0:
        return np.sqrt(np.divide(1,2*math.pi*K2))
    if myOrder==1:
        t0 = K2*(t_l**2)
        return np.sign(t_l)*np.exp(0.5*t0)*norm.cdf(-np.sqrt(t0))
    if myOrder==2:
        return K2*(getJ(l,p,c,t_l,0)-t_l*getJ(l,p,c,t_l,1))

def saddlePointDensity(l,p,c):
    t_l = getSaddlePoint(p,c,l)
    return np.exp(computeCGF(t_l,p,c)-t_l*l)*getJ(l,p,c,t_l,0)

def saddlePointTailProbability(l,p,c):
    t_l = getSaddlePoint(p,c,l)
    return np.exp(computeCGF(t_l,p,c)-t_l*l)*getJ(l,p,c,t_l,1)
    
def saddlePointShortfallIntegral(l,p,c):
    den = saddlePointTailProbability(l,p,c)
    t_l = getSaddlePoint(p,c,l)
    return l + np.exp(computeCGF(t_l,p,c)-t_l*l)*getJ(l,p,c,t_l,2)/den
    
def identifyVaRInd(x,p,c,myAlpha):
    tpY = saddlePointTailProbability(x,p,c)
    return 1e4*np.power((1-tpY)-myAlpha,2)

def getVaRC(l,p,c):
    t_l = getSaddlePoint(p,c,l)
    return np.divide(c*p*np.exp(c*t_l),computeMGF(t_l,p,c))
    
def getESC(l,p,c):
    varPart = getVaRC(l,p,c)
    myAlpha = saddlePointTailProbability(l,p,c)
    t_l = getSaddlePoint(p,c,l)
    K2 = computeCGF_2(t_l,p,c)
    myW = computeCGF_2(t_l,p,c,1)
    t0 = np.exp(computeCGF(t_l,p,c)-t_l*l)*getJ(l,p,c,t_l,2)
    return varPart + np.divide(t0*np.divide(myW,K2),myAlpha)    

def saddlePointApprox(l,p,c,t_l,myDegree,constant=0):
    if myDegree==1:
        constant = step(-t_l)
    elif myDegree==2:
        constant = step(-t_l)*(np.dot(p,c)-l)
    coefficient = np.exp(computeCGF(t_l,p,c)-t_l*l)
    return  constant + coefficient*getJ(l,p,c,t_l,myDegree)
    
def getPy(p,y,p1,p2,whichModel,v=0):
    if whichModel==0: # Gaussian threshold
        return th.computeP(p,p1,y)
    elif whichModel==1: # beta
        return y*np.ones(len(p))
    elif whichModel==2: # CreditRisk+
        v = p*(1-p1+p1*y)
        return np.maximum(np.minimum(1-np.exp(-v),0.999),0.0001)
    elif whichModel==3: # logit
        return np.reciprocal(1+np.exp(-(p1+p2*y)))
    elif whichModel==4: # probit
        return norm.ppf(p1+p2*y)    
    elif whichModel==5: # Weibull
        return np.maximum(np.minimum(1-np.exp(-y),0.999),0.0001)*np.ones(len(p))
    if whichModel==6: # t threshold
        return th.computeP_t(p,p1,y,v,p2)
    
def getYDensity(y,p1,p2,whichModel,v=0):
    if whichModel==0: 
        return util.gaussianDensity(y,0,1)
    elif whichModel==1:
        return util.betaDensity(y,p1,p2)
    elif whichModel==2:
        return util.gammaDensity(y,p2,p2)    
    elif whichModel==3:
        return util.gaussianDensity(y,0,1)
    elif whichModel==4:
        return util.gaussianDensity(y,0,1)
    elif whichModel==5:
        return util.weibullDensity(y,p1,p2)   
    elif whichModel==6:
        return util.gaussianDensity(y,0,1)*util.chi2Density(v,p2)   

def computeYIntegral(y,l,p,c,p1,p2,whichModel,myDegree):
    pY = getPy(p,y,p1,p2,whichModel)
    d = getYDensity(y,p1,p2,whichModel)
    t_l = getSaddlePoint(pY,c,l)
    return saddlePointApprox(l,pY,c,t_l,myDegree)*d  

def identifyVaR(x,p,c,p1,p2,whichModel,myAlpha):
    tpY = myApprox(x,p,c,p1,p2,whichModel,1)
    return 1e6*np.power((1-tpY)-myAlpha,2)

def getIntegrationBounds(whichModel):
    if whichModel==0:
        lB,uB = -8,8
    elif whichModel==1:
        lB,uB = 0.0001,0.9999
    elif whichModel==2:
        lB,uB = 0.0001,100
    if whichModel==3:
        lB,uB = -8,8
    if whichModel==4:
        lB,uB = -8,8
    elif whichModel==5:
        lB,uB = 0.0001,35
    return lB,uB

def myApprox(l,p,c,p1,p2,whichModel,myDegree,constant=0,den=1):
    lB,uB = getIntegrationBounds(whichModel)
    if myDegree==2:
        constant = l
        den,err = nInt.quad(computeYIntegral,lB,uB,args=(l,p,c,p1,p2,whichModel,1))        
    num,err = nInt.quad(computeYIntegral,lB,uB,args=(l,p,c,p1,p2,whichModel,myDegree))
    return constant + np.divide(num,den) 

def varCNumerator(y,l,myN,p,c,p1,p2,whichModel,v=0):
    pY = getPy(p,y,p1,p2,whichModel,v)
    d = getYDensity(y,p1,p2,whichModel,v)
    t_l = getSaddlePoint(pY,c,l)
    num = pY[myN]*np.exp(c[myN]*t_l)
    den = computeMGF(t_l,pY[myN],c[myN])
    return np.divide(num,den)*saddlePointApprox(l,pY,c,t_l,0)*d

def myVaRCY(l,p,c,p1,p2,whichModel):
    lB,uB = getIntegrationBounds(whichModel)
    den = myApprox(l,p,c,p1,p2,whichModel,0)
    num = np.zeros(len(p))
    for n in range(0,len(p)):
        num[n],err = nInt.quad(varCNumerator,lB,uB,args=(l,n,p,c,p1,p2,whichModel))
    return c*np.divide(num,den)

def esCVaR(y,l,myN,p,c,p1,p2,whichModel,myAlpha,extraTerm=0):
    pY = getPy(p,y,p1,p2,whichModel)
    d = getYDensity(y,p1,p2,whichModel)
    t_l = getSaddlePoint(pY,c,l)
    baseTerm = np.divide(pY[myN]*np.exp(c[myN]*t_l),computeMGF(t_l,pY[myN],c[myN]))
    if t_l<0:
        extraTerm = (pY[myN]-baseTerm)/myAlpha
    return (baseTerm+extraTerm)*d 

def myESCY(l,p,c,p1,p2,whichModel):
    lB,uB = getIntegrationBounds(whichModel)
    myAlpha = myApprox(l,p,c,p1,p2,whichModel,1)
    esC = np.zeros(len(p))
    for n in range(0,len(p)):
        esC[n],err = nInt.quad(integrateAll,lB,uB,args=(l,n,p,c,p1,p2,whichModel))
    return (1/myAlpha)*esC

def integrateAll(y,l,n,p,c,p1,p2,whichModel):
    pY = getPy(p,y,p1,p2,whichModel)
    d = getYDensity(y,p1,p2,whichModel)
    t_l = getSaddlePoint(pY,c,l)
    varPart = getVaRPart(l,n,pY,c,t_l)
    esPart = getESPart(l,n,pY,c,t_l)
    correctPart = getCorrectionPart(l,n,pY,c,t_l)
    return (varPart + esPart + correctPart)*d

def getVaRPart(l,myN,pY,c,t_l):
    baseTerm = np.divide(pY[myN]*np.exp(c[myN]*t_l),computeMGF(t_l,pY[myN],c[myN]))
    return c[myN]*baseTerm*saddlePointApprox(l,pY,c,t_l,1)

def getESPart(l,myN,pY,c,t_l):
    K2 = computeCGF_2(t_l,pY,c)
    myW = computeCGF_2(t_l,pY,c,1)
    t0 = np.exp(computeCGF(t_l,pY,c)-t_l*l)*getJ(l,pY,c,t_l,2)
    return np.divide(t0*np.divide(myW[myN],K2),1)    

def getCorrectionPart(l,myN,pY,c,t_l):
    t0 = c[myN]*pY[myN]
    t1 = computeCGF_1(t_l,pY[myN],c[myN])
    return step(-t_l)*(t0-t1)

def computeYIntegralT(y,v,l,p,c,p1,p2,whichModel,myDegree):
    pY = getPy(p,y,p1,p2,whichModel,v)
    d = getYDensity(y,p1,p2,whichModel,v)
    t_l = getSaddlePoint(pY,c,l)
    return saddlePointApprox(l,pY,c,t_l,myDegree)*d  

def myApproxT(l,p,c,p1,p2,whichModel,myDegree,constant=0,den=1):
    lowerBound = np.maximum(p2-20,0.0001)
    support = [[-8,8],[lowerBound,p2+8]]
    if myDegree==2:
        constant = l
        den,err = nInt.nquad(computeYIntegralT,support,args=(l,p,c,p1,p2,whichModel,1))        
    num,err = nInt.nquad(computeYIntegralT,support,args=(l,p,c,p1,p2,whichModel,myDegree))
    return constant + np.divide(num,den) 

def identifyVaRT(x,p,c,p1,p2,whichModel,myAlpha):
    tpY = myApproxT(x,p,c,p1,p2,whichModel,1)
    return 1e6*np.power((1-tpY)-myAlpha,2)

def varCNumeratorT(y,v,l,myN,p,c,p1,p2,whichModel):
    pY = getPy(p,y,p1,p2,whichModel,v)
    d = getYDensity(y,p1,p2,whichModel,v)
    t_l = getSaddlePoint(pY,c,l)
    num = pY[myN]*np.exp(c[myN]*t_l)
    den = computeMGF(t_l,pY[myN],c[myN])
    return np.divide(num,den)*saddlePointApprox(l,pY,c,t_l,0)*d

def myVaRCYT(l,p,c,p1,p2,whichModel):
    lowerBound = np.maximum(p2-20,0.0001)
    support = [[-8,8],[lowerBound,p2+8]]
    den = myApproxT(l,p,c,p1,p2,whichModel,0)
    num = np.zeros(len(p))
    for n in range(0,len(p)):
        num[n],err = nInt.nquad(varCNumeratorT,support,args=(l,n,p,c,p1,p2,whichModel))
    return c*np.divide(num,den)

def myESCYT(l,p,c,p1,p2,whichModel):
    lowerBound = np.maximum(p2-20,0.0001)
    support = [[-8,8],[lowerBound,p2+8]]
    myAlpha = myApproxT(l,p,c,p1,p2,whichModel,1)
    esC = np.zeros(len(p))
    for n in range(0,len(p)):
        esC[n],err = nInt.nquad(integrateAllT,support,args=(l,n,p,c,p1,p2,whichModel))
    return (1/myAlpha)*esC

def integrateAllT(y,v,l,n,p,c,p1,p2,whichModel):
    pY = getPy(p,y,p1,p2,whichModel,v)
    d = getYDensity(y,p1,p2,whichModel,v)
    t_l = getSaddlePoint(pY,c,l)
    varPart = getVaRPart(l,n,pY,c,t_l)
    esPart = getESPart(l,n,pY,c,t_l)
    correctPart = getCorrectionPart(l,n,pY,c,t_l)
    return (varPart + esPart + correctPart)*d

