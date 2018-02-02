import numpy as np
import time
import cmUtilities as util
import importlib 
from scipy.stats import norm

import cmUtilities as util
import binomialPoissonModels as bp
import mixtureModels as mix
import thresholdModels as th
import mertonModel as mert
from scipy.stats import gamma

importlib.reload(util)
importlib.reload(bp)
importlib.reload(mix)
importlib.reload(th)
importlib.reload(mert)

def getQ(p_n,myH=-50):
    return np.divide(1-np.exp(myH*p_n),1-np.exp(myH))

def getRho(p_n,rhoMin=0.12,rhoMax=0.24):
    myQ = getQ(p_n)
    return rhoMin*myQ + rhoMax*(1-myQ)
    
def getMaturitySlope(p_n,p0=0.11582,p1=-0.05478):
    return np.power(p0+p1*np.log(p_n),2)
                                  
def getMaturityAdjustment(tenor,p_n,p0=2.5):
    myB = getMaturitySlope(p_n)
    return np.divide(1+(tenor-p0)*myB,1-(p0-1)*myB)    
                              
def getBaselK(p_n,tenor,alpha):
    g = norm.ppf(1-alpha)
    rhoBasel = getRho(p_n)
    ma = getMaturityAdjustment(tenor,p_n)
    pG = th.computeP(p_n,rhoBasel,g)
    return np.multiply(pG-p_n,ma)
    
def getBaselRiskCapital(p_n,tenor,c,myAlpha):
    myCounter = len(myAlpha)    
    riskCapital = np.zeros(myCounter)
    for n in range(0,myCounter):
        riskCapitalCoefficient=getBaselK(p_n,tenor,myAlpha[n])
        riskCapital[n] = np.dot(c,riskCapitalCoefficient)
    return riskCapital     
    
def runModelSuite(N,M,P,C,alpha,nu,myRho,rhoTarget,tenor,modelType):
    startTime = time.time() 
    if modelType==0: # Binomial (independent-default) model
        el,ul,var,es = bp.independentBinomialSimulation(N,M,P,C,alpha)
        simTime = (time.time() - startTime)
    elif modelType==1: # Gaussian threshold model
        el,ul,var,es = th.oneFactorThresholdModel(N,M,P,C,myRho,nu,alpha,0)
        simTime = (time.time() - startTime)
    elif modelType==2: # Beta-binomial mixture model
        myP = np.mean(P)
        a,b = mix.betaCalibrate(myP,rhoTarget)
        M1 = mix.betaMoment(a,b,1)
        M2 = mix.betaMoment(a,b,2)
        print("a, b parameters are %0.1f and %0.1f." % (a,b))
        print("Targeted: %0.4f and calibrated: %0.4f default probability." % (myP,M1))
        print("Targeted: %0.3f and calibrated: %0.3f default correlation." % (rhoTarget,np.divide(M2-M1**2,M1-M1**2)))
        el,ul,var,es = mix.betaBinomialSimulation(N,M,C,a,b,alpha)
        simTime = (time.time() - startTime)
    elif modelType==3: # t-distributed threshold model
        el,ul,var,es = th.oneFactorThresholdModel(N,M,P,C,myRho,nu,alpha,1)
        simTime = (time.time() - startTime)
    elif modelType==4: # Basel IRB approach
        mAdjustedC = np.multiply(C,getMaturityAdjustment(tenor,P))
        el = np.dot(P,mAdjustedC)       
        var = getBaselRiskCapital(P,tenor,C,alpha)
        ul = np.sum(P*(1-P)*mAdjustedC)
        es = var
        simTime = (time.time() - startTime)    
    elif modelType==5: # Asymptotic Gaussian threshold model (ASRF)
        meanP = np.maximum(0.0009,np.median(P))
        a,b = th.getAsrfMoments(meanP,myRho)
        el = np.sum(C)*a
        ul = np.sum(C)*b
        pdf,cdf,var,es = th.asrfModel(meanP,myRho,C,alpha)
        simTime = (time.time() - startTime)    
    return el,ul,var,es,simTime
    
def fG(myAlpha):
    return norm.pdf(norm.ppf(1-myAlpha))
    
def dfG(myAlpha):
    z = norm.ppf(1-myAlpha)
    return -z*fG(myAlpha)
  
def mu(myAlpha,myP,myC,myRho):
    pn = th.computeP(myP,myRho,norm.ppf(1-myAlpha))
    return np.dot(myC,pn)

def dmu(myAlpha,myP,myC,myRho):
    constant = np.sqrt(np.divide(myRho,1-myRho))
    ratio = norm.ppf(th.computeP(myP,myRho,norm.ppf(1-myAlpha)))
    return -constant*np.dot(myC,norm.pdf(ratio))

def d2mu(myAlpha,myP,myC,myRho):
    constant = np.divide(myRho,1-myRho)
    ratio = norm.ppf(th.computeP(myP,myRho,norm.ppf(1-myAlpha)))
    return -constant*np.dot(ratio*myC,norm.pdf(ratio))

def nu(myAlpha,myP,myC,myRho):
    pn = th.computeP(myP,myRho,norm.ppf(1-myAlpha))
    return np.dot(np.power(myC,2),pn*(1-pn))

def dnu(myAlpha,myP,myC,myRho):
    pn = th.computeP(myP,myRho,norm.ppf(1-myAlpha))
    ratio = norm.ppf(pn)
    constant = np.sqrt(np.divide(myRho,1-myRho))
    return -constant*np.dot(norm.pdf(ratio)*np.power(myC,2),1-2*pn)

def granularityAdjustment(myAlpha,myP,myC,myRho):
    # Get the necessary functions and their derivatives
    f = fG(myAlpha)    
    df = dfG(myAlpha)            
    dg = dmu(myAlpha,myP,myC,myRho) 
    dg2 = d2mu(myAlpha,myP,myC,myRho) 
    h = nu(myAlpha,myP,myC,myRho)    
    dh = dnu(myAlpha,myP,myC,myRho)  
    # Build and return granularity adjustment formula
    t1 = np.reciprocal(dg)
    t2 = np.divide(h*df,f)+dh
    t3 = np.divide(h*dg2,np.power(dg,2))    
    return -0.5*(t1*t2-t3)
  
def getW(myP,myA,myRho,myAlpha):
    num = th.computeP(myP,myRho,norm.ppf(1-myAlpha))-myP
    den = myP*(gamma.ppf(myAlpha,myA,0,1/myA)-1)
    return np.divide(num,den)
  
def getC(gBar,xi):
    gVar = xi*gBar*(1-gBar)
    return np.divide(gBar**2+gVar,gBar)

def getRK(gBar,myA,myW,myP,myAlpha):
    q = gamma.ppf(myAlpha,myA,0,1/myA)
    return gBar*myP*(1-myW+myW*q)
  
def getK(gBar,myA,myW,myP,myAlpha):
    q = gamma.ppf(myAlpha,myA,0,1/myA)
    return gBar*myP*myW*(q-1)

def getDelta(myA,myAlpha):
    q = gamma.ppf(myAlpha,myA,0,1/myA)
    return (q-1)*(myA+np.divide(1-myA,q))

def myLGDRatio(gBar,xi):
    gVar = xi*gBar*(1-gBar)
    return np.divide(gVar,gBar**2)

def granularityAdjustmentCR(myA,myW,gBar,xi,p,c,myAlpha,isApprox=0):
    myDelta = getDelta(myA,myAlpha)
    Cn = getC(gBar,xi)
    RKn = getRK(gBar,myA,myW,p,myAlpha)
    Kn = getK(gBar,myA,myW,p,myAlpha)
    KStar = np.dot(c,Kn)
    myRatio = myLGDRatio(gBar,xi)
    if isApprox==0:
        t1 = myDelta*(Cn*RKn+np.power(RKn,2)*myRatio)
        t2 = Kn*(Cn+2*RKn*myRatio)
    else:
        t1 = myDelta*Cn*RKn
        t2 = Kn*Cn  
    return np.dot(np.power(c,2),t1-t2)/(2*KStar)  
  
def gaContribution(myA,myW,gBar,xi,myP,c,myC,myAlpha,isApprox=0):
    myDelta = getDelta(myA,myAlpha)
    Cn = getC(gBar,xi)
    RKn = getRK(gBar,myA,myW,myP,myAlpha)
    Kn = getK(gBar,myA,myW,myP,myAlpha)
    KStar = np.sum(c*Kn)
    myRatio = myLGDRatio(gBar,xi)
    if isApprox==0:
        t1 = myDelta*(Cn*RKn+np.power(RKn,2)*myRatio)
        t2 = Kn*(Cn+2*RKn*myRatio)
    else:
        t1 = myDelta*Cn*RKn
        t2 = Kn*Cn  
    return np.dot(np.power(myC,2),t1-t2)/(2*KStar)      

