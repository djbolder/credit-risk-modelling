import numpy as np
import math
import cmUtilities as util
import numpy.linalg as anp
import importlib 
from scipy.stats import norm
import scipy.integrate as nInt
import scipy

importlib.reload(util)

# -----------------------
# Beta-binomial model functions
# -----------------------

def betaMoment(a,b,momentNumber):
    num = math.gamma(a+momentNumber)*math.gamma(a+b)
    den = math.gamma(a+b+momentNumber)*math.gamma(a)
    f = np.divide(num,den)
    return f

def betaCalibrate(pTarget,rhoTarget):
    A = np.array([[rhoTarget,rhoTarget],[pTarget-1,pTarget]])
    a,b = np.dot(anp.inv(A),np.array([1-rhoTarget,0]))
    return a,b

def betaBinomialAnalytic(N,c,a,b,alpha):
    pmfBeta = np.zeros(N+1)
    den = util.computeBeta(a,b)
    for k in range(0,N+1):
        pmfBeta[k] =  util.getBC(N,k)*util.computeBeta(a+k,b+N-k)/den  
    cdfBeta = np.cumsum(pmfBeta)
    varAnalytic = c*np.interp(alpha,cdfBeta,np.linspace(0,N,N+1))
    esAnalytic = util.analyticExpectedShortfall(N,alpha,pmfBeta,c)
    return pmfBeta,cdfBeta,varAnalytic,esAnalytic
    
def betaBinomialSimulation(N,M,c,a,b,alpha):
    Z = util.generateGamma(a,b,M)
    pZ = np.transpose(np.tile(Z,(N,1)))
    U = np.random.uniform(0,1,[M,N])
    lossIndicator = 1*np.less(U,pZ)
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es  

def betaLossDistribution(N,M,c,a,b,alpha):
    Z = util.generateGamma(a,b,M)
    pZ = np.transpose(np.tile(Z,(N,1)))
    U = np.random.uniform(0,1,[M,N])
    lossIndicator = 1*np.less(U,pZ)
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    return lossDistribution  

# -----------------------
# Logit-Probit model functions
# -----------------------

def logitProbitMoment(z,mu,sigma,momentNumber,isLogit):
    if isLogit==1:
        v0 = (np.reciprocal(1+np.exp(-(mu+sigma*z))))**momentNumber
    else:
        v0 = (norm.cdf(mu+sigma*z))**momentNumber    
    density = util.gaussianDensity(z,0,1)
    f = v0*density
    return f  

def logitProbitCalibrate(x,pTarget,rhoTarget,isLogit):
    if x[1]<=0:
        return [100, 100]
    M1,err = nInt.quad(logitProbitMoment,-8,8,args=(x[0],x[1],1,isLogit)) 
    M2,err = nInt.quad(logitProbitMoment,-8,8,args=(x[0],x[1],2,isLogit)) 
    f1 = pTarget - M1    
    f2 = rhoTarget*(M1 - (M1**2)) - (M2 - (M1**2))    
    return [f1, f2]

def logitProbitBinomialAnalytic(N,c,mu,sigma,alpha,isLogit):
    pmf = np.zeros(N+1)
    for k in range(0,N+1):
        pmf[k],err=nInt.quad(logitProbitMixtureFunction,0,1,
                                          args=(N,k,mu,sigma,isLogit)) 
    cdf = np.cumsum(pmf)
    varAnalytic = c*np.interp(alpha,cdf,np.linspace(0,N,N+1))
    esAnalytic = util.analyticExpectedShortfall(N,alpha,pmf,c)
    return pmf,cdf,varAnalytic,esAnalytic

def logitProbitMixtureFunction(z,N,k,mu,sigma,isLogit):
    if isLogit==1:
        density = util.logitDensity(z,mu,sigma)
    else:
        density = util.probitDensity(z,mu,sigma)
    probTerm = util.getBC(N,k)*(z**k)*((1-z)**(N-k))
    f = probTerm*density
    return f    

def logitProbitBinomialSimulation(N,M,c,mu,sigma,alpha,isLogit):
    Z = np.random.normal(0,1,M)
    if isLogit==1:
        p = np.reciprocal(1+np.exp(-(mu+sigma*Z)))
    else:
        p = norm.cdf(mu+sigma*Z)
    pZ = np.transpose(np.tile(p,(N,1)))
    U = np.random.uniform(0,1,[M,N])
    lossIndicator = 1*np.less(U,pZ)
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es    

# -----------------------
# Poisson-Gamma model functions
# -----------------------

def poissonGammaMoment(a,b,momentNumber):
    if momentNumber==1:
        myMoment = 1-np.power(np.divide(b,b+1),a)
    if momentNumber==2:
       v1 = np.power(np.divide(b,b+1),a)
       v2 = np.power(np.divide(b,b+2),a)
       myMoment = 1 - 2*v1 + v2
    return myMoment 

def poissonGammaCalibrate(x,pTarget,rhoTarget):
    if x[1]<=0:
        return [100, 100]
    M1 = poissonGammaMoment(x[0],x[1],1)
    M2 = poissonGammaMoment(x[0],x[1],2) 
    f1 = pTarget - M1    
    f2 = rhoTarget*(M1 - (M1**2)) - (M2 - (M1**2))    
    return [f1, f2]

def poissonGammaAnalytic(N,c,a,b,alpha):
    pmfPoisson = np.zeros(N+1)
    q = np.divide(b,b+1)
    den = math.gamma(a)
    for k in range(0,N+1):
        num = np.divide(math.gamma(a+k),scipy.misc.factorial(k))
        pmfPoisson[k] = np.divide(num,den)*np.power(q,a)*np.power(1-q,k)  
    cdfPoisson = np.cumsum(pmfPoisson)
    varAnalytic = c*np.interp(alpha,cdfPoisson,np.linspace(0,N,N+1))
    esAnalytic = util.analyticExpectedShortfall(N,alpha,pmfPoisson,c)
    return pmfPoisson,cdfPoisson,varAnalytic,esAnalytic

def poissonGammaSimulation(N,M,c,a,b,alpha):
    lam = np.random.gamma(a,1/b,M)
    H = np.zeros([M,N])
    for m in range(0,M):
        H[m,:] = np.random.poisson(lam[m],[N])
    lossIndicator = 1*np.greater_equal(H,1)
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es  
 
# -----------------------
# Poisson-Mixture model functions
# -----------------------

def poissonMixtureMoment(s,a,b,momentNumber,whichModel):
    v0 = 1-np.exp(-s)
    if whichModel==0:
        myDensity = util.logNormalDensity(s,a,b)
    else:
        myDensity = util.weibullDensity(s,a,b)
    f = np.power(v0,momentNumber)*myDensity
    return f  

def poissonTransformedMixtureMoment(s,a,b,momentNumber,whichModel):
    vy = -np.log(1-s)
    jacobian = np.divide(1,1-s)
    if whichModel==0:
        psDensity = util.logNormalDensity(vy,a,b)*jacobian
    elif whichModel==1:
        psDensity = util.weibullDensity(vy,a,b)*jacobian
    myMoment = (s**momentNumber)*psDensity
    return myMoment  
      
def poissonMixtureAnalytic(N,myC,a,b,alpha,whichModel):
    pmfMixture = np.zeros(N+1)
    for k in range(0,N+1):
        pmfMixture[k],err = nInt.quad(poissonMixtureIntegral,0,k+1,
                                      args=(k,a,b,N,whichModel)) 
    cdfMixture = np.cumsum(pmfMixture)
    varAnalytic = myC*np.interp(alpha,cdfMixture,np.linspace(0,N,N+1))
    esAnalytic = util.analyticExpectedShortfall(N,alpha,pmfMixture,myC)
    return pmfMixture,cdfMixture,varAnalytic,esAnalytic

def poissonMixtureIntegral(s,k,a,b,N,whichModel):
    pDensity = util.poissonDensity(N*s,k)
    if whichModel==0:
        mixDensity = util.logNormalDensity(s,a,b)
    elif whichModel==1:
        mixDensity = util.weibullDensity(s,a,b)        
    f = pDensity*mixDensity
    return f

def poissonMixtureCalibrate(x,pTarget,rhoTarget,whichModel):
    if x[1]<=0:
        return [100, 100]
    M1,err = nInt.quad(poissonMixtureMoment,0.0001,0.9999,args=(x[0],x[1],1,whichModel)) 
    M2,err = nInt.quad(poissonMixtureMoment,0.0001,0.9999,args=(x[0],x[1],2,whichModel)) 
    f1 = pTarget - M1    
    f2 = rhoTarget*(M1 - (M1**2)) - (M2 - (M1**2))    
    return [f1, f2]

def poissonMixtureCalibrate1(x,pTarget,rhoTarget,whichModel):
    if x[1]<=0:
        return [100, 100]
    M1,err = nInt.quad(poissonTransformedMixtureMoment,0.0001,0.9999,args=(x[0],x[1],1,whichModel)) 
    M2,err = nInt.quad(poissonTransformedMixtureMoment,0.0001,0.9999,args=(x[0],x[1],2,whichModel)) 
    f1 = pTarget - M1    
    f2 = rhoTarget*(M1 - (M1**2)) - (M2 - (M1**2))    
    return [f1, f2]

def poissonMixtureSimulation(N,M,c,a,b,alpha,whichModel):
    if whichModel==0:
        lam = np.random.lognormal(a,b,M)
    elif whichModel==1:
         lam = b*np.random.weibull(a,M)
    H = np.zeros([M,N])
    for m in range(0,M):
        H[m,:] = np.random.poisson(lam[m],[N])
    lossIndicator = 1*np.greater_equal(H,1)
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es  
      
# -----------------------
# CreditRisk+ model functions
# -----------------------

def crPlusOneFactor(N,M,w,p,c,v,alpha):
    S = np.random.gamma(v, 1/v, [M]) 
    wS =  np.transpose(np.tile(1-w + w*S,[N,1]))
    pS = np.tile(p,[M,1])*wS
    H = np.random.poisson(pS,[M,N])
    lossIndicator = 1*np.greater_equal(H,1)
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es          

def calibrateBeta(myG,myXi):
    a1 = np.divide(myG*(1-myXi),myXi)
    b1 = np.divide((1-myG)*(1-myXi),myXi)
    return a1, b1

def crPlusOneFactorLGD(N,M,w,p,c,v,gBar,xi,alpha):
    a1,b1 = calibrateBeta(gBar,xi)
    LGD = np.random.beta(a1,b1,[M,N])
    S = np.random.gamma(v, 1/v, [M]) 
    wS =  np.transpose(np.tile(1-w + w*S,[N,1]))
    pS = np.tile(p,[M,1])*wS
    H = np.random.poisson(pS,[M,N])
    lossIndicator = 1*np.greater_equal(H,1)
    lossDistribution = np.sort(np.dot(LGD*lossIndicator,c),axis=None)
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es   

def crPlusMultifactor(N,M,wMat,p,c,aVec,alpha,rId):
    K = len(aVec)
    S = np.zeros([M,K])
    for k in range(0,K):        
        S[:,k] = np.random.gamma(aVec[k], 1/aVec[k], [M]) 
    W = wMat[rId,:]
    # Could replace tile with np.kron(W[:,0],np.ones([1,M])), but it's slow
    wS =  np.tile(W[:,0],[M,1]) + np.dot(S,np.transpose(W[:,1:]))
    pS = np.tile(p,[M,1])*wS
    H = np.random.poisson(pS,[M,N])
    lossIndicator = 1*np.greater_equal(H,1)
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es          
      
# -----------------------
# Miscellaneous functions
# -----------------------

def calibrateVerify(a,b,targetP,targetRho,model):
    if (model==0) | (model==4): # Binomial and Poisson
        M1 = targetP
        M2 = targetP**2
    if model==1: # Beta-binomial
        M1 = betaMoment(a,b,1)
        M2 = betaMoment(a,b,2)
    elif model==2: # Logit
        M1,err = nInt.quad(logitProbitMoment,-8,8,args=(a,b,1,1)) 
        M2,err = nInt.quad(logitProbitMoment,-8,8,args=(a,b,2,1)) 
    elif model==3: # Probit
        M1,err = nInt.quad(logitProbitMoment,-8,8,args=(a,b,1,0)) 
        M2,err = nInt.quad(logitProbitMoment,-8,8,args=(a,b,2,0)) 
    elif model==5: # Basic Poisson-gamma
        M1 = poissonGammaMoment(a,b,1)
        M2 = poissonGammaMoment(a,b,2)
    elif model==6: # Poisson log-normal
        M1,err = nInt.quad(poissonMixtureMoment,0,1,args=(a,b,1,0)) 
        M2,err = nInt.quad(poissonMixtureMoment,0,1,args=(a,b,2,0)) 
    elif model==7: # Poisson Weibull
        M1,err = nInt.quad(poissonMixtureMoment,0,1,args=(a,b,1,1)) 
        M2,err = nInt.quad(poissonMixtureMoment,0,1,args=(a,b,2,1))         
    print("Target default probability is %0.3f and target default correlation is %0.3f" % (targetP,targetRho))
    print("Calibrated default probability is %0.3f and calibrated default correlation is %0.3f" % (M1,defCorr(M1,M2)))
    return M1,M2

def defCorr(M1,M2):
    return np.divide(M2-M1**2,M1-M1**2)
      