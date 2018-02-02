import numpy as np
import cmUtilities as util

def independentBinomialSimulation(N,M,p,c,alpha):
    lossDistribution = independentBinomialLossDistribution(N,M,p,c,alpha)
    el,ul,var,es = util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es

def independentPoissonSimulation(N,M,p,c,alpha):
    lossDistribution = independentPoissonLossDistribution(N,M,p,c,alpha)
    el,ul,var,es = util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es

def independentBinomialLossDistribution(N,M,p,c,alpha,fullOutput=0):
    U = np.random.uniform(0,1,[M,N])
    defaultIndicator = 1*np.less(U,p)
    lossDistribution = np.sort(np.dot(defaultIndicator,c),axis=None)
    if fullOutput==0:
        return lossDistribution
    else:
        return lossDistribution,defaultIndicator

def independentPoissonLossDistribution(N,M,p,c,alpha):
    lam = -np.log(1-p)
    H = np.random.poisson(lam,[M,N])
    defaultIndicator = 1*np.greater_equal(H,1)
    lossDistribution = np.sort(np.dot(defaultIndicator,c),axis=None)
    return lossDistribution

def independentBinomialAnalytic(N,p,c,alpha):
    pmfBinomial = np.zeros(N+1)
    for k in range(0,N+1):
        pmfBinomial[k] =  util.getBC(N,k)*(p**k)*((1-p)**(N-k))
    cdfBinomial = np.cumsum(pmfBinomial)
    varAnalytic = c*np.interp(alpha,cdfBinomial,np.linspace(0,N,N+1))
    esAnalytic = util.analyticExpectedShortfall(N,alpha,pmfBinomial,c)
    return pmfBinomial,cdfBinomial,varAnalytic,esAnalytic

def independentPoissonAnalytic(N,c,myLam,alpha):
    pmfPoisson = np.zeros(N+1)
    for k in range(0,N+1):
        pmfPoisson[k] =  util.poissonDensity(myLam,k)
    cdfPoisson = np.cumsum(pmfPoisson)
    varAnalytic = c*np.interp(alpha,cdfPoisson,np.linspace(0,N,N+1))
    esAnalytic = util.analyticExpectedShortfall(N,alpha,pmfPoisson,c)
    return pmfPoisson,cdfPoisson,varAnalytic,esAnalytic

