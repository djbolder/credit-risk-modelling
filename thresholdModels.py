import numpy as np
import math
import cmUtilities as util
import assetCorrelation as ac
import importlib 
from scipy.stats import norm
import scipy.integrate as nInt
from scipy.stats import t as myT
import numpy.linalg as anp
import scipy

from rpy2.robjects.packages import importr
gig = importr('GIGrvg')

importlib.reload(util)

def getY(N,M,p,rho,nu,isT):
    G = np.transpose(np.tile(np.random.normal(0,1,M),(N,1)))
    e = np.random.normal(0,1,[M,N])
    if isT==1:
        W = np.transpose(np.sqrt(nu/np.tile(np.random.chisquare(nu,M),(N,1))))
        Y = np.multiply(W,math.sqrt(rho)*G + math.sqrt(1-rho)*e)
    else:
        Y = math.sqrt(rho)*G + math.sqrt(1-rho)*e
    return Y   

def getGaussianY(N,M,p,rho):
    G = np.transpose(np.tile(np.random.normal(0,1,M),(N,1)))
    e = np.random.normal(0,1,[M,N])
    Y = math.sqrt(rho)*G + math.sqrt(1-rho)*e
    return Y   

def getTY(N,M,p,rho,nu):
    G = np.transpose(np.tile(np.random.normal(0,1,M),(N,1)))
    e = np.random.normal(0,1,[M,N])
    W = np.transpose(np.sqrt(nu/np.tile(np.random.chisquare(nu,M),(N,1))))
    Y = np.multiply(W,math.sqrt(rho)*G + math.sqrt(1-rho)*e)
    return Y   

def calibrateGaussian(x,myP,targetRho):
    jointDefaultProb = ac.jointDefaultProbability(myP,myP,x)
    defaultCorrelation = np.divide(jointDefaultProb-myP**2,myP*(1-myP))
    return np.abs(defaultCorrelation-targetRho)

def getY2r(N,M,p,myRho,rId,nu,P,isT):
    rhoVector = myRho[rId]
    rhoMatrix = np.tile(rhoVector,(M,1))
    G = np.transpose(np.tile(np.random.normal(0,1,M),(N,1)))
    e = np.random.normal(0,1,[M,N])
    systematic = np.multiply(np.sqrt(rhoMatrix),G)
    idiosyncratic = np.multiply(np.sqrt(1-rhoMatrix),e)
    if isT==1:
        W = np.transpose(np.sqrt(nu/np.tile(np.random.chisquare(nu,M),(N,1))))
        Y = np.multiply(W,systematic + idiosyncratic)
    else:
        Y = systematic + idiosyncratic
    return Y    

def oneFactorGaussianModel(N,M,p,c,rho,alpha):
    Y = getGaussianY(N,M,p,rho)
    K = norm.ppf(p)*np.ones((M,1))        
    lossIndicator = 1*np.less(Y,K)     
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es      

def oneFactorTModel(N,M,p,c,rho,nu,alpha):
    Y = getTY(N,M,p,rho,nu)
    K = myT.ppf(p,nu)*np.ones((M,1))        
    lossIndicator = 1*np.less(Y,K)     
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es      

def oneFactorThresholdModel(N,M,p,c,rho,nu,alpha,isT):
    Y = getY(N,M,p,rho,nu,isT)
    if isT==1:
        K = myT.ppf(p,nu)*np.ones((M,1))        
    else:
        K = norm.ppf(p)*np.ones((M,1))        
    lossIndicator = 1*np.less(Y,K)     
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es      

def oneFactorThresholdLossDistribution(N,M,p,c,rho,nu,alpha,isT):
    Y = getY(N,M,p,rho,nu,isT)
    if isT==1:
        K = myT.ppf(p,nu)*np.ones((M,1))        
    else:
        K = norm.ppf(p)*np.ones((M,1))        
    lossIndicator = 1*np.less(Y,K)     
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    return lossDistribution      
    
def asrfModel(myP,rho,c,alpha):
    myX = np.linspace(0.0001,0.9999,100)
    num = np.sqrt(1-rho)*norm.ppf(myX)-norm.ppf(myP)
    cdf = norm.cdf(num/np.sqrt(rho))
    pdf = util.asrfDensity(myX,myP,rho)
    varAnalytic = np.sum(c)*np.interp(alpha,cdf,myX)
    esAnalytic = asrfExpectedShortfall(alpha,myX,cdf,pdf,c,rho,myP)
    return pdf,cdf,varAnalytic,esAnalytic
    
def asrfExpectedShortfall(alpha,myX,cdf,pdf,c,rho,myP):
    expectedShortfall = np.zeros(len(alpha))
    for n in range(0,len(alpha)):   
        myAlpha = np.linspace(alpha[n],1,1000)
        loss = np.sum(c)*np.interp(myAlpha,cdf,myX)
        prob = np.interp(loss,myX,pdf)
        expectedShortfall[n] = np.dot(loss,prob)/np.sum(prob)
    return expectedShortfall    
        
def asrfMoment(x,p,rho,whichMoment):
    if whichMoment==1:    
        f = x*util.asrfDensity(x,p,rho)
    elif whichMoment==2:
        f = np.power(x,2)*util.asrfDensity(x,p,rho)
    return f
    
def getAsrfMoments(p,rho):
    el,err = nInt.quad(asrfMoment,0,1,args=(p,rho,1)) 
    M2,err = nInt.quad(asrfMoment,0,1,args=(p,rho,2)) 
    ul = np.sqrt(M2 - np.power(el,2))
    return el, ul    

def computeP_t(p,rho,y,w,nu):
    num = np.sqrt(w/nu)*myT.ppf(p,nu)-np.multiply(np.sqrt(rho),y)
    pZ = norm.cdf(np.divide(num,np.sqrt(1-rho)))
    return pZ

def computeP(p,rho,g):
    num = norm.ppf(p)-np.multiply(np.sqrt(rho),g)
    pG = norm.cdf(np.divide(num,np.sqrt(1-rho)))
    return pG

def jointDefaultProbabilityT(p,q,myRho,nu):
    lowerBound = np.maximum(nu-40,2)
    support = [[-10,10],[lowerBound,nu+40]]
    pr,err=nInt.nquad(jointIntegrandT,support,args=(p,q,myRho,nu))
    return pr

def jointDefaultProbabilityG(p,q,myRho):
    pr,err=nInt.quad(jointIntegrandG,-10,10,args=(p,q,myRho))
    return pr

def jointIntegrandT(g,w,p,q,myRho,nu):
    p1 = computeP_t(p,myRho,g,w,nu)
    p2 = computeP_t(q,myRho,g,w,nu)
    density1 = util.gaussianDensity(g,0,1)
    density2 = util.chi2Density(w,nu)    
    f = p1*p2*density1*density2
    return f

def jointIntegrandG(g,p,q,myRho):
    p1 = computeP(p,myRho,g)
    p2 = computeP(q,myRho,g)
    density = util.gaussianDensity(g,0,1)
    f = p1*p2*density
    return f

def bivariateTDensity(x1,x2,rho,nu,d=2):
    Sigma = np.array([[1,rho],[rho,1]])
    myX = np.array([x1,x2])
    t1 = math.gamma((nu+d)/2)
    t2 = math.gamma(nu/2) 
    t3 = np.power(nu*math.pi,d/2)
    t4 = np.sqrt(anp.det(Sigma))
    constant = np.divide(t1,t2*t3*t4)
    t5 = np.dot(np.dot(myX,anp.inv(Sigma)),myX)
    integrand = constant*np.power(1+t5/nu,-(nu+d)/2)
    return integrand

def bivariateTCdf(yy,xx,rho,nu):    
    t_ans, err = nInt.dblquad(bivariateTDensity, -10, xx,
                   lambda x: -10,
                   lambda x: yy,args=(rho,nu))
    return t_ans

def bivariateGDensity(x1,x2,rho):
    S = np.array([[1,rho],[rho,1]])
    t1 = 2*math.pi*np.sqrt(anp.det(S))
    t2 = np.dot(np.dot(np.array([x1,x2]),anp.inv(S)),np.array([x1,x2]))
    return np.divide(1,t1)*np.exp(-t2/2)


def buildAssetCorrelationMatrix(a,b,regionId):
    J = len(b)
    R = np.zeros([J,J])
    for n in range(0,J):
        for m in range(0,J):
            if regionId[n]==regionId[m]:
                R[n,m] = a + (1-a)*np.sqrt(b[n]*b[m])
            else:
                R[n,m] = a
    return R

def buildDefaultCorrelationMatrix(a,b,pMean,regionId,nu):
    J = len(regionId)
    R = buildAssetCorrelationMatrix(a,b,regionId)    
    D = np.zeros([J,J])
    for n in range(0,J):
        p_n = pMean[n]
        for m in range(0,J):
            p_m = pMean[m]
            p_nm = bivariateTCdf(norm.ppf(p_n),norm.ppf(p_m),R[n,m],nu)
            D[n,m] = (p_nm - p_n*p_m)/math.sqrt(p_n*(1-p_n)*p_m*(1-p_m))
    return D

def calibrateOF(x,B,pMean,regionId,nu):
    a = x[0]
    b = np.array([x[1],x[2],x[3]])
    D = buildDefaultCorrelationMatrix(a,b,pMean,regionId,nu)     
    f = anp.norm(D-B,ord='fro')
    return f

def calibrateMFT(B,pMean,regionId,nu):
    myBounds = ((0.001,0.30),(0.001,0.30),(0.001,0.30),
                (0.001,0.30))                            
    M = 100
    xRandom = np.random.uniform(0,0.30,[M,4])
    functionValues = np.zeros(M)
    for m in range(0,M):
        functionValues[m] = calibrateOF(xRandom[m,:],B,pMean,regionId,nu)
    newOF=np.min(functionValues)
    xStart = xRandom[functionValues==newOF]
    xhat = scipy.optimize.minimize(calibrateOF, 
                    xStart, args=(B,pMean,regionId,nu), 
                    method='SLSQP', jac=None, bounds=myBounds)   
    return xhat    

def calibrateT(x,myP,targetRho,nu):
    jointDefaultProb = jointDefaultProbabilityT(myP,myP,x,nu)
    defaultCorrelation = np.divide(jointDefaultProb-myP**2,myP*(1-myP))
    return np.abs(defaultCorrelation-targetRho)

def thresholdCalibrationGridSearch(dGrid,myP,rhoTarget,whichModel,nu=30):
    jointDefaultProb = np.zeros([2,25])
    dEstimate = np.zeros([2,25])
    for n in range(0,len(dGrid)):
        print("Iteration %d" % (n+1))
        if whichModel==1:
            support = [[-8,norm.ppf(myP)],[-8,norm.ppf(myP)]]
            jointDefaultProb[0,n] = jointDefaultProbabilityG(myP,myP,dGrid[n])
            jointDefaultProb[1,n],err = nInt.nquad(bivariateGDensity,support,args=(dGrid[n],2))
        elif whichModel==2:   
            support = [[-8,myT.ppf(myP,nu)],[-8,myT.ppf(myP,nu)]]
            jointDefaultProb[0,n] = jointDefaultProbabilityT(myP,myP,dGrid[n],nu)
            jointDefaultProb[1,n],err = nInt.nquad(bivariateTDensity,support,args=(dGrid[n],nu))
        dEstimate[0,n] = np.divide(jointDefaultProb[0,n]-myP**2,myP*(1-myP))
        dEstimate[1,n] = np.divide(jointDefaultProb[1,n]-myP**2,myP*(1-myP))
    print("The conditonal approach gives %0.2f" % (np.interp(rhoTarget,dEstimate[0,:],dGrid)))
    print("The classic approach gives %0.2f" % (np.interp(rhoTarget,dEstimate[1,:],dGrid)))
    return dEstimate

def tTailDependenceCoefficient(rho,nu):
    a = -np.sqrt(np.divide((nu+1)*(1-rho),1+rho)) 
    tCoefficient = 2*myT.cdf(a,nu+1)    
    return tCoefficient

def tCalibrate(x,myP,rhoTarget,tdTarget):
    if (x[0]<=0) | (x[1]<=0):
        return [100, 100]
    jointDefaultProb = jointDefaultProbabilityT(myP,myP,x[0],x[1])
    rhoValue = np.divide(jointDefaultProb-myP**2,myP*(1-myP))
    tdValue = tTailDependenceCoefficient(x[0],x[1])
    f1 = rhoValue - rhoTarget    
    f2 = tdValue - tdTarget    
    return [f1, f2]

def getMultiFactorY(N,M,p,a,b,rId,nu,isT):
    G = np.transpose(np.tile(np.random.normal(0,1,M),(N,1)))
    regions = np.random.normal(0,1,[M,len(np.unique(rId))]) 
    e = np.random.normal(0,1,[M,N])
    R = regions[:,rId]
    A = np.tile(a*np.ones(N),(M,1))
    B = np.tile(b[rId],(M,1))
    T0 = np.multiply(np.sqrt(A),G)
    T1 = np.sqrt(1-A)
    T2 = np.multiply(np.sqrt(B),R) + np.multiply(np.sqrt(1-B),e)
    if isT==1: 
        W = np.transpose(np.sqrt(nu/np.tile(np.random.chisquare(nu,M),(N,1))))
        return np.multiply(W,T0+np.multiply(T1,T2))
    else: 
        return T0+np.multiply(T1,T2)

def multiFactorThresholdModel(N,M,a,b,rId,p,c,nu,alpha,isT):
    Y = getMultiFactorY(N,M,p,a,b,rId,nu,isT)
    if isT==1:
        K = myT.ppf(p,nu)*np.ones((M,1)) 
    else:
        K = norm.ppf(p)*np.ones((M,1))        
    lossIndicator = 1*np.less(Y,K)     
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es      

def nvmDensity(v,x,myA,whichModel):
    t1 = np.divide(1,np.sqrt(2*math.pi*v))
    t2 = np.exp(-np.divide(x**2,2*v))
    if whichModel==0:
        return t1*t2*util.gammaDensity(v,myA,myA)
    elif whichModel==1:
        return t1*t2*util.gigDensity(v,myA)
    
def nvmPdf(x,myA,whichModel):
    f,err = nInt.quad(nvmDensity,0,50,args=(x,myA,whichModel)) 
    return f

def nvmCdf(x,myA,whichModel):
    F,err = nInt.quad(nvmPdf,-8,x,args=(myA,whichModel)) 
    return F    

def nvmTarget(x,myVal,myA,whichModel):
    F,err = nInt.quad(nvmPdf,-8,x,args=(myA,whichModel)) 
    return F-myVal

def nvmPpf(myVal,myA,whichModel):
    r = scipy.optimize.fsolve(nvmTarget,0,args=(myVal,myA,whichModel))
    return r[0]        

def getNVMY(N,M,rho,myA,whichModel):
    G = np.transpose(np.tile(np.random.normal(0,1,M),(N,1)))
    e = np.random.normal(0,1,[M,N])
    if whichModel==0:
        V = np.transpose(np.sqrt(np.tile(np.random.gamma(myA,1/myA,M),(N,1))))
    elif whichModel==1:
        V = np.transpose(np.sqrt(np.tile(gig.rgig(M,1,myA,myA),(N,1))))
    Y = np.multiply(V,math.sqrt(rho)*G + math.sqrt(1-rho)*e)
    return Y   

def computeP_NVM(p,rho,y,v,myA,invCdf):
    num = np.sqrt(1/v)*invCdf-np.multiply(np.sqrt(rho),y)
    pZ = norm.cdf(np.divide(num,np.sqrt(1-rho)))
    return pZ
  
def nvmKurtosis(rho,myA,whichModel):
    if whichModel==0:
        return 3*(1+myA)/myA
    elif whichModel==1:
        num = scipy.special.kn(3, myA)*scipy.special.kn(1, myA)
        den = scipy.special.kn(2, myA)**2
        return 3*np.divide(num,den)  
       
def ghVariance(myA):
    return  scipy.special.kn(2, myA)/scipy.special.kn(1, myA)   
        
def jointDefaultProbabilityNVM(p,q,myRho,myA,whichModel):    
    invCdf = nvmPpf(p,myA,whichModel)
    support = [[-8,8],[0,100]]
    pr,err=nInt.nquad(jointIntegrandNVM,support,args=(p,q,myRho,myA,invCdf,whichModel))
    return pr

def jointIntegrandNVM(g,v,p,q,myRho,myA,invCdf,whichModel):
    p1 = computeP_NVM(p,myRho,g,v,myA,invCdf)
    p2 = computeP_NVM(q,myRho,g,v,myA,invCdf)
    density1 = util.gaussianDensity(g,0,1)
    if whichModel==0:
        density2 = util.gammaDensity(v,myA,myA)            
    elif whichModel==1:
        density2 = util.gigDensity(v,myA)    
    return p1*p2*density1*density2         
        
def nvmCalibrate(x,myP,rhoTarget,kTarget,whichModel):
    if (x[0]<=0) | (x[1]<=0):
        return [100, 100]
    jointDefaultProb = jointDefaultProbabilityNVM(myP,myP,x[0],x[1],whichModel)
    rhoValue = np.divide(jointDefaultProb-myP**2,myP*(1-myP))
    kValue = nvmKurtosis(x[0],x[1],whichModel)
    f1 = rhoValue - rhoTarget    
    f2 = kValue - kTarget    
    return [f1, f2]

def oneFactorNVMModel(N,M,p,c,rho,myA,alpha,whichModel):
    Y = getNVMY(N,M,rho,myA,whichModel)
    invVector = np.zeros(N)
    for n in range(0,N): 
        invVector[n] = nvmPpf(p[n],myA,whichModel)    
    K = invVector*np.ones((M,1))        
    lossIndicator = 1*np.less(Y,K)     
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es      


         