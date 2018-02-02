import numpy as np
import scipy
import math 
from scipy.stats import norm
import scipy.integrate as integrate
import numpy.linalg as anp
from scipy.stats import t as myT
import time
from scipy.stats import mvn

class myPrinter(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() 
    def flush(self) :
        for f in self.files:
            f.flush()    

def printLine(R,myString):
    M = len(R)
    for m in range(0,M):
        if m!=(M-1):
            print(myString % (R[m]) + " & ", end=" ")
        else:
            print(myString % (R[m]) + "\\\\", end="\n")
               
def printMatrix(R,myString):
    N,M = R.shape
    for n in range(0,N):
        for m in range(0,M):
            if m!=(M-1):
                print(myString % (R[n,m]) + " & ", end=" ")
            else:
                print(myString % (R[n,m]) + "\\\\", end="\n")            
  
def simulateDefaultProbabilities(N,pMean):
    p = (pMean/1.5)*np.random.chisquare(1.5,N)
    return p

def simulateTenors(N,T):
    tenor = np.random.uniform(0.25,T,N)
    return tenor
    
def simulateRegions(N,rStart):
    w = np.cumsum(rStart)
    u = np.random.uniform(0,1,N)
    myRegion = np.zeros(N)
    for n in range(0,N):
        if ((u[n]>=0) & (u[n]<=w[0])):
            myRegion[n] = 1
        elif ((u[n]>w[0]) & (u[n]<=w[1])):
            myRegion[n] = 2
        elif (u[n]>w[1]) & (u[n]<=1):
            myRegion[n] = 3        
    return myRegion    
    
def simulateExposures(N,portfolioSize):
    w = np.random.weibull(1,N)
    c = portfolioSize*np.divide(w,np.sum(w)) 
    return c
        
def computeRiskMeasures(M,lossDistribution,alpha):
    expectedLoss = np.mean(lossDistribution)
    unExpectedLoss = np.std(lossDistribution)
    expectedShortfall = np.zeros([len(alpha)])
    var = np.zeros([len(alpha)])
    for n in range(0,len(alpha)):
        expectedShortfall[n] = np.mean(lossDistribution[np.ceil(alpha[n]*(M-1)).astype(int):M-1])        
        var[n] = lossDistribution[np.ceil(alpha[n]*(M-1)).astype(int)]
    return expectedLoss,unExpectedLoss,var,expectedShortfall      

def analyticExpectedShortfall(N,alpha,pmf,c):
    cdf = np.cumsum(pmf)
    numberDefaults = np.linspace(0,N,N+1)
    expectedShortfall = np.zeros(len(alpha))
    for n in range(0,len(alpha)):   
        myAlpha = np.linspace(alpha[n],1,1000)
        nanCheck =  ~np.isnan(pmf)
        loss = c*np.interp(myAlpha,cdf[nanCheck],numberDefaults[nanCheck])
        prob = np.interp(loss,numberDefaults[nanCheck],pmf[nanCheck])
        expectedShortfall[n] = np.dot(loss,prob)/np.sum(prob)
    return expectedShortfall

def generateGamma(a,b,N):
    G1 = np.random.gamma(a,1,N)
    G2 = np.random.gamma(b,1,N)
    Z = np.divide(G1,G1+G2)  
    return Z
    
def computeBeta(a,b):
    Ga = math.gamma(a)
    Gb = math.gamma(b)
    Gab = math.gamma(a+b)
    return (Ga*Gb)/Gab

def logitDensity(z,mu,sigma):
    num = np.log(z/(1-z))-mu    
    den = 2*(sigma**2)
    K = np.reciprocal(sigma*np.sqrt(2*math.pi)*z*(1-z))
    f = K*np.exp(-np.divide(num**2,den))
    return f

def probitDensity(z,mu,sigma):
    pzInverse = np.divide(norm.ppf(z)-mu,sigma)
    num = gaussianDensity(pzInverse,0,1)
    den = sigma*gaussianDensity(norm.ppf(z),0,1)
    f = np.divide(num,den)
    return f

def gaussianDensity(z,mu,sigma):
    num = z-mu    
    den = 2*(sigma**2)
    K = np.reciprocal(np.sqrt(2*math.pi)*sigma)
    f = K*np.exp(-np.divide(num**2,den))
    return f        

#def logGaussianDensity(z,mu,sigma):
#    num = np.log(z)-mu    
#    den = 2*(sigma**2)
#    K = np.reciprocal(np.multiply(np.sqrt(2*math.pi)*sigma,z))
#    f = K*np.exp(-np.divide(num**2,den))
#    return f        

def tDensity(z,mu,sigma,nu):
    g1 = math.gamma((nu+1)/2)
    g2 = math.gamma(nu/2)
    K = np.divide(g1,g2*np.sqrt(nu*math.pi)*sigma)
    power = np.divide((z-mu)**2,nu*(sigma**2))
    f = K*np.power(1+power,-(nu+1)/2)
    return f       

def chi2Density(z,nu):
    g1 = math.gamma(nu/2)
    constant = np.multiply(g1,2**(nu/2))
    term1 = np.power(z,(nu/2)-1)
    term2 = np.exp(-z/2)   
    f = np.reciprocal(constant)*term1*term2
    return f       

def gigDensity(x,myA):
    constant = np.divide(1,2*scipy.special.kn(1, myA))
    f = constant*np.exp(-0.5*myA*(x+1/x))
    return f

def betaDensity(z,a,b):
    term1 = np.power(z,a-1)
    term2 = np.power(1-z,b-1)   
    f = np.reciprocal(computeBeta(a,b))*term1*term2
    return f           

def asrfDensity(x,p,rho):
    a = np.sqrt(np.divide(1-rho,rho))
    b = np.power(np.sqrt(1-rho)*norm.ppf(x)-norm.ppf(p),2)
    c = 0.5*(np.power(norm.ppf(x),2) - b/rho)
    return a*np.exp(c)

def getBC(N,k):
    a = scipy.misc.factorial(N)
    b = scipy.misc.factorial(N-k)
    c = scipy.misc.factorial(k)
    return a/(b*c)

def poissonDensity(lam,k):
    a = np.exp(-lam)
    b = np.power(lam,k)
    c = math.factorial(k)
    pmf = np.divide(a*b,c)
    return pmf

def binomialDensity(N,p,k):
    a = p**k
    b = (1-p)**(N-k)
    pmf =  getBC(N,k)*a*b
    return pmf            

def gammaDensity(z,a,b):
    constant = np.divide(b**a,math.gamma(a))
    t1 = np.exp(-b*z)
    t2 = np.power(z,a-1)
    pdf =  constant*t1*t2
    return pdf            

def logNormalDensity(z,mu,sigma):
    constant = np.divide(1,z*sigma*np.sqrt(2*math.pi))
    num = -(np.log(z)-mu)**2
    den = 2*sigma**2
    pdf =  constant*np.exp(num/den)
    return pdf   

#def weibullDensity1(z,a,b):
#    constant = np.divide(a,b**a)
#    t1 = np.power(z/b,a-1)
#    t2 = np.exp(-(z/b)**a)
#    pdf =  constant*t1*t2
#    return pdf   

def weibullDensity(x,a,b):
     return (a/b)*(x/b)**(a-1)*np.exp(-(x/b)**a)
    