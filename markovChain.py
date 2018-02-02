import numpy as np
import math
import cmUtilities as util
import numpy.linalg as anp
import importlib 
from scipy.stats import norm
import scipy.integrate as nInt
from scipy.stats import t as myT
import scipy.linalg as asp
from scipy.optimize import approx_fprime

importlib.reload(util)

def inventedTransitionMatrix():
    P = np.array([[0.960,0.029,0.010,0.001],
              [0.100,0.775,0.120,0.005],
              [0.120,0.220,0.650,0.010],
              [0.000,0.000,0.000,1.00]])
    return P

def initializeCounterparties(N,wStart):
    startRating = np.zeros(N)
    w = np.cumsum(wStart)
    u = np.random.uniform(0,1,N)
    for n in range(0,N):
        if ((u[n]>0) & (u[n]<=w[0])):
            startRating[n] = 1
        elif ((u[n]>w[0]) & (u[n]<=w[1])):
            startRating[n] = 2
        elif ((u[n]>w[1]) & (u[n]<=w[2])):
            startRating[n] = 3        
        elif u[n]>w[2]:
            startRating[n] = 4
    return startRating

def simulateRatingData(N,T,P,wStart):
    tStart = initializeCounterparties(N,wStart)
    D = np.zeros([N,T])
    for n in range(0,N):
        D[n,0] = tStart[n]
        for t in range(1,T):
            D[n,t] = transitionStep(D[n,t-1].astype(int),P)
    return D
    
def transitionStep(currentState,P):
    myP = np.cumsum(P[currentState-1,:])
    u = np.random.uniform(0,1)
    if ((u>0) & (u<=myP[0])):
        return 1
    elif ((u>myP[0]) & (u<=myP[1])):
        return 2
    elif ((u>myP[1]) & (u<=myP[2])):
        return 3        
    elif ((u>myP[2]) & (u<=myP[3])):
        return 4    
    
def getTransitionCount(K,T,N,data):
    n_ij = np.zeros([K,K])
    # Count the n_ij or n_{to,from} 
    # => the number of j's followed by i's.
    for i in range(0,K): # to
        for j in range(0,K): # from
            for n in range(0,N): # obligor
                for tau in range(1,T): # time
                    if ((data[n,tau]==j+1) & (data[n,tau-1]==i+1)):
                        #if data[n,tau]==K:
                            # Absorbing-state (no exit)
                        n_ij[i,j] += 1
                            #break
                        #else:
                        #    n_ij[i,j] += 1   
    #if np.sum(n_ij[-1,0])==0:
    #    n_ij[-1,-1]=1
    return n_ij    

def estimateCohortTransitionMatrix(K,n_ij,myPeriod):
    H = np.zeros([K,K])
    for i in range(0,K):
        for j in range(0,K):
            H[i,j] = np.divide(n_ij[i,j],np.sum(n_ij[i,:]))
    H[-1,:] = np.zeros(K)
    H[-1,-1] = 1        
    return anp.matrix_power(H,myPeriod)

def estimateHazardRateTransitionMatrix(K,n_ij,D,myPeriod):
    # Uses the hazard-rate technique
    H = np.zeros([K,K])
    # Construct the generator matrix
    for i in range(0,K):
        for j in range(0,K):
            if i==j:
                continue
            H[i,j] = np.divide(n_ij[i,j],np.sum(D==i+1))
    for i in range(0,K):
        H[i,i]=-(np.sum(H[i,:]-H[i,i]))
    M = asp.expm(myPeriod*H)
    M[-1,:] = np.zeros(K)
    M[-1,-1] = 1   
    return M
   
def bootstrapDistribution(K,N,T,PEstimate,wStart,S):
    PBootstrap = np.zeros([K,K,S])
    for s in range(0,S):
        if np.remainder(s+1,500)==0:
            print("Run iteration %d" % (s+1))
        DBootstrap = simulateRatingData(N,T,PEstimate,wStart)
        NBootstrap = getTransitionCount(K,T,N,DBootstrap)
        PBootstrap[:,:,s] = estimateCohortTransitionMatrix(K,NBootstrap,1)
    return PBootstrap
    

def estimateTransitionMatrix(K,T,N,data,myPeriod,whichModel):
    n_ij = getTransitionCount(K,T,N,data)
    if whichModel==0:
        M = estimateCohortTransitionMatrix(K,n_ij,myPeriod)
    elif whichModel==1:
        M = estimateHazardRateTransitionMatrix(K,n_ij,data,myPeriod)
    return M    

def tpLikelihood(x,M): 
    mVector = np.reshape(M, len(M)**2)
    L = 0
    for i in range(0,len(x)):
        if x[i]<=0:
            pass
        else:
            L += mVector[i]*np.log(x[i])
    return L   

def hessian(x0,epsilon,M):
    # The first derivative
    f1 = approx_fprime(x0,tpLikelihood,epsilon,M) 
    n = x0.shape[0]
    hessian = np.zeros([n,n])
    xx = x0
    for j in range(0,n):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + epsilon # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        f2 = approx_fprime(xx, tpLikelihood,epsilon,M) 
        hessian[:, j] = (f2 - f1)/epsilon # scale...
        xx[j] = xx0 # Restore initial value of x0        
    return hessian  
    
    
def printSEConfidenceInterval(myP,se,T):
    K = myP.shape[0]
    coeff = myT.ppf(1-0.05/2,T-1)
    for i in range(0,K):
        for j in range(0,K):
            low = np.maximum(myP[i,j]-coeff*se[i,j],0)
            up = np.minimum(myP[i,j]+coeff*se[i,j],1)
            if j!=(K-1):
                print("[%0.2f, %0.2f]" % (low,up) + " & ", end=" ")
            else:
                print("[%0.2f, %0.2f]" % (low,up) + "\\\\", end="\n")

def printQuantileConfidenceInterval(myP,S):
    K = myP.shape[0]
    for i in range(0,K):
        for j in range(0,K):
            sortP = np.sort(myP[i,j,:],axis=None)
            low = sortP[np.ceil(0.025*(S-1)).astype(int)]
            up = sortP[np.ceil(0.975*(S-1)).astype(int)]
            if j!=(K-1):
                print("[%0.2f, %0.2f]" % (low,up) + " & ", end=" ")
            else:
                print("[%0.2f, %0.2f]" % (low,up) + "\\\\", end="\n")

def bLikelihood(N,k,pDomain):
    L = util.getBC(N,k)*(pDomain**k)*((1-pDomain)**(N-k))
    return L/np.abs(np.max(L))

def bLogLikelihood(N,k,pDomain):
    ell = k*np.log(pDomain) + (N-k)*np.log(1-pDomain)
    return ell/np.abs(np.max(ell))
    
def bScore(N,k,pDomain):
    s = np.divide(k,pDomain)-np.divide(N-k,1-pDomain)
    return s
    
def bFisherInformation(N,k,pDomain):
    a = np.divide(k,np.power(pDomain,2))
    b = np.divide(N-k,np.power(1-pDomain,2))
    I = a + b 
    return I                
  
def getNSSpotCurve(l,s,c,t,v=0.10):
    den = v*t
    level = 1
    slope = np.divide(1-np.exp(-v*t),den)
    curve = np.divide(1-np.exp(-v*t),den) - np.exp(-v*t)
    return l*level + s*slope + c*curve

def getStep(myT,tenor,h):
    myRange = np.insert(tenor,0,0)
    for n in range(0,len(tenor)):
        if myT==0:
            loc = 0
        elif (myT>myRange[n]) & (myT<=myRange[n+1]):
            loc = n
        elif myT>myRange[n+1]:
            loc = -1
    try: return h[loc]
    except:
        print("No value found!")
 
def survivalProb(myT,tenor,h):    
    myH = getStep(myT,tenor,h)    
    return np.exp(-myH*myT)

def defaultDensity(myT,tenor,h):
    myH = getStep(myT,tenor,h)    
    myS = survivalProb(myT,tenor,h)
    return myH*myS

def getCouponStream(beta,fRate,tenor,h,Delta,d):    
    myCoupon = 0
    for i in range(0,beta):
        myCoupon += fRate*Delta*np.interp((i+1)*Delta,tenor,d)* \
                survivalProb((i+1)*Delta,tenor,h)
    return myCoupon

def getPremiumAccrual(beta,fRate,tenor,h,Delta,d):
    myAccrual = 0
    for i in range(0,beta):
        sIncrement = survivalProb(i*Delta,tenor,h) - survivalProb((i+1)*Delta,tenor,h)
        myAccrual += fRate*(Delta/2)*np.interp(i*Delta+(Delta/2),tenor,d)*sIncrement
    return myAccrual

def getProtectionPayment(beta,fRate,tenor,h,Delta,d,R):
    myProtection = 0
    for i in range(0,beta):
        sIncrement = survivalProb(i*Delta,tenor,h) - survivalProb((i+1)*Delta,tenor,h)        
        myProtection += (1-R)*np.interp(i*Delta+(Delta/2),tenor,d)*sIncrement
    return myProtection

def cdsPrice(h,cds,tenor,Delta,d,R):
    K = len(h)
    cStream = np.zeros(K)
    aStream = np.zeros(K)
    xStream = np.zeros(K)
    beta = np.zeros(K)
    for n in range(0,K):
        beta = (tenor[n]/Delta).astype(int)
        cStream[n] = getCouponStream(beta,cds[n],tenor,h,Delta,d)
        aStream[n] = getPremiumAccrual(beta,cds[n],tenor,h,Delta,d)
        xStream[n] = getProtectionPayment(beta,cds[n],tenor,h,Delta,d,R)
    return cStream+aStream-xStream    

      