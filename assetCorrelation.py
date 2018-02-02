import numpy as np
import math
import cmUtilities as util
import numpy.linalg as anp
import importlib 
import scipy
from scipy.stats import norm
import scipy.integrate as nInt
from scipy.stats import t as myT
import scipy.linalg as asp
from scipy.optimize import approx_fprime
import thresholdModels as th
import markovChain as mc
import time
import mixtureModels as mix

importlib.reload(util)
importlib.reload(th)
importlib.reload(mc)
importlib.reload(mix)

def jointDefaultProbability(p,q,myRho):
    pr,err=nInt.quad(jointIntegrand,-5,5,args=(p,q,myRho))
    return pr

def jointIntegrand(g,p,q,myRho):
    p1 = th.computeP(p,myRho,g)
    p2 = th.computeP(q,myRho,g)
    f = p1*p2*util.gaussianDensity(g,0,1)
    return f

def initializeRegion2r(N,rStart):
    startRegion = np.zeros(N)
    w = np.cumsum(rStart)
    u = np.random.uniform(0,1,N)
    for n in range(0,N):
        if ((u[n]>0) & (u[n]<=w[0])):
            startRegion[n] = 0
        elif ((u[n]>w[0]) & (u[n]<1)):
            startRegion[n] = 1
    return startRegion    

def createRatingData(K,N,T,Pin,wStart,myRho,nu,isT): 
    Q = cumulateTransitionMatrix(K,Pin)
    if isT==1:
        Delta = transformCumulativeTransitionMatrix_t(K,Q,nu)                            
    else:
        Delta = transformCumulativeTransitionMatrix(K,Q)                    
    Y = np.zeros([N,T]) # latent variables
    X = np.zeros([N,T]) # credit states  
    allP = np.zeros([N,T]) # default probabilities
    Xlast = mc.initializeCounterparties(N,wStart) # initial states
    X0 = Xlast
    Plast = Pin[(Xlast-1).astype(int),-1] 
    for t in range(0,T):
        Y[:,t] = th.getY(N,1,Plast,myRho,nu,isT)    
        for n in range(0,N):
            if Xlast[n] == 4:
                X[n,t] = 4
                continue
            else:
                X[n,t] = migrateRating(Xlast[n],Delta,Y[n,t])
        allP[:,t] = Pin[(Xlast-1).astype(int),-1]
        Plast = allP[:,t]
        Xlast = X[:,t]
    return X,Y,Delta,allP,X0    
    
def simulateCorrelatedTransitionData(K,N,T,Pin,wStart,myRho): 
    Q = cumulateTransitionMatrix(K,Pin)
    Delta = transformCumulativeTransitionMatrix(K,Q)                    
    Y = np.zeros([N,T]) # latent variables
    X = np.zeros([N,T]) # credit states  
    allP = np.zeros([N,T]) # default probabilities
    Xlast = mc.initializeCounterparties(N,wStart) # initial states
    X0 = Xlast
    Plast = Pin[(Xlast-1).astype(int),-1] 
    for t in range(0,T):
        Y[:,t] = th.getY(N,1,Plast,myRho)    
        for n in range(0,N):
            if Xlast[n] == 4:
                X[n,t] = 4
                continue
            else:
                X[n,t] = migrateRating(Xlast[n],Delta,Y[n,t])
        allP[:,t] = Pin[(Xlast-1).astype(int),-1]
        Plast = allP[:,t]
        Xlast = X[:,t]
    return X,Y,Delta,allP,X0    


def createRatingData2r(K,N,T,P,wStart,rStart,myRho,nu,isT): 
    Q = cumulateTransitionMatrix(K,P)
    Delta = transformCumulativeTransitionMatrix(K,Q)
    rId = initializeRegion2r(N,rStart).astype(int)
    Y = np.zeros([N,T]) # latent variables
    X = np.zeros([N,T]) # credit states
    allP = np.zeros([N,T]) # default probabilities
    Xlast = mc.initializeCounterparties(N,wStart) # initial states
    X0 = Xlast
    Plast = P[(Xlast-1).astype(int),-1] 
    for t in range(0,T):
        Y[:,t] = th.getY2r(N,1,Plast,myRho,rId,nu,P,isT)    
        for n in range(0,N):
            if Xlast[n] == 4:
                X[n,t] = 4
                continue
            else:
                X[n,t] = migrateRating(Xlast[n],Delta,Y[n,t])
        allP[:,t] = P[(Xlast-1).astype(int),-1]
        Plast = allP[:,t]
        Xlast = X[:,t]
    return X,Y,Delta,allP,X0,rId           
    
def migrateRating(lastX,Delta,myY):
        transitionRow = (lastX-1).astype(int)
        myMap = Delta[transitionRow,:]    
        if myY>=myMap[1]:
            myX = 1
        elif (myY<myMap[1]) & (myY>=myMap[2]):
            myX = 2
        elif (myY<myMap[2]) & (myY>=myMap[3]):
            myX = 3
        elif myY<myMap[3]:
            myX = 4
        return myX

def cumulateTransitionMatrix(K,M):
    H =  np.zeros([K,K]) 
    for n in range(0,K):
        for m in range(0,K):
            H[m,(K-1)-n] = np.sum(M[m,(K-1)-n:K])
    return H

def transformCumulativeTransitionMatrix(K,M_c):    
    H = np.zeros([K,K])
    for n in range(0,K):
        for m in range(0,K):
            if M_c[n,m]>=0.9999999:  
                H[n,m]=5
            elif M_c[n,m]<=0.0000001:
                H[n,m] = -5
            else:
                H[n,m] = norm.ppf(M_c[n,m])
    return H    

def transformCumulativeTransitionMatrix_t(K,M_c,nu):    
    # Element-by-element inverse-normal transform 
    # of the cumulative transition matrix
    H = np.zeros([K,K])
    for n in range(0,K):
        for m in range(0,K):
            if M_c[n,m]>=0.9999999:  
                H[n,m] = 5
            elif M_c[n,m]<=0.0000001:
                H[n,m] = -5
            else:
                H[n,m] = myT.ppf(M_c[n,m],nu)
    return H

def getSimpleEstimationData(T,X,allP):
    N,T = X.shape
    kVec = np.zeros(T)
    nVec = np.zeros(T)
    pVec = np.zeros(T)
    kVec[0] = np.sum(X[:,0]==4)
    nVec[0] = N
    pVec[0] = np.mean(allP[:,0])
    for t in range(1,T):
        kVec[t] = np.sum(X[:,t]==4)-np.sum(X[:,t-1]==4)
        nVec[t] = nVec[t-1] - kVec[t-1]
        pVec[t] = np.mean(allP[X[:,t-1]!=4,t])
    return pVec,nVec,kVec  

def get2rEstimationData(T,X,X0,rId,allP,numP):
    N,T = X.shape
    kMat = np.zeros([T,numP])
    nMat = np.zeros([T,numP])
    pMat = np.zeros([T,numP])
    for m in range(0,numP):
        xLoc = (rId==m).astype(bool)
        kMat[0,m] = np.sum(X[xLoc,0]==4)
        nMat[0,m] = np.sum(xLoc)
        pMat[0,m] = np.mean(allP[xLoc,0])
        for t in range(1,T):
            kMat[t,m] = np.sum(X[xLoc,t]==4)-np.sum(X[xLoc,t-1]==4)
            nMat[t,m] = nMat[t-1,m] - kMat[t-1,m]
            if np.sum(xLoc)==0:
                pMat[t,m] = 0.0                
            else:
                pMat[t,m] = np.mean(allP[(X[xLoc,t-1]!=4).astype(int),t])
    return pMat,nMat,kMat     
    
def getCMF(g,myRho,myP,myN,myK):
    pg = th.computeP(myP,myRho,g)
    f=util.getBC(myN,myK)*np.power(pg,myK)*np.power(1-pg,myN-myK)
    cmf = f*util.gaussianDensity(g,0,1)        
    return cmf

def logLSimple(x,T,pVec,nVec,kVec):
    L = np.zeros(T)
    for t in range(0,T):
        L[t],err = nInt.quad(getCMF,-5,5,
                     args=(x,pVec[t],nVec[t],kVec[t]))
    logL = np.sum(np.log(L))
    return -logL          

def maxSimpleLogL(T,pVec,nVec,kVec):
    myBounds = ((0.001,0.999),)                           
    xStart = 0.5
    r = scipy.optimize.minimize(logLSimple,
                    xStart,args=(T,pVec,nVec,kVec), 
                    method='TNC',jac=None,bounds=myBounds,
                    options={'maxiter':1000}) 
    return r.x,r.success    

def computeSimpleScore(x0,T,pVec,nVec,kVec):
    h = 0.00001
    fUp = logLSimple(x0+h/2,T,pVec,nVec,kVec)
    fDown = logLSimple(x0-h/2,T,pVec,nVec,kVec)    
    score = np.divide(fUp-fDown,h)
    return score

def simpleFisherInformation(x0,T,pVec,nVec,kVec):
    h = 0.000001
    f = logLSimple(x0,T,pVec,nVec,kVec)    
    fUp = logLSimple(x0+h,T,pVec,nVec,kVec)
    fDown = logLSimple(x0-h,T,pVec,nVec,kVec)    
    I = -np.divide(fUp-2*f+fDown,h**2)
    return I

def getProdCMF(g,myRho,myP,myN,myK):
    pg = th.computeP(myP,myRho,g)
    return np.multiply(util.getBC(myN,myK),np.power(pg,myK)*np.power(1-pg,myN-myK))

def getCMF2r(g,myRho,pVec3,nVec3,kVec3):
    myF=getProdCMF(g,myRho,pVec3,nVec3,kVec3)
    return np.prod(myF)*util.gaussianDensity(g,0,1)
   
def logL2r(x,T,pMat,nMat,kMat):
    L = np.zeros(T)
    for t in range(0,T):
        L[t],err = nInt.quad(getCMF2r,-5,5,args=(x,pMat[t,:],nMat[t,:],kMat[t,:]))
    return -np.sum(np.log(L))

def log2rGridSearch(gridSize,numVar,T,pMat,nMat,kMat):
    rhoRange = np.linspace(0.001,0.999,gridSize)
    bigF = np.zeros([gridSize,gridSize])
    rhoGrid = np.zeros([gridSize**numVar,numVar])
    startTime = time.time()
    for n in range(0,gridSize):
        for m in range(0,gridSize):
            print("Running: n:%d, m: %d" % (n+1,m+1))
            rhoGrid = np.array([rhoRange[n],rhoRange[m]])
            bigF[n,m] = -logL2r(rhoGrid,T,pMat,nMat,kMat)
    print("Loop takes %d minutes." % ((time.time()-startTime)/60))
    # Find the values associated with the biggest value of OF
    bigMax = np.max(bigF)
    for n in range(0,gridSize):
        for m in range(0,gridSize):
                if (bigF[n,m]==bigMax):
                    myN = n
                    myM = m
    rhoStart = np.array([rhoRange[myN],rhoRange[myM]])
    return rhoStart,bigF,rhoRange  

def max2rLogL(T,pMat,nMat,kMat,xStart):
    myBounds = ((0.001,0.999),(0.001,0.999))                           
    r = scipy.optimize.minimize(logL2r,
                    xStart,args=(T,pMat,nMat,kMat), 
                    method='TNC',jac=None,bounds=myBounds,
                    options={'maxiter':100})    
    return r.x,r.success    
   
def hessian2r(x0,epsilon,T,pMat,nMat,kMat):
    # The first derivative
    f1 = approx_fprime(x0,logL2r,epsilon,T,pMat,nMat,kMat) 
    n = x0.shape[0]
    hessian = np.zeros([n,n])
    xx = x0
    for j in range(0,n):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + epsilon # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        f2 = approx_fprime(xx, logL2r,epsilon,T,pMat,nMat,kMat) 
        hessian[:, j] = (f2 - f1)/epsilon # scale...
        xx[j] = xx0 # Restore initial value of x0        
    return hessian    

def score2r(x0,epsilon,T,pMat,nMat,kMat):
    score = approx_fprime(x0,logL2r,epsilon,T,pMat,nMat,kMat) 
    return score    

def mapRating(D,from_value,to_value,K):
    if (to_value==K) & (from_value!=K):
        d_u = D[from_value-1,to_value-1]
        d_l = -5
    elif (to_value==K) & (from_value==K):
        d_u = -5
        d_l = -5
    else:
        d_u = D[from_value-1,to_value]
        d_l = D[from_value-1,to_value-1]
    return d_l, d_u

def mapRatingData(Y,D,K):
    N,T = Y.shape
    d_low = np.zeros([N,T-1])
    d_upp = np.zeros([N,T-1])
    for n in range(0,N):
        for m in range(1,T):
            d_low[n,m-1],d_upp[n,m-1] = mapRating(D,
                                 Y[n,m-1].astype(int),
                                 Y[n,m].astype(int),K)
    return d_low,d_upp

def buildCorrelationMatrix1F(x,N):
    R = x*np.ones([N,N])+np.eye(N)
    R[R==1+x] = 1    
    return R

def logLCopula(x,X,nu):
    M,T = X.shape
    R = buildCorrelationMatrix1F(x,M)
    detR = anp.det(R)
    Rinv = anp.inv(R)
    V = 0
    for t in range(0,T):
        V += np.log(1+np.divide(np.dot(np.dot(X[:,t],Rinv),X[:,t]),nu))
    return -(-0.5*(T*np.log(detR)+(nu+M)*V))

def scoreCopula(x0,myZ,nu):
    h = 0.000000001
    fUp = -logLCopula(x0+h/2,myZ,nu)
    fDown = -logLCopula(x0-h/2,myZ,nu)    
    score = np.divide(fUp-fDown,h)
    return score

def hessianCopula(x0,myZ,nu):
    h = 0.0001
    f = logLCopula(x0,myZ,nu)    
    fUp = logLCopula(x0+h,myZ,nu)
    fDown = logLCopula(x0-h,myZ,nu)    
    I = np.divide(fUp-2*f+fDown,h**2)
    return I

def maxCopulaLogL(startX,myZ,nu):
    myBounds = ((0.001,0.999),)                           
    r = scipy.optimize.minimize(logLCopula,
                        startX,args=(myZ,nu), 
                        method='TNC',jac=None,bounds=myBounds,
                        options={'maxiter':1000}) 
    return r.x,r.success    
    

def mixtureMethodOfMoment(x,myP,myV,myModel):
    if myModel==0: # Beta-binomial
        M1 = mix.betaMoment(x[0],x[1],1)
        M2 = mix.betaMoment(x[0],x[1],2)
    elif myModel==1: # Logit
        M1,err = nInt.quad(mix.logitProbitMoment,-8,8,args=(x[0],x[1],1,1)) 
        M2,err = nInt.quad(mix.logitProbitMoment,-8,8,args=(x[0],x[1],2,1))
    elif myModel==2: # Probit
        M1,err = nInt.quad(mix.logitProbitMoment,-8,8,args=(x[0],x[1],1,0)) 
        M2,err = nInt.quad(mix.logitProbitMoment,-8,8,args=(x[0],x[1],2,0))
    elif myModel==3: # Poisson-gamma
        M1 = mix.poissonGammaMoment(x[0],x[1],1)
        M2 = mix.poissonGammaMoment(x[0],x[1],2)
    elif myModel==4: # Poisson-lognormal
        M1,err = nInt.quad(mix.poissonMixtureMoment,0.0001,0.9999,args=(x[0],x[1],1,0)) 
        M2,err = nInt.quad(mix.poissonMixtureMoment,0.0001,0.9999,args=(x[0],x[1],2,0)) 
    elif myModel==5: # Poisson-Weibull
        M1,err = nInt.quad(mix.poissonMixtureMoment,0.0001,0.9999,args=(x[0],x[1],1,1)) 
        M2,err = nInt.quad(mix.poissonMixtureMoment,0.0001,0.9999,args=(x[0],x[1],2,1)) 
    f1 = M1 - myP    
    f2 =(M2-M1**2) - myV    
    return [1e4*f1, 1e4*f2]
        
def integrateGaussianMoment(g,r,myP,myMoment):
    integrand = np.power(th.computeP(myP,r,g),myMoment)
    return integrand*util.gaussianDensity(g,0,1)

def methodOfMomentsG(x,myP,myV):
    if (x<=0) | (x>1):
        return [100,100]
    M1,err = nInt.quad(integrateGaussianMoment,-5,5,args=(x[0],myP,1)) 
    M2,err = nInt.quad(integrateGaussianMoment,-5,5,args=(x[0],myP,2)) 
    f1 = (M2-M1**2) - myV
    return 1e4*f1

def thresholdMoment(g,w,p1,p2,myP,whichModel,myMoment,invCdf=0):
    d1 = util.gaussianDensity(g,0,1)
    if whichModel==1: # t
        d2 = util.chi2Density(w,p2)
        integrand = np.power(th.computeP_t(myP,p1,g,w,p2),myMoment)
    if whichModel==2: # Variance-gamma
        d2 = util.gammaDensity(w,p2,p2)
        integrand = np.power(th.computeP_NVM(myP,p1,g,w,p2,invCdf),myMoment)
    if whichModel==3: # Generalized hyperbolic
        d2 = util.gigDensity(w,p2)
        integrand = np.power(th.computeP_NVM(myP,p1,g,w,p2,invCdf),myMoment)
    return integrand*d1*d2

def getThresholdMoments(x,myP,whichModel):
    if whichModel==0: # Gaussian
        M1,err = nInt.quad(integrateGaussianMoment,-5,5,args=(x[0],myP,1)) 
        M2,err = nInt.quad(integrateGaussianMoment,-5,5,args=(x[0],myP,2)) 
    elif whichModel==1: # t
        lowerBound = np.maximum(x[1]-40,2)
        support = [[-7,7],[lowerBound,x[1]+40]]
        M1,err=nInt.nquad(thresholdMoment,support,args=(x[0],x[1],myP,whichModel,1))
        M2,err=nInt.nquad(thresholdMoment,support,args=(x[0],x[1],myP,whichModel,2))
    elif whichModel==2: # Variance-gamma
        invCdf = th.nvmPpf(myP,x[1],0)
        support = [[-7,7],[0,100]]
        M1,err=nInt.nquad(thresholdMoment,support,args=(x[0],x[1],myP,whichModel,1,invCdf))
        M2,err=nInt.nquad(thresholdMoment,support,args=(x[0],x[1],myP,whichModel,2,invCdf))
    elif whichModel==3: # Generalized hyperbolic        
        invCdf = th.nvmPpf(myP,x[1],1)
        support = [[-7,7],[0,100]]
        M1,err=nInt.nquad(thresholdMoment,support,args=(x[0],x[1],myP,whichModel,1,invCdf))
        M2,err=nInt.nquad(thresholdMoment,support,args=(x[0],x[1],myP,whichModel,2,invCdf))
    return M1,M2

def thresholdMethodOfMoment(x,myP,myV,whichModel):
    if (x[0]<=0) | (x[0]>1):
        return 100
    M1,M2 = getThresholdMoments(x,myP,whichModel)
    f1 = M1 - myP    
    f2 =(M2-M1**2) - myV    
    return [1e4*f1,1e4*f2]

def getThresholdDefaultCorrelation(x,myP,whichModel):
    if whichModel==0: # Gaussian
        jp = th.jointDefaultProbability(myP,myP,x[0])
    elif whichModel==1: # t
        jp = th.jointDefaultProbabilityT(myP,myP,x[0],x[1])
    elif whichModel==2: # Variance-gamma
        jp = th.jointDefaultProbabilityNVM(myP,myP,x[0],x[1],0)
    elif whichModel==3: # Generalized hyperbolic
        jp = th.jointDefaultProbabilityNVM(myP,myP,x[0],x[1],1)
    return np.divide(jp-myP**2,myP*(1-myP))

def getCMF_cr(s,myW,myA,myP,myN,myK):
    ps=myP*(1-myW)+myP*myW*s
    f=util.getBC(myN,myK)*np.power(ps,myK)*np.power(1-ps,myN-myK)
    return f*util.gammaDensity(s,myA,myA)        

def logLSimple_cr(x,T,pVec,nVec,kVec,myA):
    L = np.zeros(T)
    for t in range(0,T):
        L[t],err = nInt.quad(getCMF_cr,0,100,
                     args=(x,myA,pVec[t],nVec[t],kVec[t]))
    return -np.sum(np.log(L))

def crPlusMoment(s,myW,myA,myP,momentNumber):
    v0 = myP*(1-myW) + myP*myW*s
    myDensity = util.gammaDensity(s,myA,myA)
    return np.power(v0,momentNumber)*myDensity

def crPlusScore(x0,T,pVec,nVec,kVec,myA):
    h = 0.0000001
    fUp = logLSimple_cr(x0+h/2,T,pVec,nVec,kVec,myA)
    fDown = logLSimple_cr(x0-h/2,T,pVec,nVec,kVec,myA)    
    return np.divide(fUp-fDown,h)

def crPlusFisherInformation(x0,T,pVec,nVec,kVec,myA):
    h = 0.000001
    f = logLSimple_cr(x0,T,pVec,nVec,kVec,myA)    
    fUp = logLSimple_cr(x0+h,T,pVec,nVec,kVec,myA)
    fDown = logLSimple_cr(x0-h,T,pVec,nVec,kVec,myA)    
    return -np.divide(fUp-2*f+fDown,h**2)

def jointDefaultProbabilityRegion(p,q,rhoVec):
    pr,err=nInt.quad(jointIntegrandRegion,-10,10,args=(p,q,rhoVec))
    return pr

def jointIntegrandRegion(g,p,q,rhoVec):
    p1 = th.computeP(p,rhoVec[0],g)
    p2 = th.computeP(q,rhoVec[1],g)
    density = util.gaussianDensity(g,0,1)
    f = p1*p2*density
    return f

def getRegionalDefaultCorrelation(rhoVec,myP):
    jp = jointDefaultProbabilityRegion(myP[0],myP[1],rhoVec)
    return np.divide(jp-myP[0]*myP[1],np.sqrt(myP[0]*(1-myP[0]))*np.sqrt(myP[1]*(1-myP[1])))

