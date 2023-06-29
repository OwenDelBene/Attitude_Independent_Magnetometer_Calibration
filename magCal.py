#Algorithm
#Compute centered estimate (b~*') eq 33
#Compute covariance matrix (P~bb) eq 34

#Compute F~bb equation 34
#Compute F-bb equation 44b
#Choose a value for c 
#if equation 45, return b~*

#else use b~*' as initial estimate
#iterate equation 46 until equation 48 is really small
import numpy as np 
import pandas as pd
#Bk is measured magnetic field at time tk
#sigmak = variance for gaussian white noise = np.random.normal(Size)k
# 1/(sigmabar**2) = sum(1/(sigmak**2))
#Bbar = sigmabar**2 * sum(Bk/(sigmak**2))
#B~k = Bk - Bbar

#Hk = geomagnetic field with respect to earth fixed coordinates
#zk = |Bk|**2 - |Hk|**2
#zbar = sigmabar**2 * sum(zk/sigmak**2)
#z~ = Zk - Zbar

#assumptions assumed an effective white Gaussian magnetometer
#measurement error with isotropic error distribution 
#and a standard deviation per axis of 2.0 mG

def inverse(m):
    try: 
        return np.linalg.inv(m)
    except: 
        a,b = m.shape
        assert a==b
        i = np.eye(a,a)
        return np.linalg.lstsq(m,i, rcond=None)[0]

def getSigma(n):
    #n: size of measurement data set
    return np.random.normal(size=n)

def getSigmabar(sigma):
    _sum = 0
    for sigmak in sigma:
        _sum += (1/(sigmak **2))
    return np.sqrt(1.0/_sum )

def getBbar(sigma, sigmaBar, B):
    #eq 13b
    #sigmaBar = getSigmabar(sigma)
    _sum = np.zeros(B.shape[1])
    for i,sigmak in enumerate(sigma):
        _sum += (B[i]/(sigmak**2))
    return (sigmaBar**2) * _sum


def getB_tilde(B, Bbar):
    #eq 16b
    return B - Bbar

def getZ(B, H):
    #eq 3a
    z = np.zeros(B.shape[0])
    for i,Bk in enumerate(B):
        z[i] = np.linalg.norm(Bk)**2 - np.linalg.norm(H[i])**2

    return z

def getbar(sigmabar, sigma, x):
    _sum = 0
    for i,sigmak in enumerate(sigma):
        
        _sum+= (x[i]/(sigmak**2))

    return _sum * (sigmabar**2) 

def get_tilde(x, xbar):
    return x-xbar


def errorCovarianceEstimate(B_tilde, sigma):
    #eq 34 
    N= B_tilde.shape[0]
    M = B_tilde.shape[1]
    P_tilde_bb = np.zeros((M,M))
    for i in range(N):
        P_tilde_bb += (4* np.matmul(B_tilde[i], B_tilde[i].T) / (sigma[i]**2))
    P_tilde_bb = inverse(P_tilde_bb)
    #P_tilde_bb = np.linalg.inv(P_tilde_bb)
    return P_tilde_bb

def centeredEstimate(P_tilde_bb, B_tilde, sigma, z_tilde, mu_tilde):
    #eq 33 
    n = B_tilde.shape[0]
    # b_tilde = np.zeros(B_tilde.shape)
    _sum = np.zeros(B_tilde.shape[1])
    for i in range(n):
        _sum += (z_tilde[i] - mu_tilde[i]) * 2 * B_tilde[i] / (sigma[i]**2) 
    b_tilde = np.matmul(P_tilde_bb, _sum)

    return b_tilde  

def getF_tilde_bb(P_tilde_bb):
    #Eq 34
    #return np.linalg.inv(P_tilde_bb)
    return inverse(P_tilde_bb)

def getFbar(Fbb, F_tilde_bb):
    #Eq 44
    #return np.linalg.inv(P_tilde) - F_tilde
    return Fbb - F_tilde

def needSecondStep(Fbar, F_tilde, c=.5, threshold = 1e-5):
    # Eq 45
    insufficient = True
    
    #Check diagonal terms are sufficiently small eq 45
    for i,v in enumerate(Fbar):
        #check diagonal elements
        if ( Fbar[i][i] < c*F_tilde[i][i]  ): 
            insufficient = False
        else:
            return True
    return insufficient 

def getFbb(P_tilde_bb, sigmabar, Bbar, b):
    # Eq 44
    Pinv = inverse(P_tilde_bb)
    return Pinv + (4/(sigmabar**2)) * (Bbar - b)*(Bbar -b).T

def secondStepComplete(b0,b1, F_bb, threshold = 1e-5):
    #equation 48
    a = np.matmul((b1-b0).T, F_bb)
    #b = np.matmul(a, b0)
    c = np.matmul(a, (b1-b0)) 
    return False #c < np.full(c.shape, threshold) 

def getGradientVector(P_tilde_bb, b, b_tilde_star, sigmabar, z_bar_prime, Bbar, mubar):
    #eq 47
    a = np.matmul(inverse(P_tilde_bb), b-b_tilde_star)
    #print(f'test a {a} p {P_tilde_bb} b~ {b_tilde_star} b {b} b==b {b==b_tilde_star} b-b {b-b_tilde_star} sigmabar {sigmabar**2} b**2 {np.linalg.norm(b)**2}')

    a1 = 2*(z_bar_prime - 2*np.dot(Bbar, b) + np.linalg.norm(b)**2 -mubar)/(sigmabar**2)
    a2 = Bbar - b
    return a - a1*a2#np.matmul(a1,a2)

def getNewBestimte(b0, F_bb, g ):
    #eq 46
    Finv = inverse(F_bb)
    return b0 - np.matmul( Finv, g)

    


if __name__ =="__main__" :
    for i in range(4):
        
        df = pd.read_csv(f"MagData{i}.csv")
        igrf = df.iloc[:,0:3].to_numpy() * 1e9 #nT
        magnetometer = df.iloc[:, 4:7 ].to_numpy() * 1e9 #nT

        
        #covariance = np.cov(magnetometer)
        
       
        B = magnetometer# measured magnetic field
        H = igrf#igrf13
        
        N = B.shape[0]
        #sigma = getSigma(N)
        #standardDeviation = 200 #nT
        
        standardDeviation = np.sqrt(np.var(magnetometer))
        sigma = np.full(B.shape[0], standardDeviation)
        sigmabar = getSigmabar(sigma)

        Bbar = getBbar(sigma, sigmabar, B)

        B_tilde = getB_tilde(B, Bbar)
        P_tilde_bb = errorCovarianceEstimate(B_tilde, sigma)

        z = getZ(B, H)
        zbar = getbar(sigmabar, sigma, z)
        z_tilde = get_tilde(z,zbar)

        covariance = np.array([np.cov(np.stack(( H[i], mag), axis=0)) for i,mag in enumerate(B)])
        mu = np.array( [ -np.trace(cov)  for cov in covariance])
        mubar = getbar(sigmabar, sigma, mu)
        mu_tilde = get_tilde(mu,mubar)

        #initial Guess
        b_tilde = centeredEstimate(P_tilde_bb, B_tilde, sigma, z_tilde, mu_tilde)

    #check 
        F_tilde = getF_tilde_bb(P_tilde_bb)
        Fbb = getFbb(P_tilde_bb, sigmabar, Bbar, b_tilde)
        Fbar = getFbar(Fbb, F_tilde)

        if needSecondStep(Fbar, F_tilde) and False:
            print('need second step')
            b0 = np.copy(b_tilde)
            g = getGradientVector(P_tilde_bb, b0, b_tilde, sigmabar, zbar, Bbar, mubar)
            Fbb = getFbb(P_tilde_bb, sigmabar, Bbar, b0)

            b1 = getNewBestimte(b0, Fbb, g)

            
            count = 0
            while not secondStepComplete(b0, b1, Fbb) and (count < 2):
                b0 = b1
                Fbb = getFbb(P_tilde_bb, sigmabar, Bbar, b0)

                b1 = getNewBestimte(b0, Fbb, g)
                g = getGradientVector(P_tilde_bb, b0, b_tilde, sigmabar, zbar, Bbar, mubar)
                count+=1
            b_final = b1
        else:
            print('sufficient')
            b_final  = b_tilde
        print(f'{i}th data set:\n{b_final}')
        print()


