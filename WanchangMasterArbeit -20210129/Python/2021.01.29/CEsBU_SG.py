import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from ERANataf import ERANataf
from ERADist import ERADist
"""
Cross Entropy expeciallly for sequential Bayesian updating one year one data
Comments 
W :  Importance weights of the Cross Entropy method W = phi/h
W_t : Importance weights of the improved Cross Entropy W_t = L**beta*W
----------------------------------------------------------------
Input
* N         : number of samples per level
* N_final   : number of samples for resampling level
* L_all_fun : likelihood_allyear(x,t)
* year      : interested year
* prior     : list of Nataf distribution object or marginal distribution
* max_it    : maximum number of iterations
* CV_target : target Coefficient of vairation of weights
----------------------------------------------------------------
Output
* cE                : normalisation constant(evidence)
* lv                : total number of levels
* samplesU          : object with the samples in the standard nromal space
* samplesX          : object with the samples in the original space
* U_resample        : resampled samples in the standard nromal space
* X_resample        : resampled samples in the original space
* beta_t            : intermediate tempering value
* mu_U_list         : intermediate mu_U
* si_U_lsit         : intermediate si_U
* nESS_list         : final normalised effective sample size= 1/(1+cv_Wt**2)
* Leval_allyear_list: Likelihood evaluation for all year for each level
"""
def CEsBU_SG(N,N_final,L_fun,year,prior,max_it,CV_target):
    # initial check if there exists a Nataf object
    if isinstance(prior[0], ERANataf):  # use Nataf transform (dependence)
        dim = len(prior[0].Marginals)  # number of random variables (dimension)
        u2x = lambda u: prior[0].U2X(u)  # from u to x
    elif isinstance(prior[0], ERADist):  # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise
        dim = len(prior)  # number of random variables (dimension)
        u2x = lambda u: prior[0].icdf(sp.stats.norm.cdf(u))  # from u to x
    else:
        raise RuntimeError(
            "Incorrect distribution. Please create an ERADist/Nataf object!"
        )
    mu_init = np.zeros(dim)
    si_init = np.identity(dim)
    beta_t = np.zeros(max_it+1)
    nESS = 0
    samplesU = list() 
    # CE procedure
    mu_U = mu_init
    mu_U_list = list()
    mu_U_list.append(mu_U)
    si_U =  si_init
    si_U_list = list()
    si_U_list.append(si_U)
    Leval_allyear_list = list()
    nESS_list = list()
    # Iteration
    for t in range(max_it):
        print('\n lv= ',t+1,'for beta = ',beta_t[t],',mu_U = ',mu_U,',si_U = ',si_U)
        # Generate samples and save them
        U = sp.stats.multivariate_normal.rvs(mean=mu_U.flatten(),cov=si_U,size=N).reshape(dim,-1)# dimxN
        samplesU.append(U)
        # Evaluate the likelihood function
        Leval_allyear = L_fun(u2x(U),year) # 1x1000
        Leval_allyear_list.append(Leval_allyear)
        Leval = Leval_allyear[year-1,:]
        # beta is already initialied beta will increase from 0 to 1
           
        # calculatiing h for the likelihood weight
        h = sp.stats.multivariate_normal.pdf(U.T,mu_U.flatten(),si_U) # N
        phi = sp.stats.multivariate_normal.pdf(U.T,mean=np.zeros(dim),cov = np.identity(dim)) #N
        W = phi/h # N

        Wt_fun = lambda beta: W*np.power(Leval,beta)
        CV_Wt_fun = lambda beta: np.std(Wt_fun(beta))/np.mean(Wt_fun(beta))
        
        Wt2_fun = lambda beta: np.power(Leval,beta-beta_t[t])
        CV_Wt2_fun = lambda beta: np.std(Wt2_fun(beta))/np.mean(Wt2_fun(beta))
        ESS_target = N/(1+CV_target**2)
        ESS_observed = lambda beta: N/(1+(CV_Wt2_fun(beta))**2)

        # Wt_fun = lambda beta: np.power(Leval,(beta-beta_t[t]))
        # CV_Wt_fun = lambda beta: np.std(Wt_fun(beta))/np.mean(Wt_fun(beta)
        # #ESS_target = N/(1+CV_target**2)
        # ESS_observed = lambda beta: N/(1+(CV_Wt_fun(beta))**2)
        #fmin = lambda beta: abs(ESS_target-ESS_observed(beta))
        fmin = lambda beta: np.linalg.norm((ESS_target-ESS_observed(beta))**2)
        beta_new = sp.optimize.fminbound(fmin,beta_t[t],1)
        beta_t[t+1] = beta_new
        # Update W_t
        W_t = Wt_fun(beta_new)  

        # For diagnostics precalculation  
        CV_Wt =  CV_Wt_fun(beta_new)
        nESS = 1/(1+CV_Wt**2)
        nESS_list.append(nESS)
        print('\n CV_obtained: ',CV_Wt)
        print('nESS: ',nESS)

        # Parameter update: closed form
        mu_U = U@W_t.T/np.sum(W_t) #2x1
        Utmp = U-mu_U.reshape(dim,1)# 2x1000

        Uo = Utmp*np.sqrt(W_t)# 2x1000
        si_U = np.matmul(Uo,Uo.T)/np.sum(W_t)+1e-6*np.identity(dim) 
        mu_U_list.append(mu_U)
        si_U_list.append(si_U)
        if beta_t[t+1] >=  1.0 - 1e-3:
            beta_t[t+1] = 1
            break  
    # total levels
    lv = t+2
    # Generate samples using the new parameters
    U = sp.stats.multivariate_normal.rvs(mean=mu_U.flatten(),cov=si_U,size=N_final).reshape(dim,-1)# dimx2
    samplesU.append(U)
    # Evaluate the likelihood function
    Leval_allyear = L_fun(u2x(U),year) # 1x1000
    Leval_allyear_list.append(Leval_allyear)
    Leval = Leval_allyear[year-1,:]
    # evaluate pdfs
    h = sp.stats.multivariate_normal.pdf(U.T,mu_U.flatten(),si_U) # 1xdim
    phi = sp.stats.multivariate_normal.pdf(U.T,mean=np.zeros(dim),cov = np.identity(dim)) #1xdim
    W = phi/h #1xdim
    W_final = Leval*W
    # Calculate the integral of the posterior Integral = sum(W_t)/N
    cE = np.sum(W_final)/N_final


    # Final weights
    W_resample = (W_final/np.sum(W_final)) 
    N_resample = N_final
    indices = np.random.choice(N_resample, size = N_resample, replace = True, p = W_resample.reshape(N_resample))
    U_resample = U[:,indices]
    X_resample = u2x(U_resample)
        
    CV_Wresample =  np.std(W_resample)/np.mean(W_resample)
    nESS = 1/(1+CV_Wresample**2)
    nESS_list.append(nESS)
    # Transform the samples to the physical/ original space
    samplesX = list()
    for i in range(lv):
        samplesX.append(u2x(samplesU[i]))
    return cE,lv,samplesU,samplesX,U_resample,X_resample,beta_t,mu_U_list,si_U_list,nESS_list,Leval_allyear_list
