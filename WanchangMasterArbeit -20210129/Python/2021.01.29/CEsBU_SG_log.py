import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import seaborn as sns
import mpmath as mp
import sympy as sympy
from ERANataf import ERANataf
from ERADist import ERADist
"""
Cross Entropy for sequential Bayesian updating in log scale
Comments 
W :  Importance weights of the Cross Entropy method W = phi/h
W_t : Importance weights of the improved Cross Entropy W_t = L**beta*W
----------------------------------------------------------------
Input
* N         : number of samples per level
* N_final   : number of samples for resampling level
* ln_L_fun  : loglikelihood
* year      : interested year
* prior     : list of Nataf distribution object or marginal distribution
* max_it    : maximum number of iterations
* CV_target :  target Coefficient of vairation of weights
----------------------------------------------------------------
Output
* cE                     : normalisation constant
* lv                     : total number of levels
* samplesU      	     : object with the samples in the standard nromal space
* samplesX               : object with the samples in the original space
* U_resample             : resampled samples in the standard normal space
* X_resample             : resampled samples in the original space
* beta_t                 : intermediate tempering value
* mu_U_list              : intermediate mu_U
* si_U_list              : intermediate si_U
* nESS_list              : final normalised effective sample size
* lnLeval_allyear_list   : all year likelihood evaluations
"""
def CEsBU_SG_log(N,N_final,ln_L_fun,year,prior, max_it,CV_target):
    # initial check if there exists a Nataf object
    if isinstance(prior[0], ERANataf):  # use Nataf transform (dependence)
        dim = len(prior[0].Marginals)  # number of random variables (dimension)
        u2x = lambda u: prior[0].U2X(u)  # from u to x

    elif isinstance(prior[0], ERADist):  # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise
        dim = len(prior)  # number of random variables (dimension)
        u2x = lambda u: prior[0].icdf(sp.stats.norm.cdf(u))  # from u to x

    exp_array = np.frompyfunc(mp.exp,1,1)# number of input arguments, number of output argument
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
    lnLeval_allyear_list = list()
    nESS_list = list()
    # Iteration
    for t in range(max_it):
        print('\nlv ',t+1,'for beta = ',beta_t[t],',mu_U = ',mu_U,',si_U = ',si_U)
        # Generate samples and save them
        U = sp.stats.multivariate_normal.rvs(mean=mu_U,cov=si_U,size=N).reshape(dim,-1)# dimxN
        ln_U = np.log(np.abs(U))# dimxN
        samplesU.append(U) # dimxN  2x1000
        # Evaluate the likelihood function
        ln_Leval_allyear = ln_L_fun(u2x(U),year) # yearxN
        lnLeval_allyear_list.append(ln_Leval_allyear)# lvxyearxN
        ln_Leval = ln_Leval_allyear[year-1,:]# 1xN X: 4x1000
        # beta is already initialied beta will increase from 0 to 1
        # evaluate the logpdfs   
        ln_h = sp.stats.multivariate_normal.logpdf(U.T,mu_U,si_U)# 1000
        ln_phi = sp.stats.multivariate_normal.logpdf(U.T,mean=np.zeros(dim),cov = np.identity(dim)) 
        ln_W = ln_phi-ln_h   # 1000
 
        # beta updating
        ln_Wt_fun = lambda beta: ln_W + beta*ln_Leval
        Part3 = -np.log(N)+np.log(1+CV_target**2)
        Part4 = lambda beta: 2*sp.special.logsumexp(ln_Wt_fun(beta))-sp.special.logsumexp(2*ln_Wt_fun(beta))
        fmin = lambda beta: np.abs(Part3+Part4(beta))
        beta_new = sp.optimize.fminbound(fmin,beta_t[t],1)
        #=================================================
        # ln_Wt_opt_fun = lambda dbeta: ln_W + (beta_t[t] + np.abs(dbeta))*ln_Leval
        # Part3 = -np.log(N)+np.log(1+CV_target**2)
        # Part1 = lambda dbeta: 2*np.log(np.sum(np.exp(ln_Wt_opt_fun(dbeta)-np.max(ln_Wt_opt_fun(dbeta)))))
        # Part2 = lambda dbeta: -np.log(np.sum(np.exp(2*ln_Wt_opt_fun(dbeta)-2*np.max(ln_Wt_opt_fun(dbeta)))))
        # fmin = lambda dbeta: np.abs(Part1(dbeta) + Part2(dbeta) + Part3)
        # dbeta = sp.optimize.fminbound(fmin,-1,1e-6)
        # beta_new = beta_t[t] + np.abs(dbeta)
        #=======================================
        beta_t[t+1] = beta_new
        # Update W_t
        ln_Wt = ln_Wt_fun(beta_new).reshape(1,N) # 1x1000
        # For diagonistic purpose 
        W_t = exp_array(ln_Wt)
        W_t = sympy.sympify(W_t)
        W_t = np.array(W_t, dtype=np.float64)
        
        
        CV_Wt = np.std(W_t)/np.mean(W_t)
        nESS = 1/(1+CV_Wt**2)            
        print('\nCV_obtained: ',CV_Wt)
        print('nESS: ',nESS)
        nESS = np.exp(2*sp.special.logsumexp(ln_Wt)-sp.special.logsumexp(2*ln_Wt)-np.log(N))
        nESS_list.append(nESS)
        #W_t = Wt_fun(beta_new)
        # Parameter update: closed form
        # part1 = np.log(np.sum(np.sign(U)*np.exp(ln_U+ln_Wt-np.max(ln_U)-np.max(ln_Wt))))# How to guarantee that np.sum(np.sign(U)*np.exp(ln_U+ln_Wt-np.max(ln_U)-np.max(ln_Wt)))>=0
        # part1 = np.log(np.abs(np.sum(np.sign(U)*np.exp(ln_U+ln_Wt-np.max(ln_U)-np.max(ln_Wt)))))
        # part2 = -np.log(np.sum(np.exp(ln_Wt-np.max(ln_Wt))))
        # part3 = np.max(ln_U+ln_Wt)
        # part4 = -np.max(ln_Wt)
        # ln_mu_U = part1+part2+part3+part4
        # part5 = np.zeros((1,dim))
        # sign_part5  = np.zeros((1,dim))
        # for i in range(dim):
        #     part5[0,i],sign_part5[0,i] = sp.special.logsumexp(ln_U[i]+ln_Wt,b=np.sign(U[i]),return_sign=True)
        part5,sign_part5 = sp.special.logsumexp(ln_U+ln_Wt,axis=1,b=np.sign(U),return_sign=True)
        part6 = sp.special.logsumexp(ln_Wt)
        ln_mu_U = part5-part6# dim
        mu_U = sign_part5*np.exp(ln_mu_U)# dim

        # Utmp = U-mu_U.T # 2x1000 dimxN
        # Uo = Utmp.T*np.tile(np.sqrt(W_t),(dim,1)).T# 1000x2 NXdim
        # si_U = np.matmul(Uo.T,Uo)/np.sum(W_t)+1e-6*np.identity(dim) 

        Utmp = U-mu_U.reshape(dim,1)
        Uo = Utmp*np.sqrt(W_t)
        si_U = np.matmul(Uo,Uo.T)/np.sum(W_t)+1e-6*np.identity(dim)

        # # try to write si_U also in logscale
        # Utmp = U-mu_U.reshape(dim,1) # dimxN        
        # ln_Utmp = np.log(np.abs(Utmp)) # dimxN
        # sign_Utmp = np.sign(Utmp) # dimxN
        # si_tmp = np.zeros((dim,dim))
        # sign_si_tmp = np.zeros((dim,dim))
        # #si_tmp,sign_si_tmp = sp.special.logsumexp(ln_Utmp+ln_Utmp+ln_Wt,axis=1,b=sign_Utmp*sign_Utmp,return_sign=True)
        # for j in range(dim):
        #     for k in range(dim): # actually it is symmetric
        #         si_tmp[j,k],sign_si_tmp[j,k] = sp.special.logsumexp(ln_Utmp[j]+ln_Utmp[k]+ln_Wt,axis=1,b=sign_Utmp[j]*sign_Utmp[k],return_sign=True) 
        #         #si_tmp[j,k],sign_si_tmp[j,k] = sp.special.logsumexp(ln_Utmp[j]+ln_Utmp[k]+ln_Wt,axis=1,b=sign_Utmp[j]*sign_Utmp[k],return_sign=True) 
        
        # ln_Si = si_tmp-sp.special.logsumexp(ln_Wt)
        # si_U = sign_si_tmp * np.exp(ln_Si) + 1e-6*np.identity(dim) 


        mu_U_list.append(mu_U) # 1xdim
        si_U_list.append(si_U)
        if beta_t[t+1] >=  1.0 - 1e-3:
            beta_t[t+1] = 1
            break 
    # total levels
    lv = t+2
    # Generate samples from h (u,v_T)
    U_final  = sp.stats.multivariate_normal.rvs(mean=mu_U,cov=si_U,size=N_final).reshape(dim,-1)# dimxN
    samplesU.append(U_final)
    ln_Leval_allyear = ln_L_fun(u2x(U_final),year) # yearxN
    lnLeval_allyear_list.append(ln_Leval_allyear) # lvxyearxN
    ln_Leval = ln_Leval_allyear[year-1,:]# 1xN
    # Evaluate the logpdfs
    ln_h = sp.stats.multivariate_normal.logpdf(U_final.T,mu_U,si_U)# 1xdim
    ln_phi = sp.stats.multivariate_normal.logpdf(U_final.T,mean=np.zeros(dim),cov = np.identity(dim)) #1xdim
    ln_W = ln_phi-ln_h #1xdim
    ln_Wfinal = ln_Leval+ln_W
    ln_cE = sp.special.logsumexp(ln_Wfinal)-np.log(N_final)
    cE = exp_array(ln_cE)
    cE = sympy.sympify(cE)
    cE = np.array(cE, dtype=np.float64)
    W_final = exp_array(ln_Wfinal)
    W_final = sympy.sympify(W_final)
    W_final = np.array(W_final, dtype=np.float64)
    # CV_Wt =  np.std(W_t)/np.mean(W_t)
    # nESS = 1/(1+CV_Wt**2)
  
    # Calculate the integral of the posterior Integral = sum(W_t)/N
    # W_t = np.nan_to_num(W_t)
    # Z = np.sum(W_t)/N
    # Final weights
    ln_Wresample = ln_Wfinal- sp.special.logsumexp(ln_Wfinal)
    N_resample = N_final    
    nESS = np.exp(2*sp.special.logsumexp(ln_Wresample)-sp.special.logsumexp(2*ln_Wresample)-np.log(N_resample))
    nESS_list.append(nESS)
    W_resample = exp_array(ln_Wresample)
    W_resample = sympy.sympify(W_resample)
    W_resample = np.array(W_resample, dtype=np.float64)   
    indices = np.random.choice(N_final, size = N_resample, replace = True, p = W_resample.reshape(N_final))
    U_resample = U_final[:,indices]# dimxN 2x1ooo
    X_resample = u2x(U_resample)# dimxN 2x1000
    # Transform the samples to the physical/ original space
    samplesX = list()
    for i in range(lv):
        samplesX.append(u2x(samplesU[i]))#samplesX[0]:dimxN 2x1000
        
    return cE,lv,samplesU,samplesX,U_resample,X_resample,beta_t,mu_U_list,si_U_list,nESS_list,lnLeval_allyear_list
