import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import seaborn as sns
from ERANataf import ERANataf
from ERADist import ERADist
"""
CE based IS scheme especially for Sequential Bayesian updating in log scale
Comments 
W :   Importance weights of the Cross Entropy method W = phi/h
W_t : Importance weights of the improved Cross Entropy W_t = L**beta*W
----------------------------------------------------------------
Input
* prior                     : list of Nataf distribution object or marginal distribution
* T_object                  : object year
* lv                        : total number of levels
* samplesU                  : object with the samples in the standard normal space
* mu_U_list                 : intermediate mu_U
* si_U_list                 : intermediate si_U
* lnLeval_allyear_list      : loglikelihood at object year (it is written in physical space)
----------------------------------------------------------------
Output
* U_object          : samples in standard normal space in object year
* X_object          : samples in original space in object year
* Z_object          : normalisation constant in object year
"""
def Sequential_step2_log(prior,T_object ,lv ,samplesU ,mu_U_list,si_U_list,lnLeval_allyear_list):
     # initial check if there exists a Nataf object
    if isinstance(prior[0], ERANataf):  # use Nataf transform (dependence)
        dim = len(prior[0].Marginals)  # number of random variables (dimension)
        u2x = lambda u: prior[0].U2X(u)  # from u to x
        
    elif isinstance(prior[0], ERADist):  # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise
        dim = len(prior)  # number of random variables (dimension)
        u2x = lambda u: prior[0].icdf(sp.stats.norm.cdf(u))  # from u to x
    N = samplesU[0].shape[1] # number of samples per level
    ln_z_list = list()
    nESS_list = list()
    ln_w_normalised_list = list()
    w_resample_list = list()
    for t in range(lv):
        ln_h = sp.stats.multivariate_normal.logpdf(samplesU[t].T,mu_U_list[t],si_U_list[t]) # N h_list[0]->h_list[lv-1] prior to posterior pdf
        #ln_h_list.append(ln_h)
        ln_f_prior = sp.stats.multivariate_normal.logpdf(samplesU[t].T,mean=np.zeros(dim),cov = np.identity(dim)) #1xN
        ln_L_object = lnLeval_allyear_list[t][T_object-1,:]
        ln_w_new  = ln_f_prior+ln_L_object-ln_h
        ln_w_normalised = ln_w_new-sp.special.logsumexp(ln_w_new)
        ln_w_normalised_list.append(ln_w_normalised)
        ln_z_t = sp.special.logsumexp(ln_w_new)-np.log(N)# sum(w_new)/N # logsumexp(ln_w_new)-ln(N)
        ln_z_list.append(ln_z_t)
        # var_wnew = np.var(w_new)
        # mu_wnew = np.mean(w_new)
        # CV_wnew = np.sqrt(var_wnew)/mu_wnew
        # nESS = 1/(1+CV_wnew**2)
        nESS = np.exp(2*sp.special.logsumexp(ln_w_new)-sp.special.logsumexp(2*ln_w_new)-np.log(N))
        nESS_list.append(nESS)
                
    pi_list = nESS_list / np.sum(nESS_list)
    # Evidence evaluation
    Z = 0
    for t in range(lv):
        Z = Z + pi_list[t]*np.exp(ln_z_list[t])
        w_resample = pi_list[t]*np.exp(ln_w_normalised_list[t])
        w_resample_list.append(w_resample)

    Z_object = Z

    # generate samples according to filter distribution

    W_resample = np.array(w_resample_list) # lvxN
    W_resample = W_resample.flatten() # (N)
    W_resample_normalised = W_resample / np.sum(W_resample)
    SamplesU = np.array(samplesU) # lvxdimxN
    SU = SamplesU.reshape(dim,-1) # dimx(lvxN)
    indices = np.random.choice(lv*N,size=N,replace=True,p=W_resample_normalised)
    U_object = SU[:,indices] # dimxN 
    X_object = u2x(U_object)
    #prior_object = sp.stats.multivariate_normal.pdf(U_object.T,mean=np.zeros(dim),cov = np.identity(dim))

    
    
    # Z_object = np.mean(W_new.flatten())
    return U_object,X_object,Z_object