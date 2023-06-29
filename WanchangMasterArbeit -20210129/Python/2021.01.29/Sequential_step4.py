import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import seaborn as sns
from ERANataf import ERANataf
from ERADist import ERADist
"""
Sequential Step 4 
Comments 
W :   Importance weights of the Cross Entropy method W = phi/h
W_t : Importance weights of the improved Cross Entropy W_t = L**beta*W
w_newt : phi(ut)*L(ut)/h_mixture, where hmixture = sum(1/lv*ht)
----------------------------------------------------------------
Input
* prior                     : list of Nataf distribution object or marginal distribution
* T_object                  : object year
* lv                        : total number of levels
* samplesU                  : object with the samples in the standard normal space
* mu_U_list                 : intermediate mu_U
* si_U_list                 : intermediate si_U
* Leval_allyear_list        : likelihood at object year (it is written in physical space)
----------------------------------------------------------------
Output
* U_object          : samples in standard normal space in object year
* X_object          : samples in original space in object year
* Z_object          : normalisation constant in object year
"""
def Sequential_step4(prior,T_object,lv,samplesU,mu_U_list,si_U_list,Leval_allyear_list):
     # initial check if there exists a Nataf object
    if isinstance(prior[0], ERANataf):  # use Nataf transform (dependence)
        dim = len(prior[0].Marginals)  # number of random variables (dimension)
        u2x = lambda u: prior[0].U2X(u)  # from u to x
        
    elif isinstance(prior[0], ERADist):  # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise
        dim = len(prior)  # number of random variables (dimension)
        u2x = lambda u: prior[0].icdf(sp.stats.norm.cdf(u))  # from u to x

    N = samplesU[0].shape[1]
    h_list = list()
    for t in range(lv):
        h = sp.stats.multivariate_normal.pdf(samplesU[t].T,mu_U_list[t],si_U_list[t]) # N h_list[0]->h_list[lv-1] prior to posterior pdf
        h_list.append(h)
    h_mixture = sum(h_list)/lv # N,
    nESS_list = list()
    w_normalised_list = list()
    z_list = list()
    for t in range(lv):
        f_prior = sp.stats.multivariate_normal.pdf(samplesU[t].T,mean=np.zeros(dim),cov = np.identity(dim)) #N
        L_object = Leval_allyear_list[t][T_object-1,:]
        w_new  = f_prior*L_object/h_mixture # N
        w_normalised = w_new/sum(w_new)
        w_normalised_list.append(w_normalised)
        z = np.mean(w_new)
        z_list.append(z)
        var_wnew = np.var(w_new)
        mu_wnew = np.mean(w_new)
        CV_wnew = np.sqrt(var_wnew)/mu_wnew
        nESS = 1/(1+CV_wnew**2)
        nESS_list.append(nESS)
    pi_list = nESS_list/sum(nESS_list)
    # Evaluate the evidence    
    Z = 0
    for t in range(lv):
        Z = Z + pi_list[t]*z_list[t]
        w_resample = pi_list[t]*w_normalised_list[t]
        w_resample_list.append(w_resample)

    Z_object = Z

    # resample to get the samples following the distribution in the object year
    W_resample = np.array(w_resample_list) # lvxN
    W_reshape = W_resample.flatten() # (N)
    W_reshape_normalised = W_reshape / np.sum(W_reshape)
    SamplesU = np.array(samplesU) # lvxdimxN
    SU = SamplesU.reshape(dim,-1) # dimxlvxN
    indices = np.random.choice(lv*N,size=N,replace=True,p=W_resample)
    U_object = SU[:,indices] # dimxN 
    X_object = u2x(U_object)

    return U_object,X_object,Z_object
