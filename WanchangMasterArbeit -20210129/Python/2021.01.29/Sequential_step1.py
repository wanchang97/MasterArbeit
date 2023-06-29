import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import seaborn as sns
from ERANataf import ERANataf
from ERADist import ERADist
"""
Sequential Bayesian updating 1 
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
* nESS_list_final           : intermediate nESS
* Leval_allyear_list        : likelihood at object year (it is written in physical space)
----------------------------------------------------------------
Output
* U_object          : samples in standard normal space in object year
* X_object          : samples in original space in object year
* Z_object          : normalisation constant in object year
"""
def Sequential_step1(prior,T_object,lv,samplesU,mu_U_list,si_U_list,nESS_list,Leval_allyear_list):
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
    pi_list = nESS_list/sum(nESS_list)
    Pi = np.array(pi_list)
    h_list = list()
    w_final_list = list()
    z_list = list()
    for t in range(lv):
        h = sp.stats.multivariate_normal.pdf(samplesU[t].T,mu_U_list[t],si_U_list[t]) # N h_list[0]->h_list[lv-1] prior to posterior pdf
        h_list.append(h)
    h_mixture = sum(h_list)/lv # N,
    for t in range(lv):
        f_prior = sp.stats.multivariate_normal.pdf(samplesU[t].T,mean=np.zeros(dim),cov = np.identity(dim)) #N
        L_object = Leval_allyear_list[t][T_object-1,:]
        w_new  = f_prior*L_object/h_mixture # N
        z = np.mean(w_new)
        z_list.append(z)
        w_normalised = w_new/sum(w_new)
        w_final = w_normalised*pi_list[t]
        w_final_list.append(w_final)
    # Evaluate the evidence    
    # Z_object1 = np.sum(w_new_list)/lv
    # Z_object2 = np.sum(w_real_list)/lv
    # Z_object3 = np.sum(w_final_list)/lv
    Z = np.array(z_list)
    Z_object4 = np.sum(Z*Pi)

    # resample to get the samples following the distribution in the object year
    W_final = np.array(w_final_list).flatten()
    W_resample = W_final/sum(W_final)
    SamplesU = np.array(samplesU) # lvxdimxN
    SU = SamplesU.reshape(dim,-1) # dimxlvxN
    indices = np.random.choice(lv*N,size=N,replace=True,p=W_resample)
    U_object = SU[:,indices] # dimxN 
    X_object = u2x(U_object)

    return U_object,X_object,Z_object4
