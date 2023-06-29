import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pytictoc import TicToc
from ERADist import ERADist
from ERANataf import ERANataf
from CEBU_SG import CEBU_SG
from CEBU_SG_log import CEBU_SG_log
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


time = TicToc() # create instance of class
#===================================================================================================================================================
# Problem definition
# This is an example where prior p(mu) ~ N(0,1)
dim = 1; M_array = np.array([2,10,20])# number of observations   
mu_prior, si_prior = 60, 10
#prior = lambda theta: sp.stats.norm.pdf(theta,loc = mu_prior, scale = si_prior)
dist_prior = ERADist("Normal","MOM",[mu_prior,si_prior])
prior_pdf = lambda theta: dist_prior.pdf(theta)
#log_prior = lambda theta: sp.stats.norm.logpdf(theta,loc=mu_theta_prior,scale = si_prior)
# Likelihood functions p(D|mu) ~ normal
# a
a = lambda x: x
np.random.seed(19)
x_given = 90
a_true = a(x_given)

for M in M_array: 
    # data = np.array([2])
    data = np.random.normal(loc = a_true,scale = a_true/10,size=M)
    print('data:',data)
    # np.mean(data)#data.size
    # Likelihood function
    # We observe directly the some observations of the inputs, not the model outputs
    # the additive error is normal distributed with zero mean and std 0.5
    mu_error = 0; si_error = 1
    def Likelihood(theta):
        total_likelihood = 1
        for i in range(data.size):
            total_likelihood = total_likelihood*sp.stats.norm.pdf(data[i]-theta,loc = mu_error,scale = si_error)
        return total_likelihood
    def log_Likelihood(theta):
        total_likelihood = 0
        for i in range(data.size):
            total_likelihood = total_likelihood+sp.stats.norm.logpdf(data[i]-theta,loc = mu_error, scale = si_error)
        return total_likelihood
    # maxL_x = sp.optimize.fmin(lambda x: -Likelihood(x),0)
    # c = 1/Likelihood(maxL_x)   #0.5*np.sqrt(2*np.pi) 

    #====================================================================================================================================
    # reference solutions
    n = data.size
    x_bar = np.mean(data)
    mu_exact = (mu_prior/si_prior**2+n*x_bar/si_error**2)/(1/si_prior**2+n/si_error**2)
    si_exact = (1/si_prior**2+n/si_error**2)**(-1/2)
    def posterior_analytical(theta):
        return sp.stats.norm.pdf(theta,loc = mu_exact, scale = si_exact)
    # cE_exact = 0.09 / c
    def target_dist(theta):
        return prior_pdf(theta) * Likelihood(theta)
    lower = x_given-3*si_prior; upper = x_given + 3*si_prior
    cE_reference = sp.integrate.fixed_quad(target_dist,lower,upper,n=1000)

    #=======================================================================================================================================================================
    # aBUS with Cross Entropy method
    N_CE = 1000
    N_final = 1000
    max_it = 50
    CV_target = 1.5
    time.tic()
    method = 'CEBU_SG'
    if method == 'CEBU_SG':
        [cE,lv,samplesU,samplesX,U_resample,X_resample,beta_t,mu_U_list,si_U_list,nESS] = CEBU_SG(N_CE,N_final,Likelihood,[dist_prior],max_it,CV_target)
        t_CEBU_SG = time.tocvalue()
        text = ('$Z_{{ref}}$  = {}\n$Z_{{CEBU}}$ ={}\n$mu_{{post_{{exact}}}}$ ={}\n$mu_{{post_{{CEBU}}}}$ = {}\n$si_{{post_{{exact}}}}$= {}\n$si_{{post_{{CEBU}}}}$={}\nnESS = {}\n$t_{{CEBU_{{SG}}}}$={}s'.format(cE_reference[0],cE,mu_exact,np.mean(samplesX[-1]),si_exact,np.std(samplesX[-1]),nESS,t_CEBU_SG))
    if method == 'CEBU_SG_log':
        [cE,lv,samplesU,samplesX,U_resample,X_resample,beta_t,mu_U_list,si_U_list,nESS] = CEBU_SG_log(N_CE,N_final,log_Likelihood,[dist_prior],max_it,CV_target)
        t_CEBU_SG_log = time.tocvalue()
        text = ('$Z_{{ref}}$  = {}\n$Z_{{CEBU}}$ ={}\n$mu_{{post_{{exact}}}}$ ={}\n$mu_{{post_{{CEBU}}}}$ = {}\n$si_{{post_{{exact}}}}$= {}\n$si_{{post_{{CEBU}}}}$={}\nnESS = {}\n$t_{{CEBU_{{SG_{{log}}}}}}$={}s'.format(cE_reference[0],cE,mu_exact,np.mean(samplesX[-1]),si_exact,np.std(samplesX[-1]),nESS,t_CEBU_SG_log))
    nCE_total = (lv-1)*N_CE+N_final

    
    # # Plot the intermediate samples using the intermediate parameters
    # fig1 = plt.figure()
    # ax = fig1.add_axes([0.1,0.1,0.8,0.8])# left, bottom, width, height (range 0 to 1)
    # nnp = 200
    # xx = np.linspace(-6,6,nnp)
    # prior = sp.stats.norm.pdf(xx,mu_U_list[0],si_U_list[0])
    # posterior = sp.stats.norm.pdf(xx,mu_U_list[-1],si_U_list[-1])
    # # posterior_ana = sp.stats.norm.pdf(xx,mu_exact,si_exact)
    # p1 = ax.plot(xx,prior.flatten(),label='prior, beta=0')
    # for t in range(1,lv):
    #     ax.plot(xx,sp.stats.norm.pdf(xx,mu_U_list[t],si_U_list[t]).flatten())
    # p2 = ax.plot(xx,posterior.flatten(),label = 'posterior, beta=1')
    # # p3 = ax.plot(xx,posterior_ana,'-*',label='posterior_analytical')
    # p4 = ax.plot(xx,Likelihood(xx),'-.',label='Likelihood')
    # ax.text(0.24,0.5,text,horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,bbox=dict(facecolor='yellow', alpha=0.5))
    # ax.set_xlabel('u')
    # ax.set_ylabel('PDF')
    # ax.set_title('1RV_Intermediate distribution with {} observations'.format(M))
    # ax.legend()
    # if method == 'CEBU_SG':
    #     fig1.savefig('figures\\CEBUexamples\\Example1_1RV_directObservations\\SG\\Intermediate distribution with {} observations using {} lvs.png'.format(M,lv),dpi=200)
    # if method == 'CEBU_SG_log':
    #     fig1.savefig('figures\\CEBUexamples\\Example1_1RV_directObservations\\SG_log\\Intermediate distribution with {} observations.png'.format(M),dpi=200)
    

    # # Plot the intermediate density using the intermediate samples
    # fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4))
    # axes2[0] = plt.subplot(1,2,1)    
    # sns.distplot(samplesU[0].flatten(),label='prior,beta=0')
    # for sample in samplesU[1:-1]:
    #     sns.distplot(sample.flatten())
    # sns.distplot(samplesU[-1].flatten(),label='posterior,beta=1,lv = {}'.format(lv))
    # axes2[0].set_xlabel('u')
    # axes2[0].set_ylabel('PDF')
    # axes2[0].legend()
    # axes2[0].set_title('1RV_direct_Intermediate distribution in tranformed space')
    # axes2[1] = plt.subplot(1,2,2)
    # sns.distplot(samplesX[0].flatten(),label='prior,beta=0')
    # for sample in samplesX[1:-1]:
    #     sns.distplot(sample.flatten())
    # sns.distplot(samplesX[-1].flatten(),label='posterior,beta=1,lv = {}'.format(lv))
    # axes2[1].set_xlabel('x')
    # axes2[1].set_ylabel('PDF')
    # axes2[1].set_title('...in physical space')
    # axes2[1].legend()
    # if method == 'CEBU_SG':
    #     fig2.savefig('figures\\CEBUexamples\\Example1_1RV_directObservations\\SG\\Intermediate distribution2 with {} observations using {}lvs InterSamples.png'.format(M,lv),dpi=200)
    # if method == 'CEBU_SG_log':
    #     fig2.savefig('figures\\CEBUexamples\\Example1_1RV_directObservations\\SG_log\\Intermediate distribution2 with {} observations using {}lvs InterSamples.png'.format(M,lv),dpi=200)
    # Plot the intermediate density using the intermediate samples
    fig3, ax3 = plt.subplots(1, 1, figsize=(5, 4))
    # fig3 = plt.figure()

    # ax3 = fig3.add_axes([0.1,0.1,0.6,0.6])# left, bottom, width, height (range 0 to 1)
    sns.distplot(samplesX[0].flatten(),label='$dist_{prior}$,beta=0')
    for sample in samplesX[1:-1]:
        sns.distplot(sample.flatten())
    sns.distplot(samplesX[-1].flatten(),label='$dist_{{posterior}}$,beta=1,lv = {}'.format(lv))
    ax3.set_xlabel('x')
    ax3.set_ylabel('PDF')
    ax3.set_title('1RV_direct_InterDist. in X space with {} observations '.format(M))
    ax3.text(0.35,0.52,text,horizontalalignment='center',verticalalignment='center', transform=ax3.transAxes,bbox=dict(facecolor='none',edgecolor='none', alpha=0.5))
    ax3.legend()
    if method == 'CEBU_SG':
        fig3.savefig('figures\\CEBUexamples\\Example1_1RV_directObservations\\SG\\InterDist in X space with {} observations using {}lvs InterSamples.png'.format(M,lv),dpi=200)
    if method == 'CEBU_SG_log':
        fig3.savefig('figures\\CEBUexamples\\Example1_1RV_directObservations\\SG_log\\InterDist in X space with {} observations using {}lvs InterSamples.png'.format(M,lv),dpi=200)

    # show results of CEBU

    print('\nModel evidence reference = ', cE_reference[0])
    print('Model evidence BUS_CE = ', cE)
    print('\nExact posterior mean = ', mu_exact)
    print('Mean value of samples = ', np.mean(samplesX[-1]))
    print('\nExact posterior std = ',si_exact)
    print('Std of samples = ',np.std(samplesX[-1]))
    print('\nFinal effective sample size = ',nESS)
    print('total evaluated samples : ',nCE_total,'\n')

