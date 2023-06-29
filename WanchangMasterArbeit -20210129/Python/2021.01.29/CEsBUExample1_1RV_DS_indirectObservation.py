import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pytictoc import TicToc
from ERADist import ERADist
from ERANataf import ERANataf
from CEsBU_SG import CEsBU_SG
from CEsBU_SG_log import CEsBU_SG_log
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
time = TicToc() # create instance of class
from Sequential_step1 import Sequential_step1
from Sequential_step1_log import Sequential_step1_log
from Sequential_step2 import Sequential_step2
from Sequential_step2_log import Sequential_step2_log
from Sequential_step3 import Sequential_step3
from Sequential_step3_log import Sequential_step3_log
from Sequential_step4 import Sequential_step4
from Sequential_step4_log import Sequential_step4_log
#===================================================================================================================================================
# Parameter definition
# Fixed parameters
T_final = 50
T_final_text = '$T_{{final}}$= {:d}\n'.format(T_final)

# RV
dim = 1
mu_prior , si_prior = 60,10
dist_prior = ERADist("Normal","MOM",[mu_prior,si_prior])
# rho = np.zeros(dim)
# dist_prior = ERANataf(dist_prior,rho)
prior_pdf = lambda x: sp.stats.norm.pdf(x,loc = mu_prior,scale=si_prior)
log_prior_pdf =  lambda x: sp.stats.norm.logpdf(x,loc = mu_prior,scale=si_prior)
#===================================================================================================================================================
# Model
m = 3.9; C = np.exp(-1.5667*m - 27.5166); a_0 = 0.1
def a(x,t):
    aPart1 = (1-m/2)*C*x**m*(np.pi)**(m/2)*10**5 * t# should between -1 ~ -1e-3
    aPart2 = a_0**(1-m/2)
    aPart3 = (1-m/2)**(-1)
    a = np.power(aPart1+aPart2,aPart3)
    return np.nan_to_num(a,nan = 50)# at between 0.01-50
np.random.seed(10)
# a_meassurements generation
x_given = 90
t_m = np.arange(1,T_final+1,1)
a_true = a(x_given,t_m)
a_data = np.random.normal(a_true,scale=a_true[0]/50,size=T_final)
mu_prior_text = '$mu_{{prior}}$ = {:.3f}\n'.format(mu_prior);si_prior_text = 'si_prior = {:.3f}\n'.format(si_prior)
x_given_text = '$x_{{given}}$ = {:.3f}\n'.format(x_given)
#=============================================================================================================================================================
fig0,ax0 = plt.subplots()
ax0.set_xlabel('$t_m$ (year)')
ax0.set_ylabel('a CrackLength(mm)')
ax0.text(0.4,0.7,'a = a(DS,t)',horizontalalignment='center',verticalalignment='center', transform=ax0.transAxes,bbox=dict(facecolor='none',edgecolor='none', alpha=0.5),size='24')
ax0.plot(t_m,a_true,'c',linewidth=2,label='$a_{{True}}$')
ax0.plot(t_m,a_data,'ro',label='$a_M$')
ax0.legend()
fig0.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\a_true_a_meassurements.png',dpi=400)
#===================================================================================================================================================
# Likelihood function  we now save all the intermediate likelihood from year 1-50
def Likelihood(x,year):# If we want to use all data in 50 years
    total_likelihood_allyear = np.zeros((year,np.size(x)))
    total_likelihood = 1
    for i in range(year):# 0 to 49
        t = int(i+1)#1 to 50
        total_likelihood = total_likelihood * sp.stats.norm.pdf(a(x,t) - a_data[i], loc=0, scale= a_true[0]/5)
        total_likelihood_allyear[i,:] = total_likelihood
    return total_likelihood_allyear

Likelihood_finalyear = lambda x: Likelihood(x,T_final)[T_final-1,:] # at year 50
Likelihood_objectyear = lambda x: Likelihood(x,T_final)[T_object-1,:]# at year 40

def log_Likelihood(x,year):
    total_likelihood_allyear = np.zeros((year,np.size(x)))
    total_likelihood  = 0
    for i in range(year):
        t = int(i+1)
        total_likelihood = total_likelihood + sp.stats.norm.logpdf(a(x,t) - a_data[i], loc=0, scale= a_true[0]/5)
        total_likelihood_allyear[i,:] = total_likelihood
    return total_likelihood_allyear

log_Likelihood_finalyear = lambda x: log_Likelihood(x,T_final)[T_final-1,:]
log_Likelihood_objectyear = lambda x: log_Likelihood(x,T_final)[T_object-1,:]# at year 40

lower = x_given-3*si_prior; upper = x_given + 3*si_prior
def target_finalyear(x):
    return prior_pdf(x)*Likelihood_finalyear(x)



constant_finalyear = sp.integrate.fixed_quad(target_finalyear,lower,upper,n=1000)

Z_final_ref_text = '$Z_{{final_{{ref}}}}$= {:.3e}\n'.format(constant_finalyear[0])

print('\nconstant_finalyear:',constant_finalyear)
print(Z_final_ref_text)
#===============================================================================================================================
# use CEsBU1 to solve the filter distribution at year T_final
print('\n......using CEsBU1 to solve the filter distr. at year T_final: {:d}'.format(T_final))
N_CE = 1000
N_final = 1000
max_it = 50
CV_target = 1.5
time.tic()
method1 = 'CEsBU_SG_log';method2 = 'CEsBU_SG_log'# method1 and method3 should be the same
if method1 == 'CEsBU_SG':
    [cE_final,lv_final,samplesU_final,samplesX_final,U_resample_final,X_resample_final,beta_t_final,mu_U_list_final,si_U_list_final,nESS_list_final,Leval_allyear_list_final] = CEsBU_SG(N_CE,N_final,Likelihood,T_final,[dist_prior],max_it,CV_target)
if method1 == 'CEsBU_SG_log':
    [cE_final,lv_final,samplesU_final,samplesX_final,U_resample_final,X_resample_final,beta_t_final,mu_U_list_final,si_U_list_final,nESS_list_final,lnLeval_allyear_list_final] = CEsBU_SG_log(N_CE,N_final,log_Likelihood,T_final,[dist_prior],max_it,CV_target) 
t_CEsBU1 = time.tocvalue() 
# analyse the result of CEsBU1
TotalSamples_CEsBU1 = (lv_final-1)*N_CE+N_final
t_CEsBU1_text = '$t_{{CEsBU1}}$ = {:.3f}\n'.format(t_CEsBU1)
nESS_CEsBU1_final_text = '$nESS_{{CEsBU1_{{final}}}}$ = {:.3f}\n'.format(nESS_list_final[-1])
lv_CEsBU1_text = '$lv_{{CEsBU1}}$ = {:d}\n'.format(lv_final)
TotalSamples_CEsBU1_list = '$TotalSamples_{{CEsBU1}}$ = {:d}\n'.format(TotalSamples_CEsBU1)
# Analyse the posterior distribution after CEsBU
mu_CEsBU1 = np.zeros((dim,1));si_CEsBU1 = np.zeros((dim,1))
for d in range(dim):
    mu_CEsBU1[d] = np.mean(X_resample_final[d,:])
    si_CEsBU1[d] = np.std(X_resample_final[d,:])

# Analyse the model output a
# mu_final_CEsBU1 = np.mean(X_resample_final)
a_CEsBU1 = np.zeros((T_final,N_CE))
for i in range(T_final):
    t = int(i+1)
    a_CEsBU1[i,:] = a(X_resample_final,t)

# credible interval
mu_a_CEsBU1 = np.mean(a_CEsBU1,axis=1)
si_a_CEsBU1 = np.std(a_CEsBU1,axis=1)
alpha = 0.05
lower_a_CEsBU1 = mu_a_CEsBU1  - si_a_CEsBU1*sp.stats.norm.ppf(1-alpha/2)
upper_a_CEsBU1 = mu_a_CEsBU1  + si_a_CEsBU1*sp.stats.norm.ppf(1-alpha/2)
# # text

mu_CEsBU1_text = '$mu_{{CEsBU1}}$ = {:.3f}\n'.format(mu_CEsBU1[0,0]);si_CEsBU1_text = 'si_CEsBU1 = {}\n'.format(si_CEsBU1[0,0])
Z_final_CEsBU1_text = '$Z_{{final_{{CEsBU1}}}}$= {:.3e}\n'.format(cE_final)
# Plot CrackLength a_True, a_M,a_CEsBU1
fig1,ax1 = plt.subplots(figsize=(8,6),dpi=400)
ax1.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
ax1.plot(t_m,a_data,'ro',label='$a_M$')
ax1.plot(t_m,mu_a_CEsBU1,'m+-',label='$mu_{a_{CEsBU1}}$')
ax1.fill_between(t_m,upper_a_CEsBU1,lower_a_CEsBU1,alpha = 0.2,color='magenta',label='$C.I._{a_{CEsBU1}}$',ls=':')
s = Z_final_ref_text+Z_final_CEsBU1_text+nESS_CEsBU1_final_text+lv_CEsBU1_text
ax1.text(0.2,0.85,s,horizontalalignment='left',verticalalignment='center', transform = ax1.transAxes,size = '14')
ax1.set_xlabel('$t_{m}$(year)')
ax1.set_ylabel('a CrackLength (mm)')
ax1.set_title('$a_{True},a_M,a_{CEsBU1}$')
ax1.legend()

if method1 == 'CEsBU_SG':
    fig1.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_CEsBU1.png',dpi=400)
if method1 == 'CEsBU_SG_log':
    fig1.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_CEsBU1.png',dpi=400)
#=====================================================================================================================================================================================================================================================================================
# Plot InterDist for CEsBU1
fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
sns.distplot(samplesX_final[0],label='$dist_{prior}$')
for sample in samplesX_final[1:-1]:
    sns.distplot(sample)
sns.distplot(samplesX_final[-1],label='$dist_{{CEsBU1}}$,lv={}'.format(lv_final))
ax2.plot(mu_prior,0,'b^',label='$mu_{{prior}}$ ')
ax2.plot(x_given,0,'bv',label='$x_{{given}}$')
ax2.plot(mu_CEsBU1[0,0],0,'ro',label='$mu_{{CEsBU1}}$')
s = Z_final_ref_text+Z_final_CEsBU1_text+mu_prior_text+x_given_text+mu_CEsBU1_text+nESS_CEsBU1_final_text
ax2.text(0.04,0.7,s,horizontalalignment='left',verticalalignment='center', transform=ax2.transAxes,bbox=dict(facecolor='none',edgecolor='none', alpha=0.5),size=12)
ax2.set_xlabel('x')
ax2.set_ylabel('PDF')
ax2.set_title('InterDist. of CEsBU1 for RV DS in X space with'+T_final_text)
ax2.legend(bbox_to_anchor=(0.7,1), loc="upper left")
#fig2.suptitle('InterDist. of CEsBU1 for RV DS in X space with'+T_final_text, fontsize=16)
if method1 == "CEsBU_SG":
    fig2.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\InterDist\\CEsBU1 with T_final = {} in X space.png'.format(T_final),dpi=400)
elif method1 == "CEsBU_SG_log":
    fig2.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\InterDist\\CEsBU1 with T_final = {} in X space.png'.format(T_final),dpi=400)
print('lv=',lv_final)
print('beta_t:',beta_t_final[0:lv_final])
print('mu_U: ',mu_U_list_final)
print('si_U: ',si_U_list_final)
print('normalisation constant cE: ',cE_final)
print('reference normalisation constant Z: ',constant_finalyear[0])
print('final normalised effective sample size : ',nESS_list_final[-1])
# # #======================================================================================================================================
# # use CEsBU to solve the filter distribution at year T_object here we are only interested in U_final X_final

# T_object = 20
# T_object_text = '$T_{{object}}$ = {:d}\n'.format(T_object)
# def target_objectyear(x):
#     return prior_pdf(x)*Likelihood_objectyear(x)
# constant_objectyear = sp.integrate.fixed_quad(target_objectyear,lower,upper,n=1000)
# Z_object_ref_text = '$Z_{{object_{{ref}}}}$= {:.3e}\n'.format(constant_objectyear[0])
# print('\n......using CEsBU2 to solve the filter distr. at year T_object: {}'.format(T_object))
# N_CE = 1000
# N_object = 1000
# max_it = 50
# CV_target = 1.5
# time.tic()
# if method2 == 'CEsBU_SG':
#     [cE_object,lv_object,samplesU_object,samplesX_object,U_resample_object,X_resample_object,beta_t_object,mu_U_list_object,si_U_list_object,nESS_list_object,Leval_allyear_list_object] = CEsBU_SG(N_CE,N_object,Likelihood,T_object,[dist_prior],max_it,CV_target)
# if method2 == 'CEsBU_SG_log':
#     [cE_object,lv_object,samplesU_object,samplesX_object,U_resample_object,X_resample_object,beta_t_object,mu_U_list_object,si_U_list_object,nESS_list_object,Leval_allyear_list_object] = CEsBU_SG_log(N_CE,N_object,log_Likelihood,T_object,[dist_prior],max_it,CV_target)
# t_CEsBU2 = time.tocvalue()
# # analyse the CEsBU2 result
# TotalSamples_CEsBU1 = (lv_object-1)*N_CE+N_object
# t_CEsBU2_text = '$t_{{CEsBU2}}$= {:.3f}\n'.format(t_CEsBU2)
# nESS_CEsBU2_final_text = '$nESS_{{CEsBU2_{{final}}}}$= {:.3f}\n'.format(nESS_list_object[-1])
# lv_CEsBU2_text = '$lv_{{CEsBU2}}$ = {:d}\n'.format(lv_object)
# TotalSamples_CEsBU1_list = '$TotalSamples_{{CEsBU1}}$ = {:d}\n'.format(TotalSamples_CEsBU1)
# # mu_final_CEsBU2 = np.mean(X_resample_object)
# a_CEsBU2 = np.zeros((T_final,N_CE))
# for i in range(T_final):
#     t = int(i+1)
#     a_CEsBU2[i,:] = a(X_resample_object,t)

# mu_a_CEsBU2 = np.mean(a_CEsBU2,axis=1)
# si_a_CEsBU2 = np.std(a_CEsBU2,axis=1)

# # Credible interval
# alpha = 0.05
# lower_a_CEsBU2 = mu_a_CEsBU2  - si_a_CEsBU2*sp.stats.norm.ppf(1-alpha/2)
# upper_a_CEsBU2 = mu_a_CEsBU2  + si_a_CEsBU2*sp.stats.norm.ppf(1-alpha/2)
# # Posterior distribution
# mu_CEsBU2 = np.zeros((dim,1));si_CEsBU2 = np.zeros((dim,1))
# for d in range(dim):
#     mu_CEsBU2[d] = np.mean(X_resample_object[d,:])
#     si_CEsBU2[d] = np.std(X_resample_object[d,:])
# # text
# mu_CEsBU2_text = '$mu_{{CEsBU2}}$ = {:.3f}\n'.format(mu_CEsBU2[0,0]);si_CEsBU2_text = 'si_CEsBU2 = {}\n'.format(si_CEsBU2[0,0])
# Z_object_CEsBU2_text = '$Z_{{object_{{CEsBU2}}}}$= {:.3e}\n'.format(cE_object)
# # Plot CrackLength a_True, a_M,a_CEsBU2
# fig3,ax3 = plt.subplots(figsize=(8,6),dpi=400)
# ax3.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
# ax3.plot(t_m,a_data,'ro',label='$a_M$')
# ax3.plot(t_m,mu_a_CEsBU2,'k+-',label='$mu_{a_{CEsBU2}}$')
# ax3.fill_between(t_m,upper_a_CEsBU2,lower_a_CEsBU2,alpha = 0.2,color='black',label='$C.I._{a_{CEsBU2}}$',ls=':')
# ax3.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
# s = Z_object_ref_text+Z_object_CEsBU2_text+nESS_CEsBU2_final_text+lv_CEsBU2_text
# ax3.text(0.2,0.8,s,horizontalalignment='left',verticalalignment='center', transform = ax3.transAxes,size=22)
# #     ax3.text(0.5,0.8,s,horizontalalignment='left',verticalalignment='center', transform = ax3.transAxes,size=15)
# ax3.set_xlabel('$t_{m}$(year)')
# ax3.set_ylabel('a CrackLength (mm)')
# ax3.set_title('$a_{{True}},a_M,a_{{CEsBU2}} ; T_{{object}}$ = {:d}'.format(T_object))
# ax3.legend(bbox_to_anchor=(0.01,1), loc="upper left")
# if method2 == 'CEsBU_SG':
#     fig3.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_CEsBU2 with T_object = {}.png'.format(T_object),dpi=400)
# if method2 == 'CEsBU_SG_log':
#     fig3.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_CEsBU2 with T_object = {}.png'.format(T_object),dpi=400)
# # Plot CrackLength a_True, a_M,a_CEsBU1,2
# fig4,ax4 = plt.subplots(figsize=(8,6),dpi=400)
# ax4.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
# ax4.plot(t_m,a_data,'ro',label='$a_M$')
# ax4.plot(t_m,mu_a_CEsBU1,'m+-',label='$mu_{a_{CEsBU1}}$')
# ax4.fill_between(t_m,upper_a_CEsBU1,lower_a_CEsBU1,alpha = 0.2,color='magenta',label='$C.I._{a_{CEsBU1}}$',ls=':')
# ax4.plot(t_m,mu_a_CEsBU2,'k+-',label='$mu_{a_{CEsBU2}}$')
# ax4.fill_between(t_m,upper_a_CEsBU2,lower_a_CEsBU2,alpha = 0.2,color='black',label='$C.I._{a_{CEsBU2}}$',ls=':')
# ax4.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
# s = Z_final_ref_text+Z_final_CEsBU1_text+Z_object_ref_text+Z_object_CEsBU2_text
# ax4.text(0.2,0.8,s,horizontalalignment='left',verticalalignment='center', transform = ax4.transAxes,size=22)
# #ax4.text(0.5,0.8,s,horizontalalignment='left',verticalalignment='center', transform = ax4.transAxes)
# ax4.set_xlabel('$t_{m}$(year)')
# ax4.set_ylabel('a CrackLength (mm)')
# ax4.set_title('$a_{{True}},a_M,a_{{CEsBU1,2}} ; T_{{object}}$ = {}'.format(T_object))
# ax4.legend(bbox_to_anchor=(0.01,1), loc="upper left")
# if method2 == 'CEsBU_SG':
#     fig4.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_CEsBU1,2 with T_object = {}.png'.format(T_object),dpi=400)
# if method2 == 'CEsBU_SG_log':
# #     fig4.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_CEsBU2.png',dpi=400)
#     fig4.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_CEsBU1,2 with T_object = {}.png'.format(T_object),dpi=400)
# #=====================================================================================================================================================================================================================================================================================
# # Plot InterDist for CEsBU2
# fig5, ax5 = plt.subplots(1, 1, figsize=(5, 4))
# sns.distplot(samplesX_object[0],label='$dist_{prior}$')
# for sample in samplesX_object[1:-1]:
#     sns.distplot(sample)
# sns.distplot(samplesX_object[-1],label='$dist_{{CEsBU2}}$,lv={}'.format(lv_object))
# ax5.plot(mu_prior,0,'b^',label='$mu_{{prior}}$ ')
# ax5.plot(x_given,0,'bv',label='$x_{{given}}$')
# ax5.plot(mu_CEsBU2[0,0],0,'ro',label='$mu_{{CEsBU2}}$')
# s = Z_object_ref_text+Z_object_CEsBU2_text+mu_prior_text+x_given_text+mu_CEsBU2_text+nESS_CEsBU2_final_text
# ax5.text(0.35,0.7,s,horizontalalignment='left',verticalalignment='center', transform=ax5.transAxes,bbox=dict(facecolor='none',edgecolor='none', alpha=0.5),size=12)
# ax5.set_xlabel('x')
# ax5.set_ylabel('PDF')
# ax5.set_title('InterDist. of CEsBU2 for RV DS in X space with'+T_object_text)
# ax5.legend(bbox_to_anchor=(0.7,1), loc="upper left")
        
# #fig2.suptitle('InterDist. of CEsBU1 for RV DS in X space with'+T_final_text, fontsize=16)
# if method2 == "CEsBU_SG":
#     fig5.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\InterDist\\CEsBU2 with T_obejct = {} in X space.png'.format(T_object),dpi=400)
# elif method2 == "CEsBU_SG_log":
#     fig5.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\InterDist\\CEsBU2 with T_object = {} in X space.png'.format(T_object),dpi=400)
# print('beta_t:',beta_t_object[0:lv_object])
# print('mu_U: ',mu_U_list_object)
# print('si_U: ',si_U_list_object)
# print('normalisation constant cE: ',cE_object)
# print('reference normalisation constant Z: ',constant_objectyear[0])
# print('final normalised effective sample size : ',nESS_list_object[-1])
# #=============================================================================
# # compare the three sequential steps
# if method1 == "CEsBU_SG" : 
#     [U_object1,X_object1,Z_object1] = Sequential_step1([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,nESS_list_final,Leval_allyear_list_final)
#     [U_object2,X_object2,Z_object2] = Sequential_step2([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,Leval_allyear_list_final)
#     [U_object3,X_object3,Z_object3] = Sequential_step3([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,nESS_list_final,Leval_allyear_list_final)
#     [U_object4,X_object4,Z_object4] = Sequential_step4([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,Leval_allyear_list_final)

# elif method1 == "CEsBU_SG_log" : 
#     [U_object1,X_object1,Z_object1] = Sequential_step1_log([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,nESS_list_final,lnLeval_allyear_list_final)
#     [U_object2,X_object2,Z_object2] = Sequential_step2_log([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,lnLeval_allyear_list_final)
#     [U_object3,X_object3,Z_object3] = Sequential_step3_log([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,nESS_list_final,lnLeval_allyear_list_final)
#     [U_object4,X_object4,Z_object4] = Sequential_step4_log([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,lnLeval_allyear_list_final)
# # analyse the sequential result
# mu_SS1 = np.mean(X_object1)
# mu_SS2 = np.mean(X_object2)
# mu_SS3 = np.mean(X_object3)
# mu_SS4 = np.mean(X_object4)
# si_SS1 = np.std(X_object1)
# si_SS2 = np.std(X_object2)
# si_SS3 = np.std(X_object3)
# si_SS4 = np.std(X_object4)
# a_SS1 = np.zeros((T_final,N_CE))
# a_SS2 = np.zeros((T_final,N_CE))
# a_SS3 = np.zeros((T_final,N_CE))
# a_SS4 = np.zeros((T_final,N_CE))
# for i in range(T_final):
#     t = int(i+1)
#     a_SS1[i,:] = a(X_object1,t)
#     a_SS2[i,:] = a(X_object2,t)
#     a_SS3[i,:] = a(X_object3,t)
#     a_SS4[i,:] = a(X_object4,t)
# mu_a_SS1 = np.mean(a_SS1,axis=1)
# mu_a_SS2 = np.mean(a_SS2,axis=1)
# mu_a_SS3 = np.mean(a_SS3,axis=1)
# mu_a_SS4 = np.mean(a_SS4,axis=1)
# si_a_SS1 = np.std(a_SS1,axis=1)
# si_a_SS2 = np.std(a_SS2,axis=1)
# si_a_SS3 = np.std(a_SS3,axis=1)
# si_a_SS4 = np.std(a_SS4,axis=1)
# # Credible interval
# alpha = 0.05
# lower_a_SS1 = mu_a_SS1 - si_a_SS1*sp.stats.norm.ppf(1-alpha/2)
# upper_a_SS1 = mu_a_SS1 + si_a_SS1*sp.stats.norm.ppf(1-alpha/2)
# lower_a_SS2 = mu_a_SS2 - si_a_SS2*sp.stats.norm.ppf(1-alpha/2)
# upper_a_SS2 = mu_a_SS2 + si_a_SS2*sp.stats.norm.ppf(1-alpha/2)
# lower_a_SS3 = mu_a_SS3 - si_a_SS3*sp.stats.norm.ppf(1-alpha/2)
# upper_a_SS3 = mu_a_SS3 + si_a_SS3*sp.stats.norm.ppf(1-alpha/2)
# lower_a_SS4 = mu_a_SS4 - si_a_SS4*sp.stats.norm.ppf(1-alpha/2)
# upper_a_SS4 = mu_a_SS4 + si_a_SS4*sp.stats.norm.ppf(1-alpha/2)
# # Text
# Z_SS1_text = '$Z_{{object_{{SS1}}}}$= {:.3e}\n '.format(Z_object1)
# Z_SS2_text = '$Z_{{object_{{SS2}}}}$= {:.3e}\n'.format(Z_object2)
# Z_SS3_text = '$Z_{{object_{{SS3}}}}$= {:.3e}\n '.format(Z_object3)
# Z_SS4_text = '$Z_{{object_{{SS4}}}}$= {:.3e}\n '.format(Z_object4)
# mu_SS1_text = '$mu_{{SS1}}$= {:.3f}\n '.format(mu_SS1)
# mu_SS2_text = '$mu_{{SS2}}$= {:.3f}\n '.format(mu_SS2)
# mu_SS3_text = '$mu_{{SS3}}$= {:.3f}\n '.format(mu_SS3)
# mu_SS4_text = '$mu_{{SS4}}$= {:.3f}\n '.format(mu_SS4)
# si_SS1_text = '$si_{{SS1}}$= {:.3f}\n '.format(si_SS1)
# si_SS2_text = '$si_{{SS2}}$= {:.3f}\n '.format(si_SS2)
# si_SS3_text = '$si_{{SS3}}$= {:.3f}\n '.format(si_SS3)
# si_SS4_text = '$si_{{SS4}}$= {:.3f}\n '.format(si_SS4)
# mu_a_SS1_object_text = '$mu_{{a_{{SS1_{{object}}}}}}$= {:.3f}\n '.format(mu_a_SS1[T_object-1])
# mu_a_SS2_object_text = '$mu_{{a_{{SS2_{{object}}}}}}$= {:.3f}\n '.format(mu_a_SS2[T_object-1])
# mu_a_SS3_object_text = '$mu_{{a_{{SS3_{{object}}}}}}$= {:.3f}\n '.format(mu_a_SS3[T_object-1])
# mu_a_SS4_object_text = '$mu_{{a_{{SS4_{{object}}}}}}$= {:.3f}\n '.format(mu_a_SS4[T_object-1])
# si_a_SS1_object_text = '$si_{{a_{{SS1_{{object}}}}}}$= {:.3f}\n '.format(si_a_SS1[T_object-1])
# si_a_SS2_object_text = '$si_{{a_{{SS2_{{object}}}}}}$= {:.3f}\n '.format(si_a_SS2[T_object-1])
# si_a_SS3_object_text = '$si_{{a_{{SS3_{{object}}}}}}$= {:.3f}\n '.format(si_a_SS3[T_object-1])
# si_a_SS4_object_text = '$si_{{a_{{SS4_{{object}}}}}}$= {:.3f}\n '.format(si_a_SS4[T_object-1])
# # Plot a_True, a_M,a_SS1
# fig6,ax6 = plt.subplots(figsize=(8,6),dpi=400)
# # plt.figure(figsize=(8,6),dpi=400)
# ax6.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
# ax6.plot(t_m,a_data,'ro',label='$a_M$')
# ax6.plot(t_m,mu_a_SS1,'b+',label='$mu_{a_{SS1}}$')
# ax6.fill_between(t_m,upper_a_SS1,lower_a_SS1,alpha = 0.2,color='blue',label='$C.I._{a_{SS1}}$',ls=':')
# s = Z_object_ref_text+Z_object_CEsBU2_text+Z_SS1_text
# ax6.text(0.2,0.8,s,horizontalalignment='left',verticalalignment='center', transform = ax6.transAxes,size='20')
# ax6.set_xlabel('$t_{m}$(year)')
# ax6.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
# ax6.set_ylabel('a CrackLength (mm)')
# ax6.set_title('$a_{{True}},a_M,a_{{SS1}}; T_{{object}}$ = {}'.format(T_object))
# ax6.legend(bbox_to_anchor=(0.01,1), loc="upper left")
# # ax6.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
# if method2 == 'CEsBU_SG':
#     fig6.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_SS1 with T_object = {}.png'.format(T_object),dpi=400)
# if method2 == 'CEsBU_SG_log':
#     fig6.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_SS1 with T_object = {}.png'.format(T_object),dpi=400)
    
# # Plot a_True, a_M,a_SS2
# fig7,ax7 = plt.subplots(figsize=(8,6),dpi=400)

# ax7.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
# ax7.plot(t_m,a_data,'ro',label='$a_M$')
# ax7.plot(t_m,mu_a_SS2,'g+',label='$mu_{a_{SS2}}$')
# ax7.fill_between(t_m,upper_a_SS2,lower_a_SS2,alpha = 0.2,color='green',label='$C.I._{a_{SS2}}$',ls=':')
# s = Z_object_ref_text+Z_object_CEsBU2_text+Z_SS2_text
# ax7.text(0.2,0.72,s,horizontalalignment='left',verticalalignment='center', transform = ax7.transAxes,size=22)
# #ax7.text(0.3,0.68,s,horizontalalignment='left',verticalalignment='center', transform = ax7.transAxes)
# ax7.set_xlabel('$t_{m}$(year)')
# ax7.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
# ax7.set_ylabel('a CrackLength (mm)')
# ax7.set_title('$a_{{True}},a_M,a_{{SS2}}; T_{{object}}$ = {}'.format(T_object))
# ax7.legend(bbox_to_anchor=(0.01,1), loc="upper left")
# # ax7.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
# if method2 == 'CEsBU_SG':
#     fig7.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_SS2 with T_object = {}.png'.format(T_object),dpi=400)
# if method2 == 'CEsBU_SG_log':
#     fig7.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_SS2 with T_object = {}.png'.format(T_object),dpi=400)
    
# # Plot a_True, a_M,a_SS3
# fig8,ax8 = plt.subplots(figsize=(8,6),dpi=400)
# ax8.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
# ax8.plot(t_m,a_data,'ro',label='$a_M$')
# ax8.plot(t_m,mu_a_SS3,'y+',label='$mu_{a_{SS3}}$')
# ax8.fill_between(t_m,upper_a_SS3,lower_a_SS3,alpha = 0.2,color='yellow',label='$C.I._{a_{SS3}}$',ls=':')
# s = Z_object_ref_text+Z_object_CEsBU2_text+Z_SS3_text
# ax8.text(0.2,0.72,s,horizontalalignment='left',verticalalignment='center', transform = ax8.transAxes,size=22)
# #ax8.text(0.3,0.68,s,horizontalalignment='left',verticalalignment='center', transform = ax8.transAxes)
# ax8.set_xlabel('$t_{m}$(year)')
# ax8.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
# ax8.set_ylabel('a CrackLength (mm)')
# ax8.set_title('$a_{{True}},a_M,a_{{SS3}}; T_{{object}}$ = {}'.format(T_object))
# ax8.legend(bbox_to_anchor=(0.01,1), loc="upper left")
# if method2 == 'CEsBU_SG':
#     fig8.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_SS3 with T_object = {}.png'.format(T_object),dpi=400)
# if method2 == 'CEsBU_SG_log':
#     fig8.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_SS3 with T_object = {}.png'.format(T_object),dpi=400)
# # Plot a_True, a_M,a_SS4
# fig20,ax20 = plt.subplots(figsize=(8,6),dpi=400)
# ax20.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
# ax20.plot(t_m,a_data,'ro',label='$a_M$')
# ax20.plot(t_m,mu_a_SS4,'r+',label='$mu_{a_{SS4}}$')
# ax20.fill_between(t_m,upper_a_SS4,lower_a_SS4,alpha = 0.2,color='red',label='$C.I._{a_{SS4}}$',ls=':')
# s = Z_object_ref_text+Z_object_CEsBU2_text+Z_SS4_text
# ax20.text(0.2,0.72,s,horizontalalignment='left',verticalalignment='center', transform = ax20.transAxes,size=22)
# #ax20.text(0.3,0.68,s,horizontalalignment='left',verticalalignment='center', transform = ax20.transAxes)
# ax20.set_xlabel('$t_{m}$(year)')
# ax20.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
# ax20.set_ylabel('a CrackLength (mm)')
# ax20.set_title('$a_{{True}},a_M,a_{{SS4}}; T_{{object}}$ = {}'.format(T_object))
# ax20.legend(bbox_to_anchor=(0.01,1), loc="upper left")
# if method2 == 'CEsBU_SG':
#     fig20.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_SS4 with T_object = {}.png'.format(T_object),dpi=400)
# if method2 == 'CEsBU_SG_log':
#     fig20.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_SS4 with T_object = {}.png'.format(T_object),dpi=400)

# # Plot a_True, a_M,a_SS1,2,3,4
# fig9,ax9 = plt.subplots(figsize=(8,6),dpi=400)
# ax9.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
# ax9.plot(t_m,a_data,'ro',label='$a_M$')
# ax9.plot(t_m,mu_a_SS1,'b+',label='$mu_{a_{SS1}}$')
# ax9.fill_between(t_m,upper_a_SS1,lower_a_SS1,alpha = 0.2,color='blue',label='$C.I._{a_{SS1}}$',ls=':')
# ax9.plot(t_m,mu_a_SS2,'g+',label='$mu_{a_{SS2}}$')
# ax9.fill_between(t_m,upper_a_SS2,lower_a_SS2,alpha = 0.2,color='green',label='$C.I._{a_{SS2}}$',ls=':')
# ax9.plot(t_m,mu_a_SS3,'y+',label='$mu_{a_{SS3}}$')
# ax9.fill_between(t_m,upper_a_SS3,lower_a_SS3,alpha = 0.2,color='yellow',label='$C.I._{a_{SS3}}$',ls=':')
# ax9.plot(t_m,mu_a_SS4,'r+',label='$mu_{a_{SS4}}$')
# ax9.fill_between(t_m,upper_a_SS4,lower_a_SS4,alpha = 0.2,color='red',label='$C.I._{a_{SS4}}$',ls=':')

# s = Z_object_ref_text+Z_object_CEsBU2_text+Z_SS1_text+Z_SS2_text+Z_SS3_text+Z_SS4_text
# ax9.text(0.2,0.7,s,horizontalalignment='left',verticalalignment='center', transform = ax9.transAxes,size='20')
# ax9.set_xlabel('$t_{m}$(year)')
# ax9.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
# ax9.set_ylabel('a CrackLength (mm)')
# ax9.set_title('$a_{{True}},a_M,a_{{SS1,2,3,4}}; T_{{object}}$ = {}'.format(T_object))
# ax9.legend(bbox_to_anchor=(0.01,1), loc="upper left")
# # ax9.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
# if method2 == 'CEsBU_SG':
#     fig9.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_SS1,2,3,4 with T_object = {}.png'.format(T_object),dpi=400)
# if method2 == 'CEsBU_SG_log':
#     fig9.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_SS1,2,3,4 with T_object = {}.png'.format(T_object),dpi=400)

# # Plot a_True, a_M,a_CEsBU1,2,a_SS1,2,3,4
# fig10,ax10 = plt.subplots(figsize=(8,6),dpi=400)


# ax10.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
# ax10.plot(t_m,a_data,'ro',label='$a_M$')

# ax10.plot(t_m,mu_a_CEsBU1,'m+-',label='$mu_{a_{CEsBU1}}$')
# ax10.fill_between(t_m,upper_a_CEsBU1,lower_a_CEsBU1,alpha = 0.2,color='magenta',label='$C.I._{a_{CEsBU1}}$',ls=':')
# ax10.plot(t_m,mu_a_CEsBU2,'k+-',label='$mu_{a_{CEsBU2}}$')
# ax10.fill_between(t_m,upper_a_CEsBU2,lower_a_CEsBU2,alpha = 0.2,color='black',label='$C.I._{a_{CEsBU2}}$',ls=':')

# ax10.plot(t_m,mu_a_SS1,'b+',label='$mu_{a_{SS1}}$')
# ax10.fill_between(t_m,upper_a_SS1,lower_a_SS1,alpha = 0.2,color='blue',label='$C.I._{a_{SS1}}$',ls=':')
# ax10.plot(t_m,mu_a_SS2,'g+',label='$mu_{a_{SS2}}$')
# ax10.fill_between(t_m,upper_a_SS2,lower_a_SS2,alpha = 0.2,color='green',label='$C.I._{a_{SS2}}$',ls=':')
# ax10.plot(t_m,mu_a_SS3,'y+',label='$mu_{a_{SS3}}$')
# ax10.fill_between(t_m,upper_a_SS3,lower_a_SS3,alpha = 0.2,color='yellow',label='$C.I._{a_{SS3}}$',ls=':')
# ax10.plot(t_m,mu_a_SS4,'r+',label='$mu_{a_{SS4}}$')
# ax10.fill_between(t_m,upper_a_SS4,lower_a_SS4,alpha = 0.2,color='red',label='$C.I._{a_{SS4}}$',ls=':')

# s = Z_final_ref_text+Z_final_CEsBU1_text+Z_object_ref_text+Z_object_CEsBU2_text+Z_SS1_text+Z_SS2_text+Z_SS3_text+Z_SS4_text
# #ax10.text(0.2,0.6,s,horizontalalignment='left',verticalalignment='center', transform = ax10.transAxes,size=22)
# ax10.set_xlabel('$t_{m}$(year)',fontsize = 18)
# ax10.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
# ax10.set_ylabel('a CrackLength (mm)',fontsize = 18)
# ax10.set_title('Crack Length evolution Prediction for $T_{{object}}$ = {}'.format(T_object),fontsize = 18)

# ax10.legend(bbox_to_anchor=(0,1.01), loc="upper left",fontsize = 14)

# if method1 == 'CEsBU_SG':
#     fig10.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_CEsBU1,2,a_SS1,2,3,4 with T_object = {}.png'.format(T_object),dpi=400)
# if method1 == 'CEsBU_SG_log':
#     fig10.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_CEsBU1,2,a_SS1,2,3,4 with T_object = {}.png'.format(T_object),dpi=400)
# # Plot Posterior Dist for  CEsBU1,2; SS1,2,3
# fig11,ax11 = plt.subplots(figsize=(5,4),dpi=400)
# sns.distplot(X_resample_final,label='$dist_{T_{final_{CEsBU1}}}$')
# sns.distplot(X_resample_object,label='$dist_{T_{object_{CEsBU2}}}$')
# sns.distplot(X_object1,label='$dist_{T_{object_{SS1}}}$')
# sns.distplot(X_object2,label='$dist_{T_{object_{SS2}}}$')
# sns.distplot(X_object3,label='$dist_{T_{object_{SS3}}}$')
# sns.distplot(X_object4,label='$dist_{T_{object_{SS4}}}$')
# ax11.plot([mu_prior],[0],'b^',label='$mu_{prior}$')
# ax11.plot([x_given],[0],'bv',label='$x_{given}$')
# s = mu_prior_text+x_given_text+mu_CEsBU1_text+mu_CEsBU2_text+mu_SS1_text+mu_SS2_text+mu_SS3_text+mu_SS4_text
# ax11.text(0.35,0.68,s,horizontalalignment='left',verticalalignment='center', transform = ax11.transAxes)
# ax11.set_xlabel('DS')
# ax11.set_ylabel('PDF')
# ax11.set_title('Filtering distribution plot for $T_{{object}}$ = {}'.format(T_object))
# ax11.legend(bbox_to_anchor=(0,1), loc="upper left")
# if method1 == 'CEsBU_SG':
#     fig11.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\InterDist\\CEsBU1,2,SS1,2,3,4 with T_object = {}.png'.format(T_object),dpi=400)
# if method1 == 'CEsBU_SG_log':
#     fig11.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\InterDist\\CEsBU1,2,SS1,2,3,4 with T_object = {}.png'.format(T_object),dpi=400)
    
















# #======================================================================================================================================
# use CEsBU to solve the filter distribution at year T_object here we are only interested in U_final X_final
T_object_array = np.arange(5,50,5)
for T_object in T_object_array:
    # T_object = 20
    T_object_text = '$T_{{object}}$ = {:d}\n'.format(T_object)
    def target_objectyear(x):
        return prior_pdf(x)*Likelihood_objectyear(x)
    constant_objectyear = sp.integrate.fixed_quad(target_objectyear,lower,upper,n=1000)
    Z_object_ref_text = '$Z_{{object_{{ref}}}}$= {:.3e}\n'.format(constant_objectyear[0])
    print('\n......using CEsBU2 to solve the filter distr. at year T_object: {}'.format(T_object))
    N_CE = 1000
    N_object = 1000
    max_it = 50
    CV_target = 1.5
    time.tic()
    if method2 == 'CEsBU_SG':
        [cE_object,lv_object,samplesU_object,samplesX_object,U_resample_object,X_resample_object,beta_t_object,mu_U_list_object,si_U_list_object,nESS_list_object,Leval_allyear_list_object] = CEsBU_SG(N_CE,N_object,Likelihood,T_object,[dist_prior],max_it,CV_target)
    if method2 == 'CEsBU_SG_log':
        [cE_object,lv_object,samplesU_object,samplesX_object,U_resample_object,X_resample_object,beta_t_object,mu_U_list_object,si_U_list_object,nESS_list_object,Leval_allyear_list_object] = CEsBU_SG_log(N_CE,N_object,log_Likelihood,T_object,[dist_prior],max_it,CV_target)
    t_CEsBU2 = time.tocvalue()
    # analyse the CEsBU2 result
    TotalSamples_CEsBU1 = (lv_object-1)*N_CE+N_object
    t_CEsBU2_text = '$t_{{CEsBU2}}$= {:.3f}\n'.format(t_CEsBU2)
    nESS_CEsBU2_final_text = '$nESS_{{CEsBU2_{{final}}}}$= {:.3f}\n'.format(nESS_list_object[-1])
    lv_CEsBU2_text = '$lv_{{CEsBU2}}$ = {:d}\n'.format(lv_object)
    TotalSamples_CEsBU1_list = '$TotalSamples_{{CEsBU1}}$ = {:d}\n'.format(TotalSamples_CEsBU1)
    # mu_final_CEsBU2 = np.mean(X_resample_object)
    a_CEsBU2 = np.zeros((T_final,N_CE))
    for i in range(T_final):
        t = int(i+1)
        a_CEsBU2[i,:] = a(X_resample_object,t)

    mu_a_CEsBU2 = np.mean(a_CEsBU2,axis=1)
    si_a_CEsBU2 = np.std(a_CEsBU2,axis=1)

    # Credible interval
    alpha = 0.05
    lower_a_CEsBU2 = mu_a_CEsBU2  - si_a_CEsBU2*sp.stats.norm.ppf(1-alpha/2)
    upper_a_CEsBU2 = mu_a_CEsBU2  + si_a_CEsBU2*sp.stats.norm.ppf(1-alpha/2)
    # Posterior distribution
    mu_CEsBU2 = np.zeros((dim,1));si_CEsBU2 = np.zeros((dim,1))
    for d in range(dim):
        mu_CEsBU2[d] = np.mean(X_resample_object[d,:])
        si_CEsBU2[d] = np.std(X_resample_object[d,:])
    # text
    mu_CEsBU2_text = '$mu_{{CEsBU2}}$ = {:.3f}\n'.format(mu_CEsBU2[0,0]);si_CEsBU2_text = 'si_CEsBU2 = {}\n'.format(si_CEsBU2[0,0])
    Z_object_CEsBU2_text = '$Z_{{object_{{CEsBU2}}}}$= {:.3e}\n'.format(cE_object)
    # Plot CrackLength a_True, a_M,a_CEsBU2
    fig3,ax3 = plt.subplots(figsize=(8,6),dpi=400)
    ax3.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
    ax3.plot(t_m,a_data,'ro',label='$a_M$')
    ax3.plot(t_m,mu_a_CEsBU2,'k+-',label='$mu_{a_{CEsBU2}}$')
    ax3.fill_between(t_m,upper_a_CEsBU2,lower_a_CEsBU2,alpha = 0.2,color='black',label='$C.I._{a_{CEsBU2}}$',ls=':')
    ax3.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
    s = Z_object_ref_text+Z_object_CEsBU2_text+nESS_CEsBU2_final_text+lv_CEsBU2_text
    ax3.text(0.2,0.8,s,horizontalalignment='left',verticalalignment='center', transform = ax3.transAxes,size=22)
#     ax3.text(0.5,0.8,s,horizontalalignment='left',verticalalignment='center', transform = ax3.transAxes,size=15)
    ax3.set_xlabel('$t_{m}$(year)')
    ax3.set_ylabel('a CrackLength (mm)')
    ax3.set_title('$a_{{True}},a_M,a_{{CEsBU2}} ; T_{{object}}$ = {:d}'.format(T_object))
    ax3.legend(bbox_to_anchor=(0.01,1), loc="upper left")
    if method2 == 'CEsBU_SG':
        fig3.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_CEsBU2 with T_object = {}.png'.format(T_object),dpi=400)
    if method2 == 'CEsBU_SG_log':
        fig3.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_CEsBU2 with T_object = {}.png'.format(T_object),dpi=400)
    # Plot CrackLength a_True, a_M,a_CEsBU1,2
    fig4,ax4 = plt.subplots(figsize=(8,6),dpi=400)
    ax4.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
    ax4.plot(t_m,a_data,'ro',label='$a_M$')
    ax4.plot(t_m,mu_a_CEsBU1,'m+-',label='$mu_{a_{CEsBU1}}$')
    ax4.fill_between(t_m,upper_a_CEsBU1,lower_a_CEsBU1,alpha = 0.2,color='magenta',label='$C.I._{a_{CEsBU1}}$',ls=':')
    ax4.plot(t_m,mu_a_CEsBU2,'k+-',label='$mu_{a_{CEsBU2}}$')
    ax4.fill_between(t_m,upper_a_CEsBU2,lower_a_CEsBU2,alpha = 0.2,color='black',label='$C.I._{a_{CEsBU2}}$',ls=':')
    ax4.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
    s = Z_final_ref_text+Z_final_CEsBU1_text+Z_object_ref_text+Z_object_CEsBU2_text
    ax4.text(0.2,0.8,s,horizontalalignment='left',verticalalignment='center', transform = ax4.transAxes,size=22)
    ax4.set_xlabel('$t_{m}$(year)')
    ax4.set_ylabel('a CrackLength (mm)')
    ax4.set_title('$a_{{True}},a_M,a_{{CEsBU1,2}} ; T_{{object}}$ = {}'.format(T_object))
    ax4.legend(bbox_to_anchor=(0.01,1), loc="upper left")
    if method2 == 'CEsBU_SG':
        fig4.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_CEsBU1,2 with T_object = {}.png'.format(T_object),dpi=400)
    if method2 == 'CEsBU_SG_log':
    #     fig4.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_CEsBU2.png',dpi=400)
        fig4.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_CEsBU1,2 with T_object = {}.png'.format(T_object),dpi=400)
    #=====================================================================================================================================================================================================================================================================================
    # Plot InterDist for CEsBU2
    fig5, ax5 = plt.subplots(1, 1, figsize=(5, 4))
    sns.distplot(samplesX_object[0],label='$dist_{prior}$')
    for sample in samplesX_object[1:-1]:
        sns.distplot(sample)
    sns.distplot(samplesX_object[-1],label='$dist_{{CEsBU2}}$,lv={}'.format(lv_object))
    ax5.plot(mu_prior,0,'b^',label='$mu_{{prior}}$ ')
    ax5.plot(x_given,0,'bv',label='$x_{{given}}$')
    ax5.plot(mu_CEsBU2[0,0],0,'ro',label='$mu_{{CEsBU2}}$')
    s = Z_object_ref_text+Z_object_CEsBU2_text+mu_prior_text+x_given_text+mu_CEsBU2_text+nESS_CEsBU2_final_text
    ax5.text(0.35,0.7,s,horizontalalignment='left',verticalalignment='center', transform=ax5.transAxes,bbox=dict(facecolor='none',edgecolor='none', alpha=0.5),size=12)
    ax5.set_xlabel('x')
    ax5.set_ylabel('PDF')
    ax5.set_title('InterDist. of CEsBU2 for RV DS in X space with'+T_object_text)
    ax5.legend(bbox_to_anchor=(0.7,1), loc="upper left")
          
    #fig2.suptitle('InterDist. of CEsBU1 for RV DS in X space with'+T_final_text, fontsize=16)
    if method2 == "CEsBU_SG":
        fig5.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\InterDist\\CEsBU2 with T_obejct = {} in X space.png'.format(T_object),dpi=400)
    elif method2 == "CEsBU_SG_log":
        fig5.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\InterDist\\CEsBU2 with T_object = {} in X space.png'.format(T_object),dpi=400)
    print('beta_t:',beta_t_object[0:lv_object])
    print('mu_U: ',mu_U_list_object)
    print('si_U: ',si_U_list_object)
    print('normalisation constant cE: ',cE_object)
    print('reference normalisation constant Z: ',constant_objectyear[0])
    print('final normalised effective sample size : ',nESS_list_object[-1])
    #=============================================================================
    # compare the three sequential steps
    if method1 == "CEsBU_SG" : 
        [U_object1,X_object1,Z_object1] = Sequential_step1([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,nESS_list_final,Leval_allyear_list_final)
        [U_object2,X_object2,Z_object2] = Sequential_step2([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,Leval_allyear_list_final)
        [U_object3,X_object3,Z_object3] = Sequential_step3([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,nESS_list_final,Leval_allyear_list_final)
        [U_object4,X_object4,Z_object4] = Sequential_step4([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,Leval_allyear_list_final)

    elif method1 == "CEsBU_SG_log" : 
        [U_object1,X_object1,Z_object1] = Sequential_step1_log([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,nESS_list_final,lnLeval_allyear_list_final)
        [U_object2,X_object2,Z_object2] = Sequential_step2_log([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,lnLeval_allyear_list_final)
        [U_object3,X_object3,Z_object3] = Sequential_step3_log([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,nESS_list_final,lnLeval_allyear_list_final)
        [U_object4,X_object4,Z_object4] = Sequential_step4_log([dist_prior],T_object,lv_final,samplesU_final,mu_U_list_final,si_U_list_final,lnLeval_allyear_list_final)
    # analyse the sequential result
    mu_SS1 = np.mean(X_object1)
    mu_SS2 = np.mean(X_object2)
    mu_SS3 = np.mean(X_object3)
    mu_SS4 = np.mean(X_object4)
    si_SS1 = np.std(X_object1)
    si_SS2 = np.std(X_object2)
    si_SS3 = np.std(X_object3)
    si_SS4 = np.std(X_object4)
    a_SS1 = np.zeros((T_final,N_CE))
    a_SS2 = np.zeros((T_final,N_CE))
    a_SS3 = np.zeros((T_final,N_CE))
    a_SS4 = np.zeros((T_final,N_CE))
    for i in range(T_final):
        t = int(i+1)
        a_SS1[i,:] = a(X_object1,t)
        a_SS2[i,:] = a(X_object2,t)
        a_SS3[i,:] = a(X_object3,t)
        a_SS4[i,:] = a(X_object4,t)
    mu_a_SS1 = np.mean(a_SS1,axis=1)
    mu_a_SS2 = np.mean(a_SS2,axis=1)
    mu_a_SS3 = np.mean(a_SS3,axis=1)
    mu_a_SS4 = np.mean(a_SS4,axis=1)
    si_a_SS1 = np.std(a_SS1,axis=1)
    si_a_SS2 = np.std(a_SS2,axis=1)
    si_a_SS3 = np.std(a_SS3,axis=1)
    si_a_SS4 = np.std(a_SS4,axis=1)
    # Credible interval
    alpha = 0.05
    lower_a_SS1 = mu_a_SS1 - si_a_SS1*sp.stats.norm.ppf(1-alpha/2)
    upper_a_SS1 = mu_a_SS1 + si_a_SS1*sp.stats.norm.ppf(1-alpha/2)
    lower_a_SS2 = mu_a_SS2 - si_a_SS2*sp.stats.norm.ppf(1-alpha/2)
    upper_a_SS2 = mu_a_SS2 + si_a_SS2*sp.stats.norm.ppf(1-alpha/2)
    lower_a_SS3 = mu_a_SS3 - si_a_SS3*sp.stats.norm.ppf(1-alpha/2)
    upper_a_SS3 = mu_a_SS3 + si_a_SS3*sp.stats.norm.ppf(1-alpha/2)
    lower_a_SS4 = mu_a_SS4 - si_a_SS4*sp.stats.norm.ppf(1-alpha/2)
    upper_a_SS4 = mu_a_SS4 + si_a_SS4*sp.stats.norm.ppf(1-alpha/2)
    # Text
    Z_SS1_text = '$Z_{{object_{{SS1}}}}$= {:.3e}\n '.format(Z_object1)
    Z_SS2_text = '$Z_{{object_{{SS2}}}}$= {:.3e}\n'.format(Z_object2)
    Z_SS3_text = '$Z_{{object_{{SS3}}}}$= {:.3e}\n '.format(Z_object3)
    Z_SS4_text = '$Z_{{object_{{SS4}}}}$= {:.3e}\n '.format(Z_object4)
    mu_SS1_text = '$mu_{{SS1}}$= {:.3f}\n '.format(mu_SS1)
    mu_SS2_text = '$mu_{{SS2}}$= {:.3f}\n '.format(mu_SS2)
    mu_SS3_text = '$mu_{{SS3}}$= {:.3f}\n '.format(mu_SS3)
    mu_SS4_text = '$mu_{{SS4}}$= {:.3f}\n '.format(mu_SS4)
    si_SS1_text = '$si_{{SS1}}$= {:.3f}\n '.format(si_SS1)
    si_SS2_text = '$si_{{SS2}}$= {:.3f}\n '.format(si_SS2)
    si_SS3_text = '$si_{{SS3}}$= {:.3f}\n '.format(si_SS3)
    si_SS4_text = '$si_{{SS4}}$= {:.3f}\n '.format(si_SS4)
    mu_a_SS1_object_text = '$mu_{{a_{{SS1_{{object}}}}}}$= {:.3f}\n '.format(mu_a_SS1[T_object-1])
    mu_a_SS2_object_text = '$mu_{{a_{{SS2_{{object}}}}}}$= {:.3f}\n '.format(mu_a_SS2[T_object-1])
    mu_a_SS3_object_text = '$mu_{{a_{{SS3_{{object}}}}}}$= {:.3f}\n '.format(mu_a_SS3[T_object-1])
    mu_a_SS4_object_text = '$mu_{{a_{{SS4_{{object}}}}}}$= {:.3f}\n '.format(mu_a_SS4[T_object-1])
    si_a_SS1_object_text = '$si_{{a_{{SS1_{{object}}}}}}$= {:.3f}\n '.format(si_a_SS1[T_object-1])
    si_a_SS2_object_text = '$si_{{a_{{SS2_{{object}}}}}}$= {:.3f}\n '.format(si_a_SS2[T_object-1])
    si_a_SS3_object_text = '$si_{{a_{{SS3_{{object}}}}}}$= {:.3f}\n '.format(si_a_SS3[T_object-1])
    si_a_SS4_object_text = '$si_{{a_{{SS4_{{object}}}}}}$= {:.3f}\n '.format(si_a_SS4[T_object-1])
    # Plot a_True, a_M,a_SS1
    fig6,ax6 = plt.subplots(figsize=(8,6),dpi=400)
    # plt.figure(figsize=(8,6),dpi=400)
    ax6.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
    ax6.plot(t_m,a_data,'ro',label='$a_M$')
    ax6.plot(t_m,mu_a_SS1,'b+',label='$mu_{a_{SS1}}$')
    ax6.fill_between(t_m,upper_a_SS1,lower_a_SS1,alpha = 0.2,color='blue',label='$C.I._{a_{SS1}}$',ls=':')
    s = Z_object_ref_text+Z_object_CEsBU2_text+Z_SS1_text
    ax6.text(0.2,0.8,s,horizontalalignment='left',verticalalignment='center', transform = ax6.transAxes,size='20')
    ax6.set_xlabel('$t_{m}$(year)')
    ax6.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
    ax6.set_ylabel('a CrackLength (mm)')
    ax6.set_title('$a_{{True}},a_M,a_{{SS1}}; T_{{object}}$ = {}'.format(T_object))
    ax6.legend(bbox_to_anchor=(0.01,1), loc="upper left")
    # ax6.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
    if method2 == 'CEsBU_SG':
        fig6.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_SS1 with T_object = {}.png'.format(T_object),dpi=400)
    if method2 == 'CEsBU_SG_log':
        fig6.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_SS1 with T_object = {}.png'.format(T_object),dpi=400)
        
    # Plot a_True, a_M,a_SS2
    fig7,ax7 = plt.subplots(figsize=(8,6),dpi=400)

    ax7.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
    ax7.plot(t_m,a_data,'ro',label='$a_M$')
    ax7.plot(t_m,mu_a_SS2,'g+',label='$mu_{a_{SS2}}$')
    ax7.fill_between(t_m,upper_a_SS2,lower_a_SS2,alpha = 0.2,color='green',label='$C.I._{a_{SS2}}$',ls=':')
    s = Z_object_ref_text+Z_object_CEsBU2_text+Z_SS2_text
    ax7.text(0.2,0.72,s,horizontalalignment='left',verticalalignment='center', transform = ax7.transAxes,size=22)
    #ax7.text(0.3,0.68,s,horizontalalignment='left',verticalalignment='center', transform = ax7.transAxes)
    ax7.set_xlabel('$t_{m}$(year)')
    ax7.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
    ax7.set_ylabel('a CrackLength (mm)')
    ax7.set_title('$a_{{True}},a_M,a_{{SS2}}; T_{{object}}$ = {}'.format(T_object))
    ax7.legend(bbox_to_anchor=(0.01,1), loc="upper left")
    # ax7.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
    if method2 == 'CEsBU_SG':
        fig7.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_SS2 with T_object = {}.png'.format(T_object),dpi=400)
    if method2 == 'CEsBU_SG_log':
        fig7.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_SS2 with T_object = {}.png'.format(T_object),dpi=400)
        
    # Plot a_True, a_M,a_SS3
    fig8,ax8 = plt.subplots(figsize=(8,6),dpi=400)
    ax8.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
    ax8.plot(t_m,a_data,'ro',label='$a_M$')
    ax8.plot(t_m,mu_a_SS3,'y+',label='$mu_{a_{SS3}}$')
    ax8.fill_between(t_m,upper_a_SS3,lower_a_SS3,alpha = 0.2,color='yellow',label='$C.I._{a_{SS3}}$',ls=':')
    s = Z_object_ref_text+Z_object_CEsBU2_text+Z_SS3_text
    ax8.text(0.2,0.72,s,horizontalalignment='left',verticalalignment='center', transform = ax8.transAxes,size=22)
    #ax8.text(0.3,0.68,s,horizontalalignment='left',verticalalignment='center', transform = ax8.transAxes)
    ax8.set_xlabel('$t_{m}$(year)')
    ax8.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
    ax8.set_ylabel('a CrackLength (mm)')
    ax8.set_title('$a_{{True}},a_M,a_{{SS3}}; T_{{object}}$ = {}'.format(T_object))
    ax8.legend(bbox_to_anchor=(0.01,1), loc="upper left")
    if method2 == 'CEsBU_SG':
        fig8.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_SS3 with T_object = {}.png'.format(T_object),dpi=400)
    if method2 == 'CEsBU_SG_log':
        fig8.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_SS3 with T_object = {}.png'.format(T_object),dpi=400)
    # Plot a_True, a_M,a_SS4
    fig20,ax20 = plt.subplots(figsize=(8,6),dpi=400)
    ax20.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
    ax20.plot(t_m,a_data,'ro',label='$a_M$')
    ax20.plot(t_m,mu_a_SS4,'r+',label='$mu_{a_{SS4}}$')
    ax20.fill_between(t_m,upper_a_SS4,lower_a_SS4,alpha = 0.2,color='red',label='$C.I._{a_{SS4}}$',ls=':')
    s = Z_object_ref_text+Z_object_CEsBU2_text+Z_SS4_text
    ax20.text(0.2,0.72,s,horizontalalignment='left',verticalalignment='center', transform = ax20.transAxes,size=22)
    #ax20.text(0.3,0.68,s,horizontalalignment='left',verticalalignment='center', transform = ax20.transAxes)
    ax20.set_xlabel('$t_{m}$(year)')
    ax20.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
    ax20.set_ylabel('a CrackLength (mm)')
    ax20.set_title('$a_{{True}},a_M,a_{{SS4}}; T_{{object}}$ = {}'.format(T_object))
    ax20.legend(bbox_to_anchor=(0.01,1), loc="upper left")
    if method2 == 'CEsBU_SG':
        fig20.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_SS4 with T_object = {}.png'.format(T_object),dpi=400)
    if method2 == 'CEsBU_SG_log':
        fig20.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_SS4 with T_object = {}.png'.format(T_object),dpi=400)
    
    # Plot a_True, a_M,a_SS1,2,3,4
    fig9,ax9 = plt.subplots(figsize=(8,6),dpi=400)
    ax9.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
    ax9.plot(t_m,a_data,'ro',label='$a_M$')
    ax9.plot(t_m,mu_a_SS1,'b+',label='$mu_{a_{SS1}}$')
    ax9.fill_between(t_m,upper_a_SS1,lower_a_SS1,alpha = 0.2,color='blue',label='$C.I._{a_{SS1}}$',ls=':')
    ax9.plot(t_m,mu_a_SS2,'g+',label='$mu_{a_{SS2}}$')
    ax9.fill_between(t_m,upper_a_SS2,lower_a_SS2,alpha = 0.2,color='green',label='$C.I._{a_{SS2}}$',ls=':')
    ax9.plot(t_m,mu_a_SS3,'y+',label='$mu_{a_{SS3}}$')
    ax9.fill_between(t_m,upper_a_SS3,lower_a_SS3,alpha = 0.2,color='yellow',label='$C.I._{a_{SS3}}$',ls=':')
    ax9.plot(t_m,mu_a_SS4,'r+',label='$mu_{a_{SS4}}$')
    ax9.fill_between(t_m,upper_a_SS4,lower_a_SS4,alpha = 0.2,color='red',label='$C.I._{a_{SS4}}$',ls=':')

    s = Z_object_ref_text+Z_object_CEsBU2_text+Z_SS1_text+Z_SS2_text+Z_SS3_text+Z_SS4_text
    ax9.text(0.2,0.7,s,horizontalalignment='left',verticalalignment='center', transform = ax9.transAxes,size='20')
    ax9.set_xlabel('$t_{m}$(year)')
    ax9.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
    ax9.set_ylabel('a CrackLength (mm)')
    ax9.set_title('Crack Length Evolution Prediction for $T_{{object}}$ = {}'.format(T_object))
    ax9.legend(bbox_to_anchor=(0.8,1.1), loc="upper left",fontsize=14)
    if method2 == 'CEsBU_SG':
        fig9.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_SS1,2,3,4 with T_object = {}.png'.format(T_object),dpi=400)
    if method2 == 'CEsBU_SG_log':
        fig9.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_SS1,2,3,4 with T_object = {}.png'.format(T_object),dpi=400)
   
    # Plot a_True, a_M,a_CEsBU1,2,a_SS1,2,3,4
    fig10,ax10 = plt.subplots(figsize=(8,6),dpi=400)


    ax10.plot(t_m,a_true,'c',linewidth=2,label = '$a_{True}$')
    ax10.plot(t_m,a_data,'ro',label='$a_M$')

    ax10.plot(t_m,mu_a_CEsBU1,'m+-',label='$mu_{a_{CEsBU1}}$')
    ax10.fill_between(t_m,upper_a_CEsBU1,lower_a_CEsBU1,alpha = 0.2,color='magenta',label='$C.I._{a_{CEsBU1}}$',ls=':')
    ax10.plot(t_m,mu_a_CEsBU2,'k+-',label='$mu_{a_{CEsBU2}}$')
    ax10.fill_between(t_m,upper_a_CEsBU2,lower_a_CEsBU2,alpha = 0.2,color='black',label='$C.I._{a_{CEsBU2}}$',ls=':')

    ax10.plot(t_m,mu_a_SS1,'b+',label='$mu_{a_{SS1}}$')
    ax10.fill_between(t_m,upper_a_SS1,lower_a_SS1,alpha = 0.2,color='blue',label='$C.I._{a_{SS1}}$',ls=':')
    ax10.plot(t_m,mu_a_SS2,'g+',label='$mu_{a_{SS2}}$')
    ax10.fill_between(t_m,upper_a_SS2,lower_a_SS2,alpha = 0.2,color='green',label='$C.I._{a_{SS2}}$',ls=':')
    ax10.plot(t_m,mu_a_SS3,'y+',label='$mu_{a_{SS3}}$')
    ax10.fill_between(t_m,upper_a_SS3,lower_a_SS3,alpha = 0.2,color='yellow',label='$C.I._{a_{SS3}}$',ls=':')
    ax10.plot(t_m,mu_a_SS4,'r+',label='$mu_{a_{SS4}}$')
    ax10.fill_between(t_m,upper_a_SS4,lower_a_SS4,alpha = 0.2,color='red',label='$C.I._{a_{SS4}}$',ls=':')
    
    s = Z_final_ref_text+Z_final_CEsBU1_text+Z_object_ref_text+Z_object_CEsBU2_text+Z_SS1_text+Z_SS2_text+Z_SS3_text+Z_SS4_text
    ax10.text(0.2,0.6,s,horizontalalignment='left',verticalalignment='center', transform = ax10.transAxes,size=22)
    #ax10.text(0.5,0.72,s,horizontalalignment='left',verticalalignment='center', transform = ax10.transAxes)
    ax10.set_xlabel('$t_{m}$(year)',fontsize = 18)
    ax10.axvline(T_object,label='$T_{object}$',ls='--',color = 'black')
    ax10.set_ylabel('a CrackLength (mm)',fontsize = 18)
    ax10.set_title('Crack Length Evolution Prediction for $T_{{object}}$ = {}'.format(T_object),fontsize = 18)

    ax10.legend(bbox_to_anchor=(0.8,1.1), loc="upper left",fontsize = 14)
    
    if method1 == 'CEsBU_SG':
        fig10.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\CrackLength\\a_True,a_M,a_CEsBU1,2,a_SS1,2,3,4 with T_object = {}.png'.format(T_object),dpi=400)
    if method1 == 'CEsBU_SG_log':
        fig10.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\CrackLength\\a_True,a_M,a_CEsBU1,2,a_SS1,2,3,4 with T_object = {}.png'.format(T_object),dpi=400)
    # Plot InterDist for Prior; CEsBU1,2; SS1,2,3
    fig11,ax11 = plt.subplots(figsize=(5,4),dpi=400)
    sns.distplot(X_resample_final,label='$dist_{T_{final_{CEsBU1}}}$')
    sns.distplot(X_resample_object,label='$dist_{T_{object_{CEsBU2}}}$')
    sns.distplot(X_object1,label='$dist_{T_{object_{SS1}}}$')
    sns.distplot(X_object2,label='$dist_{T_{object_{SS2}}}$')
    sns.distplot(X_object3,label='$dist_{T_{object_{SS3}}}$')
    sns.distplot(X_object4,label='$dist_{T_{object_{SS4}}}$')
    ax11.plot([mu_prior],[0],'b^',label='$mu_{prior}$')
    ax11.plot([x_given],[0],'bv',label='$x_{given}$')
    s = mu_prior_text+x_given_text+mu_CEsBU1_text+mu_CEsBU2_text+mu_SS1_text+mu_SS2_text+mu_SS3_text+mu_SS4_text
    ax11.text(0.35,0.68,s,horizontalalignment='left',verticalalignment='center', transform = ax11.transAxes)
    ax11.set_xlabel('DS')
    ax11.set_ylabel('PDF')
    ax11.set_title('CEsBU1,2,SS1,2,3,4 with $T_{{object}}$ = {}'.format(T_object))
    ax11.legend(bbox_to_anchor=(0,1), loc="upper left")
    if method1 == 'CEsBU_SG':
        fig11.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG\\InterDist\\CEsBU1,2,SS1,2,3,4 with T_object = {}.png'.format(T_object),dpi=400)
    if method1 == 'CEsBU_SG_log':
        fig11.savefig('figures\\CEsBUexamples\\Example1_1RV_DS_indirectObservations_t\\SG_log\\InterDist\\CEsBU1,2,SS1,2,3,4 with T_object = {}.png'.format(T_object),dpi=400)
        














