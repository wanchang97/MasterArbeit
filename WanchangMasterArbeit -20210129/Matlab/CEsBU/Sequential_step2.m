function [U_object,X_object,Z_object] = Sequential_step2(priorobj,T_object,lv,u_tot,nu_tot,lnLeval_allyear,dim, k, ISdensity,N)
%{
The Sequential_step is used to get the samples of the target year from the
CEBU calculation of the final year
-----------------------------------------------------------------------------------------------
Input
* prior                     : list of Nataf distribution object or marginal distribution
* T_object                  : object year
* lv                        : total number of levels
* samplesU                  : object with the samples in the standard normal space
* mu_U_list                 : intermediate mu_U
* si_U_list                 : intermediate si_U
* lnLeval_allyear_list      : loglikelihood at object year (it is written in physical space)
-----------------------------------------------------------------------------------------------
Output
* U_object          : samples in standard normal space in object year
* X_object          : samples in original space in object year
* Z_object          : normalisation constant in object year
-----------------------------------------------------------------------------------------------
    %}
    fprintf('\nstart Sequential_step2 (with log-scaled input)\n')
    %% initialize posteriormodel
    if strcmp(ISdensity,'normal')    
        %% initialize v0 for GM
        mu = 0*ones(k,dim);
        si = zeros(dim,dim,k);
        for i=1:k
            si(:,:,i) = si(:,:,i)+1*eye(dim);
        end
        ki = ones(k,1)./k;
        v0 = {mu,si,ki};
        CEdensity = posteriormodel(dim,ISdensity,k);
    elseif strcmp(ISdensity,'vMFN') 
        %% initialize v0 for vMFNM
        mu = 1/sqrt(dim).*ones(k,dim);
        kappa = zeros(k,1);
        omega = ones(k,1)*(dim);
        m = ones(k,1).*(dim/2);
        alpha = ones(k,1)./k;
        v0 = {mu,kappa,omega,m,alpha};
        CEdensity = posteriormodel(dim,ISdensity,k);
    else
        error('\nWrong input for ISdensity!\n')
    end
    
    %% initialize prior in standard normal space
    CEprior = priormodel(dim,ISdensity);
    ln_z = zeros(1,lv);
    nESS = zeros(1,lv);
    ln_w_normalised_cell = {};
    w_resample_cell = {};
    for t= 1: lv
        ln_h = CEdensity.jointpdf(u_tot{t},nu_tot{t});
        %ln_h = loggausspdf(transpose(u_tot{t}),nu_tot{t}{1},nu_tot{t}{2});
        %sp.stats.multivariate_normal.logpdf(samplesU[t].T,mu_U_list[t],si_U_list[t]) % N h_list[0]->h_list[lv-1] prior to posterior pdf
        ln_f_prior = logPrior(u_tot{t},CEprior);% 1xN
        ln_L_object = lnLeval_allyear{t}{T_object};
%         ln_w_const = ln_f_prior-CEdensity.jointpdf(u_tot{t},nu_tot{t});
        %ln_w_const = logwconst2(u_tot{t},nu_tot{t},CEdensity,ln_f_prior);
        ln_w_new  = ln_f_prior + ln_L_object - ln_h;
        ln_w_normalised = ln_w_new-logsumexp(ln_w_new);
%         ln_w_new_cell{end+1} = ln_w_new;
        ln_w_normalised_cell{end+1} = ln_w_normalised;
        ln_z(t) = logsumexp(ln_w_new)-log(N);% sum(w_new)/N # logsumexp(ln_w_new)-ln(N)    
        nESS(t) = exp(2*logsumexp(ln_w_new)-logsumexp(2*ln_w_new)-log(N));

    end
    pi = nESS / sum(nESS);
    % Evidence evaluation
    Z = 0;
    for t = 1: lv
        Z = Z + pi(t) *exp(ln_z(t));
        %Z = Z + pi(t) * vpa(exp(ln_z(t)),10000);
        w_resample = pi(t) * exp(cell2mat(ln_w_normalised_cell(t)));
        %w_resample = pi(t) * vpa(exp(cell2mat(ln_w_normalised_cell(t))),10000);
        w_resample_cell{end+1} = w_resample;
    end
    Z_object = Z;
    % Generate samples according to filter distribution
    W_resample = cell2mat(w_resample_cell);
    W_resample = reshape(W_resample,1,[]);
    W_resample_normalised = W_resample/sum(W_resample);
    SamplesU = cell2mat(u_tot);
    U_object = MErandsample(SamplesU,N,true,W_resample_normalised);
    X_object = priorobj.U2X(U_object);
end

