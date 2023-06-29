function [evidence,lv,u_tot,x_tot,randsamples,beta_tot,v_tot,nESS_tot,lnLeval_allyear] = CEsBU(Nstep, Nlast,ln_L_fun,year, priorobj, maxsteps, cvtarget, dim, k, ISdensity)
    fprintf('\nstart CEsBU (with log-scaled input)\n')
    
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
    
    %% initialize parameters
    u_tot = {};
    x_tot = {};
    v_tot = {};
    beta_tot = [0];
    cv_tot = [];
    lnLeval_allyear = [];
    v = v0;
    v_tot{1} = v0;
    beta = 0;
    nESS_tot = [0];

    %% iteration
    j = 1;
    while (j<=maxsteps && beta<=1)
        fprintf(['\n\nstep ',num2str(j),' for beta=',num2str(beta)])
        %% Generate samples
        u = CEdensity.random(Nstep,v);
        u_tot{end+1} = u;
        %% precalculation for weights, parameter updates and beta
        fprintf('\ncompute likelihood')
        logLikelihood_allyear = ln_L_fun(priorobj.U2X(u),year);
        %logLikelihood_allyear = logLikelihood(priorobj.U2X(u),year,ln_L_fun);
        lnLeval_allyear{end+1} = logLikelihood_allyear;
        logLikelihood_ = logLikelihood_allyear{year};
        %logLikelihood_ = logLikelihood(priorobj.U2X(u),likelihood);  
        logPriorpdf_ = logPrior(u,CEprior);
        
        %% update beta
        fprintf('\nupdate beta')
        beta = betanew(beta,u,v,CEdensity,logLikelihood_,logPriorpdf_,Nstep,cvtarget);
        if beta > 1
            beta = 1;
        end
        beta_tot(end+1) = beta;        
        fprintf(['\nnext beta: ', num2str(beta)])
        
        %% likelihood ratio
        fprintf('\nstart likelihood ratio')
        logWconst_ = logwconst2(u,v,CEdensity,logPriorpdf_);
        logW = logwl(beta,logLikelihood_,logWconst_); 
        
        cv_tot(end+1) = std(exp(logW))/mean(exp(logW)); % only for diagnostics, hence not optimized for precision
        fprintf('\ncvtarget: %f\ncvobtained: %f',cvtarget,cv_tot(end));
        nESS  = exp(2*logsumexp(logW)-logsumexp(2*logW)-log(Nstep));
        nESS_tot(end+1) = nESS;
        %% update parameter
        fprintf('\nstart parameter estimation')
        v = vnew(u,logW,k,ISdensity);
        v_tot{end+1} = v;
        
        if beta>=1
            fprintf('\nbeta is equal to 1')
            break;
        end
            
        j = j+1;
    end
    
    %% results
    lv  = j+1;
    fprintf('\n\nstart final sampling')
    ulast = CEdensity.random(Nlast,v);
    u_tot{end+1} = ulast;
    logLikelihood_allyear = ln_L_fun(priorobj.U2X(ulast),year);
    lnLeval_allyear{end+1} = logLikelihood_allyear;
    logLikelihood_ = logLikelihood_allyear{year};
    logPriorpdf_ = logPrior(ulast,CEprior);
    logWconst_ = logwconst2(ulast,v,CEdensity,logPriorpdf_);
    logWlast = logwl(1,logLikelihood_,logWconst_); 
    
    %logWlast = logwl2(ulast,v,ln_L_fun,year,1,CEprior,CEdensity,priorobj);
    Wlast = exp(logWlast);
    Wlast_normed = exp(logWlast-logsumexp(logWlast,2));
    nESS = exp(2*logsumexp(logWlast)-logsumexp(2*logWlast)-log(Nlast));
    nESS_tot(end+1) = nESS;
    fprintf(['\nnESS of randsamples: ',num2str(nESS)])
    evidence = evidencehat2(logWlast,Nlast);     
    xlast = priorobj.U2X(ulast);    
    randsamples = MErandsample(xlast,Nlast,true,Wlast_normed);
    for i= 1: lv
        x_tot{end+1} = priorobj.U2X(u_tot{i});
    end
    fprintf(['\nfinished after ',num2str(j-1),' steps at beta=',num2str(beta),'\n'])
end