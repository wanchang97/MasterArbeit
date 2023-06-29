function newv = vnew(u,logWold,k,ISdensity)
    if strcmp(ISdensity,'normal')
        if k == 1
            fprintf('\n...using analytical formulas')
            sumwold = vpa(exp(logsumexp(logWold,2)),10000);
%             disp(sumwold)
            dim = size(u,1);
            
            signu = sign(u);
            logu = log(abs(u));
            
            [mu_unnormed,signmu] = signedlogsumexp(logu+logWold,2,signu);
            mu = exp(mu_unnormed-logsumexp(logWold,2)).*signmu;

            v0 = mu';
            
            v1 = zeros(dim,dim,k);
            sqrtR = exp(0.5.*logWold);
            Xo = bsxfun(@minus,u,v0(1,:)');
            Xo = bsxfun(@times,Xo,sqrtR);
            v1(:,:,1) = Xo*Xo'./sumwold;
            v1(:,:,1) = v1(:,:,1)+eye(dim)*(1e-6);
            
            v2 = [1];
            
            newv = {v0,v1,v2};
        else
            if all(u==0)
                fprintf('issue with samples')
            end
%             fprintf('\n...using EMGM ')
%             [v0,v1,v2] = EMGM(u,Wold',k);
            fprintf('\n...using EMGM_log ')
            [v0,v1,v2] = EMGM_log(u,logWold,k);
            newv = {v0',v1,v2};
        end
    
    elseif strcmp(ISdensity,'vMFN')
        if k == 1
            dim = size(u,1);
            [R,X] = MEvMFN_u2ra(u);
            logsumwold = logsumexp(logWold,2);
            
            % mean direction mu
            signX = sign(X);
            logX = log(abs(X));
            
            [mu_unnormed,signmu] = signedlogsumexp(logX+logWold,2,signX);
            lognorm_mu = 0.5*logsumexp(2*mu_unnormed,1);
            mu = exp(mu_unnormed-lognorm_mu).*signmu;
            mu(isinf(mu)) = sqrt(dim);
            
            v0 = mu;
                
            % concentration parameter kappa
            xi = min(exp(lognorm_mu-logsumwold),0.999999999);
            kappa = abs((xi*dim-xi^3)/(1-xi^2)); 
                
            v1 = kappa;
        
            % spread parameter omega
            R(R==0)=1e-300; % for numerical stability; should actually be equal to the number closest to zero without cancellation error by machine
            
            logR = log(R);
            logRsquare = 2*logR;
            omega = exp(logsumexp(logWold+logRsquare,2) - logsumwold);
            
            v2 = omega;
            
            % shape parameter m
            logRpower = 4*logR;
            mu4       = exp(logsumexp(logWold+logRpower,2) - logsumwold);
            m         = omega.^2./(mu4-omega.^2);
            
            m(m<0.5)  = dim/2;
            m(isinf(m)) = dim/2;
            
            v3 = m;
            
            % distribution weights alpha
            alpha = [1];
            
            v4 = alpha;
            
            newv = {v0',v1,v2,v3,v4};
        else
%             % EMvMFNM from ERA
%             fprintf('\nusing EMvMFNM ')
%             [v0,v1,v3,v2,v4] = EMvMFNM(u,Wold',k);
            fprintf('\nusing EMvMFNM_log ')
            [v0,v1,v3,v2,v4] = EMvMFNM_log(u,logWold,k); % for sake of consistency I let m and omega computed in the same order as within the EMvMFNM-script -> therefore the transposed digits
            
            newv = {v0',v1,v2,v3,v4};
        end
    else
        error ('\nWrong input for parametric importance sampling density model!\n')
    end
end