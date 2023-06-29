function h = MEGM_logpdf(X,mu,si,ki)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%     X   samples             [dim x Nsamples]
%     mu  mean                [Nmodes x dim]
%     si  covariance          [dim x dim x Nmodes]
%     ki  weights of modes    [1 x Nmodes]
% Output:
%     h   logpdf of samples   [1 x Nsamples]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% given as
    N = size(X,2); % number of samples
    k_tmp = length(ki); % number of modes
    if k_tmp == 1
        h = loggausspdf(X,mu',si);
    else
        h_pre = zeros(k_tmp,N);
        for q = 1:k_tmp
            mu_ = mu(q,:);
            si_ = si(:,:,q);
            h_pre(q,:) = log(ki(q)) + loggausspdf(X,mu_',si_);
        end
        h = logsumexp(h_pre,1);
    end
end