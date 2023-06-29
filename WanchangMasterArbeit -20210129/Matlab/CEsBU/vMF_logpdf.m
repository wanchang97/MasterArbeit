function y = vMF_logpdf(X,mu,kappa)
    % Returns the von Mises-Fisher mixture log pdf on the unit hypersphere
    d = size(X,1);
    n = size(X,2);
    if kappa == 0
        % unit hypersphere
        A = log(d) + log(pi^(d/2)) - gammaln(d/2+1);
        y = -A*ones(1,n);
    elseif kappa > 0
        % concentrated direction
        c = (d/2-1)*log(kappa)-(d/2)*log(2*pi)-logbesseli(d/2-1,kappa);
        q = bsxfun(@times,mu,kappa)*X;
        y = bsxfun(@plus,q,c');
    else
        error('kappa<0 or NaN');
    end
end
