function y = loggausspdf(X, mu, Sigma)
    d = size(X,1);
    X = bsxfun(@minus,X,mu);
    [U,~] = chol(Sigma);
    Q = U'\X;
    q = dot(Q,Q,1); % quadratic term (Mahalanobis distance)
    c = d*log(2*pi)+2*sum(log(diag(U))); % normalization constant
    y = -(c+q)/2;
end