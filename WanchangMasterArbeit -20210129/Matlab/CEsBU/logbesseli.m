function logb = logbesseli(nu,kappa)
    if nu == 0 % special case when nu=0
        logb = log(besseli(0,kappa,1))+abs(kappa); % for stability
    else % normal case
        n = 1;
        frac   = kappa./nu;
        square = ones(n,1) + frac.^2;
        root   = sqrt(square);
        eta    = root + log(frac) - log(ones(n,1)+root);
        logb   = - log(sqrt(2*pi*nu)) + nu.*eta - 0.25*log(square);
    end
end