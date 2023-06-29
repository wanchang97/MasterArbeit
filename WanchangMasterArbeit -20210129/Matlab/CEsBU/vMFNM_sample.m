function X = vMFNM_sample(mu,kappa,omega,m,alpha,N)
    % Returns samples from the von Mises-Fisher-Nakagami mixture
    [k,dim] = size(mu);
    if k == 1
       % sampling the radius
       R = sqrt(gamrnd(m,omega./m,N,1));
       % sampling direction on unit hypersphere
       X_norm = vsamp(mu',kappa,N);    
    else   
       % Determine number of samples from each distribution
       z = sum(dummyvar(randsample(k,N,true,alpha)));
       k = length(z);
       
       % Generation of samples
       R = zeros(N,1);
       R_last = 0;
       X_norm = zeros(N,dim);
       X_last = 0;
       for i = 1:k      
          % sampling the radius
          R(R_last+1:R_last+z(i)) = sqrt(gamrnd(m(i),omega(i)./m(i),z(i),1));
          R_last = R_last + z(i);
          % sampling direction on unit hypersphere
          X_norm(X_last+1:X_last+z(i),:) = vsamp(mu(i,:)',kappa(i),z(i));
          X_last = X_last+z(i);
          clear pd;
       end
    end
    % assign sample vector
    X = bsxfun(@times,R,X_norm);
end