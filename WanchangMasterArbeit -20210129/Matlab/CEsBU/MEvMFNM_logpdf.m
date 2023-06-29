function h = MEvMFNM_logpdf(X,mu,kappa,omega,m,alpha)
    X=X';
    k = length(alpha);
    [N,dim] = size(X);
    [R,X_norm] = MEvMFN_u2ra(X');
    h = zeros(k,N);
    if k == 1   
       % log pdf of vMF distribution
       logpdf_vMF = vMF_logpdf(X_norm,mu(1,:),kappa(1))';
       % log pdf of Nakagami distribution
       logpdf_N = nakagami_logpdf(R',m(1),omega(1));
       % log pdf of weighted combined distribution
       h(1,:) = logpdf_vMF+logpdf_N;
    else   
       logpdf_vMF = zeros(N,k);
       logpdf_N   = zeros(N,k);
       h_log      = zeros(k,N);

       % log pdf of distributions in the mixture
       for i = 1:k
          % log pdf of vMF distribution
          logpdf_vMF(:,i) = vMF_logpdf(X_norm,mu(i,:),kappa(i))';
          % log pdf of Nakagami distribution
          logpdf_N(:,i) = nakagami_logpdf(R',m(i),omega(i));
          % log pdf of weighted combined distribution
          h_log(i,:) = logpdf_vMF(:,i)+logpdf_N(:,i)+log(alpha(i));
       end

       % mixture log pdf
       h = logsumexp(h_log,1);
    end
end