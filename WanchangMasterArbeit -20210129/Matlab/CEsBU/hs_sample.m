function X = hs_sample(N,n,R)
    Y    = randn(n,N);
    Y    = Y';
    norm = repmat(sqrt(sum(Y.^2,2)),[1 n]);
    X    = Y./norm*R;
end