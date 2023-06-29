function X = GM_sample(mu,Si,Pi,N)
    if size(mu,1) == 1
        X = mvnrnd(mu,Si,N);
    else
        ind = randsample(size(mu,1),N,true,Pi);
        z = histcounts(ind,[(1:size(mu,1)) size(mu,1)+1]);
        X = ones(N,size(mu,2));
        ind = 1;
        for i = 1:size(mu,1)
            np = z(i);
            X(ind:ind+np-1,:) = mvnrnd(mu(i,:),Si(:,:,i),np);
            ind = ind+np;
        end
    end
end