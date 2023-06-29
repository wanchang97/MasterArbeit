function counter = MEuniquecount(W)
    counter = length(unique(randsample(1:length(W),length(W),true,W)));
end