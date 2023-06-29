function evidence = evidencehat2(logW,N)
    evidence = exp(logsumexp(logW)-log(N));
end