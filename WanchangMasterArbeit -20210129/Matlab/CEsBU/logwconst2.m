function logconstw = logwconst2(u,v,CEdensity,logpriorpdf)
    logconstw = logpriorpdf-CEdensity.jointpdf(u,v);
end