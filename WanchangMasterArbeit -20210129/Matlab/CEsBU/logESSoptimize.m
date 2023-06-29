function logw = logESSoptimize(beta,logLikelihood,logwconst,logstep)
    logw = beta.*logLikelihood+logwconst;
    logw = ((2*logsumexp(logw)-logsumexp(2*logw))-(logstep))^2;
end