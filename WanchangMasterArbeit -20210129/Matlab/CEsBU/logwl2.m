function logw = logwl2(u,v,ln_L_fun,year,beta,CEpriorobj,CEdensity,priorobj)
    logL_ = ln_L_fun(priorobj.U2X(u),year);
    logP_ = logPrior(u,CEpriorobj);
    logwconst_ = logwconst2(u,v,CEdensity,logP_);
    logw = beta*logL_+logwconst_;
end