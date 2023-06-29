function priori = logPrior(u,CEpriorobj)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the function which evaluates the logpdf of the prior. It is done
% separately as this function could be an interface for additional features
% like shifting then.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    priori = CEpriorobj.jointpdf(u);
end