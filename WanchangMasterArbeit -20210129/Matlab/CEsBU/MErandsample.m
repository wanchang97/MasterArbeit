function randsamples = MErandsample(u,n,replace,W)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is a workaround so that the MATLAB and python versions are
% consistent. See MATLAB-function randsample for documentation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    randsamples = u(:,randsample(1:length(W),n,replace,W));
end     