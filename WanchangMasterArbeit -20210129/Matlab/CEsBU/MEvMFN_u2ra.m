function [R,X_norm] = MEvMFN_u2ra(u)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%     u       samples in cartesian coordinates
% Output:
%     R       samples radius
%     X_norm  samples direction in rad
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    R      = sqrt(sum(u.^2,1));
    X_norm = u./R;
end