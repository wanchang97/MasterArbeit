function [lnL_total_allyear] = log_Likelihood(x,year,a_m,t_m,mu_error,si_error)
lnL_total_allyear = [];
lnL_total = 0;
%lnL_total_allyear{end+1} = lnL_total;
% m = size(a_m);% m: 10x1
% mu_error = 0; si_error = 1;
for i = 1: year
    aeval = a(x,t_m(i));
    lnL_total = lnL_total + loggausspdf(a_m(i), aeval+mu_error,si_error);
    lnL_total(isnan(lnL_total))= min(lnL_total);
    lnL_total_allyear{end+1} = lnL_total;
end
end

