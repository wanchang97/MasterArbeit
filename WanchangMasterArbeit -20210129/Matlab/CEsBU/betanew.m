function newbeta = betanew(betaold,u,v,CEdensity,logLikelihood_,logpriorpdf_,Nstep,cvtarget)
        logWconst_ = logwconst2(u,v,CEdensity,logpriorpdf_); % necessary as new set of v given as computation after parameter update
        logstep_ = logstep(Nstep,cvtarget); % could be computed beforehand (once) but computational cost is negligible

        logESSoptimize_ = @(dbeta) logESSoptimize(betaold+abs(dbeta),logLikelihood_,logWconst_,logstep_);

        options = optimset('TolFun',1e-100,'TolX',1e-100,'MaxFunEvals',1000,'MaxIter',1000);%,'Display','iter');%,'PlotFcns',@optimplotfval);
        [dbeta,~] = fminbnd(logESSoptimize_,-1,1e-6,options);
        dbeta = abs(dbeta);
        newbeta = betaold+dbeta;

        % for documentation
% %         figure(99)
% %         close;
%         figure(99)
%         logESSoptimize_plot = @(beta) logESSoptimize(beta,logLikelihood_,logWconst_,logstep_);
%         hold on;
%         xaxis = linspace(-1,2,200);
%         ESS = arrayfun(@(i)logESSoptimize_plot(xaxis(i)),1:200);
%         plot(xaxis,ESS)
%         plot(xaxis,logstep_^2*ones(size(xaxis)),'r--');
%         plot(betaold,logESSoptimize_(0),'rx')
%         plot(newbeta,logESSoptimize_(dbeta),'gx')
%         hold off;

        % for convergence
        if newbeta >=1-1e-2
            newbeta=1;
        end
end