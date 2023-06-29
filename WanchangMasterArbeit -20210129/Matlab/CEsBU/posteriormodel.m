classdef posteriormodel
    properties
        dim
        dist
        k
    end
    methods
        % initialization
        function obj = posteriormodel(dim_,dist_,k_)
            obj.dim = dim_;
            obj.dist = dist_;
            obj.k = k_;
        end
        % draw samples
        function samples = random(obj,N,v)
            if strcmp(obj.dist,'normal')
                mu = v{1};
                si = v{2};
                ki = v{3};
                samples = GM_sample(mu,si,ki,N)';
            elseif strcmp(obj.dist,'vMFN')
                mu = v{1};
                kappa = v{2};
                omega = v{3};
                m = v{4};
                alpha = v{5};
                samples = vMFNM_sample(mu,kappa,omega,m,alpha,N)';
            else
                error('\nnot an implemented IS-density\n')
            end
        end
        % evaluate logPDF
        function pdf = jointpdf(obj,u,v)
            if strcmp(obj.dist,'normal')
                X = u;
                mu = v{1};
                si = v{2};
                ki = v{3};
                pdf = MEGM_logpdf(X,mu,si,ki);
            elseif strcmp(obj.dist,'vMFN')
                X = u;
                mu = v{1};
                kappa = v{2};
                omega = v{3};
                m = v{4};
                alpha = v{5};
                pdf = MEvMFNM_logpdf(X,mu,kappa,omega,m,alpha);
            else
                error('\nnot an implemented IS-density\n')
            end
        end
    end
end