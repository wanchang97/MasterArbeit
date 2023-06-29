classdef priormodel
    properties
        dim
        dist
        mu
        si
    end
    methods
        % initialization
        function obj = priormodel(dim_,dist_)
            obj.dim = dim_;
            obj.dist = dist_;
            obj.mu = zeros(obj.dim,1);
            obj.si = eye(obj.dim);
        end
        % draw samples
        function samples = random(obj,N)
            samples = mvnrnd(obj.mu,obj.si,N)';
        end
        % evaluate logPDF
        function pdf = jointpdf(obj,u)
            if strcmp(obj.dist,'vMFN')
                [R,X_norm] = MEvMFN_u2ra(u);
                R(R==0)=1e-300; %numerical stability; should actually be equal to number closest to zero without cancellation error
                % uniform hypersphere
                A   = log(obj.dim)+obj.dim/2*log(pi)-gammaln(obj.dim/2+1);
                f_u = -A;
                % chi distribution
                f_chi = log(2)*(1-obj.dim/2)+log(R)*(obj.dim-1)-0.5*R.^2-gammaln(obj.dim/2);
                pdf = f_u + f_chi;
            elseif strcmp(obj.dist,'normal')
                pdf = loggausspdf(u,obj.mu,obj.si);
            else
                error('\nnot an implemented IS-density\n')
            end
        end
    end
end