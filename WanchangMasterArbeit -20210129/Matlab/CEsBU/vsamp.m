function X = vsamp(center, kappa, n)
    % only d > 1 
    d  = size(center,1);	% dimension
    
    t1 = sqrt(4*kappa*kappa + (d-1)*(d-1));
    b  = (-2*kappa + t1 )/(d-1);
    x0 = (1-b)/(1+b);
    X  = zeros(n,d);
    m  = (d-1)/2;
    c  = kappa*x0 + (d-1)*log(1-x0*x0);
    for i = 1:n
        t = -1000; 
        u = 1;   
        while (t < log(u))
            z = betarnd(m , m);	   % z is a beta rand var
            u = rand;			    	% u is uniform rand var
            w = (1-(1+b)*z)/(1-(1-b)*z);
            t = kappa*w+(d-1)*log(1-x0*w)-c;
        end   
        v = hs_sample(1,d-1,1);
        X(i,1:d-1) = sqrt(1-w*w)*v';
        X(i,d) = w;
    end
    [v,b] = house(center);
    Q = eye(d) - b*(v*v');
    for i = 1:n
        tmpv = Q*X(i,:)';
        X(i,:) = tmpv';
    end
end