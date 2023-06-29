function [s,sign] = signedlogsumexp(x, dim, b)
    % Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
    % By default dim = 1 (columns).
    % Written by Michael Chen (sth4nth@gmail.com).
    
    % Adapted by Michael Engel such that log(sum(exp(x)*b,dim)) and
    % negative results are supported.
    % Only recommended for b working as a sign-vector of exp(x).

    if nargin == 1
       % Determine which dimension sum will use
       dim = find(size(x)~=1,1);
       if isempty(dim)
          dim = 1;
       end
    end
    
    if isempty(b)
        b = ones(1,size(x,dim));
    end

    % subtract the largest in each column (rescaling x to (0,1] where log offers better precision)
    y = max(x,[],dim);
    x = bsxfun(@minus,x,y);
    term = sum(exp(x).*b,dim);
    sign = ones(size(term));
    sign(term<0) = -1;

    % return nonfinite value if existing
    s = y + log(abs(term));
    i = find(~isfinite(y));
    if ~isempty(i)
       s(i) = y(i);
    end
end