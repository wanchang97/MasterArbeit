function [v,b] = house(x)
    n = length(x);
    s = x(1:n-1)'*x(1:n-1);
    v = [x(1:n-1)', 1]';
    if (s == 0)
       b = 0;
    else
       m = sqrt(x(n)*x(n) + s);   
       if (x(n) <= 0)
          v(n) = x(n)-m;
       else
          v(n) = -s/(x(n)+m);
       end
       b = 2*v(n)*v(n)/(s + v(n)*v(n));
       v = v/v(n);
    end
end
