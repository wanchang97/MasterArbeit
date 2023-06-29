function y = nakagami_logpdf(X,m,om)
    y = log(2)+m*(log(m)-log(om)-X.^2./om)+log(X).*(2*m-1)-gammaln(m);
end