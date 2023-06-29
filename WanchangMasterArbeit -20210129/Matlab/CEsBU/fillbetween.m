function fillbetween(x,y1,y2,varargin)
    x2 = [x fliplr(x)];
    inbetween = [y1, fliplr(y2)];
    patch(x2,inbetween,varargin{:})
end