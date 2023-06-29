% plot sample based contour of joint PDF
function handle = scattercontour(samples,bins,levels,color,xlimit,ylimit,varargin)
    if nargin == 1
        xbins = ceil(sqrt(size(samples,2)));
        ybins = ceil(sqrt(size(samples,2)));
    elseif length(bins)==2
        xbins = bins(1);
        ybins = bins(2);
    else
        xbins = bins;
        ybins = bins;
    end
    
    if nargin <= 2
        [n,c] = hist3([samples'],'Nbins',[xbins,ybins]);
        handle = contour(c{1},c{2},n',varargin{:});
    elseif nargin==3
        [n,c] = hist3([samples'],'Nbins',[xbins,ybins]);
        handle = contour(c{1},c{2},n',levels,varargin{:});
    elseif nargin==4
        [n,c] = hist3([samples'],'Nbins',[xbins,ybins]);
        if isempty(color)
            handle = contour(c{1},c{2},n',levels);
        else
            handle = contour(c{1},c{2},n',levels,'LineColor',color,varargin{:});
        end
    elseif nargin==5
        edges = {linspace(xlimit(1),xlimit(2),xbins+1);linspace(xlimit(1),xlimit(2),ybins+1)};
        [n,c] = hist3([samples'],'Edges',edges);
        if isempty(color)
            handle = contour(c{1},c{2},n',levels);
        else
            handle = contour(c{1},c{2},n',levels,'LineColor',color,varargin{:});
        end
    elseif nargin==6
        edges = {linspace(xlimit(1),xlimit(2),xbins+1);linspace(ylimit(1),ylimit(2),ybins+1)};
        [n,c] = hist3([samples'],'Edges',edges);
        if isempty(color)
            handle = contour(c{1},c{2},n',levels);
        else
            handle = contour(c{1},c{2},n',levels,'LineColor',color,varargin{:});
        end
    else
        edges = {linspace(xlimit(1),xlimit(2),xbins+1);linspace(ylimit(1),ylimit(2),ybins+1)};
        [n,c] = hist3([samples'],'Edges',edges);
        if isempty(color)
            handle = contour(c{1},c{2},n',levels);
        else
            handle = contour(c{1},c{2},n',levels,'LineColor',color,varargin{:});
        end
    end
end