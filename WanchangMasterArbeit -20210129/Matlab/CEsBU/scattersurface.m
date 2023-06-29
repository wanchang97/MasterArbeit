% plot sample based surface of joint PDF
function handle = scattersurface(samples,bins,xlimit,ylimit)
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
        handle = surface(c{1},c{2},n','FaceColor','interp');
    elseif nargin==3
        edges = {linspace(xlimit(1),xlimit(2),xbins+1);linspace(xlimit(1),xlimit(2),ybins+1)};
        [n,c] = hist3([samples'],'Edges',edges);
        handle = surface(c{1},c{2},n','FaceColor','interp');
    elseif nargin==4
        edges = {linspace(xlimit(1),xlimit(2),xbins+1);linspace(ylimit(1),ylimit(2),ybins+1)};
        [n,c] = hist3([samples'],'Edges',edges);
        handle = surface(c{1},c{2},n','FaceColor','interp');
    end
end