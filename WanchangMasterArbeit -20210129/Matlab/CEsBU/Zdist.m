function Z = Zdist(samples,x,varargin)
    samples = squeeze(samples);
    try
        pd = fitdist(samples,varargin{:});
        Z = pdf(pd,x);
    catch
        try
            pd = fitdist(samples',varargin{:});
            Z = pdf(pd,x);
        catch
            Z = nan;
        end
    end
end