function signal = MEpadded(orig,len2,val,dir)
    len = length(orig);
    if len>=len2
        signal=orig(1:len2);
    else
        signal = padarray(orig',len2-len,val,dir)';
    end
end