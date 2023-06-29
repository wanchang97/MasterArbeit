function logS = logstep(Nstep,cvtarget)
    logS = log(Nstep)-log(1+cvtarget^2);
end