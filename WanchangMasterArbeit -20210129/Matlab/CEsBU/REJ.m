% Simple Rejection Sampling Algorithm
function [samples, evidence, TME] = REJ(minsamples,loglikelihood,priorobj,c)
    fprintf('\nstart Rejection Sampling\n')
    u0 = priorobj.random(1);
    dim = size(u0,1);
    samples = zeros(dim,minsamples);
    
    TME = 0;
    counter = 0;
    logc = log(c);
    mu = zeros(1,dim);
    sigma = eye(dim);
    toprint = false;
    while counter<minsamples
        y = mvnrnd(mu,sigma,1)';
        a = loglikelihood(priorobj.U2X(y))-logc;
        TME = TME+1;
        
        if a >= log(rand())
%             fprintf('\nsample accepted')
            counter = counter+1;
            samples(:,counter) = y;
            toprint = true;
        end
        
        if mod(counter,floor(0.1*minsamples))==0 && toprint
                fprintf(['\n\nacceptance rate: ',num2str(counter/TME*100),'%%'])
                fprintf(['\n',num2str(counter/minsamples*100),'%% of desired samples\n'])
                toprint = false;
        end
        
        if mod(TME,10000)==0
            fprintf(['\n',num2str(TME),' model evaluations done'])
        end
    end
    samples = priorobj.U2X(samples);
    evidence = counter/TME*c;
    fprintf('\n\nfinished')
    fprintf(['\nacceptance rate: ',num2str(counter/TME*100),'%%'])
    fprintf(['\nRejection Sampling done'])
end