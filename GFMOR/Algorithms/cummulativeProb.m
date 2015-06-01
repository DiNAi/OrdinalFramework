function [y] = cummulativeProb(x,beta)
    y =  1 ./ (1+exp((x-beta))); %Logit
%    y =  1-exp(-exp(beta-x)); %log log complementario
%    y = exp(-exp(x-beta)); %log log negativo
%    y = normcdf(beta-x); %probit
%    y = atan(beta-x)/pi + 0.5; % cauchit
end
