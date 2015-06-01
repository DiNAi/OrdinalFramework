function [predictedPatterns] = predictGravitation(x,X,Y,W)
%GRAVITATION Summary of this function goes here
%   Detailed explanation goes here


nEvaluationPatterns = size(x,1);
nClasses = size(Y,2);
gravitationalVector = zeros(nEvaluationPatterns,nClasses);

for n = 1:nEvaluationPatterns,
    for j = 1:nClasses,
       gravitationalVector(n,j) = gravitation(x(n,:),j,X,Y,W); 
    end
end

[maxValues predictedPatterns] = max(gravitationalVector');
predictedPatterns = predictedPatterns';

end