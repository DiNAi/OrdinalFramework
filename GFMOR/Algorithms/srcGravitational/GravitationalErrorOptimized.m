function [y] = GravitationalErrorOptimized(W,X,Y,multiplierClass)

% Define the main parameters of the problem
nClasses = numel(unique(Y));
ClassType = 1:nClasses;
nAttributes = size(X,2);
nPatterns = size(X,1);

ForceMatrix = zeros(nPatterns,nClasses);

Yzeroone = (LabelFormatConvertion(Y',ClassType,1))';

for j = 1:nClasses,
    
    % For global optimizer
    weight = W(((j-1)*nAttributes+1):(((j-1)*nAttributes) + nAttributes),1)';
    
    Xsub = X(find(Yzeroone(:,j)),:);
  
    weuc = @(XI,XJ,W)(bsxfun(@minus,XI,XJ).^2 * W');
    Dwgt = pdist2(Xsub,X, @(Xi,Xj) weuc(Xi,Xj,weight));
      
    Sumatori = (1./Dwgt);
    ForceMatrix(:,j) = multiplierClass(j,:) .* nansum(Sumatori.*(~isinf(Sumatori)));

end

[maxValues predictedPatterns] = max(ForceMatrix');

y = 1 - (sum(Y==predictedPatterns')/numel(Y));


