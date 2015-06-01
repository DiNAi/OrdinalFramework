function [grav] = gravitationRiccardi(x,j,X,Y,W,v,maxForce)
%GRAVITATION Summary of this function goes here
%   Detailed explanation goes here

a = (1/maxForce)^(1/v(j));

nClasses = numel(unique(Y));
ClassType = 1:nClasses;
Yzeroone = (LabelFormatConvertion(Y',ClassType,1))';

trainingPatterns = X(find(Yzeroone(:,j)),:);
nPatternsClass = size(trainingPatterns,1);
nAttributes = size(trainingPatterns,2);
nPatterns = size(X,1);

grav = 0;
%dgrav = zeros(nAttributes+1,1);

for n = 1:nPatternsClass,
    if (~isequal(x,trainingPatterns(n,:)))    
        weight = W(((j-1)*nAttributes+1):(((j-1)*nAttributes) + nAttributes),1)';
        distance = sqrt((weight)*((x-trainingPatterns(n,:)).^2)');
        grav = grav + (1/(distance+a))^(v(j));
    end
end

grav = grav * (1- ((nPatternsClass- 1)/nPatterns));

end

