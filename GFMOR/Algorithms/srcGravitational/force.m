function [grav] = force(x,j,X,Y,W,maxForce)
%GRAVITATION Summary of this function goes here
%   Detailed explanation goes here

trainingPatterns = X(find(Y(:,j)),:);
nClasses = size(Y,2);
nPatternsClass = size(trainingPatterns,1);
nAttributes = size(trainingPatterns,2);
nPatterns = size(X,1);

grav = 0;
%dgrav = zeros(nAttributes+1,1);

for n = 1:nPatternsClass,
    if (~isequal(x,trainingPatterns(n,:)))
        
        % Obtain the weight and the force for each algorithm
        if(size(W,1) ~= 1)
            % For global optimizer
            weight = W(((j-1)*nAttributes+1):(((j-1)*nAttributes) + nAttributes),1)';
            force = W((nClasses*nAttributes) + j,1);
        else
            % For local optimizer
            weight = W(1,((j-1)*nAttributes+1):(((j-1)*nAttributes) + nAttributes));
            force = W(1,(nClasses*nAttributes) + j);
        end
        
        distance2 = ((weight)*((x-trainingPatterns(n,:)).^2)');
        
        a = (1.0/maxForce).^(1/force);
        distance = sqrt(distance2) + a;
        grav = grav + (1/distance)^(force);
        
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%         if nargout > 1
%             for k=1:nAttributes
%                 dgrav(k) = dgrav(k) - (force)*(1/distance^(force+1))*0.5*1./sqrt(distance2)*((x(k)-trainingPatterns(n,k))^2);
%             end
%                 dgrav(k+1) = dgrav(k+1) + exp(-force*log(distance))*(-log(distance)+1.0/force*1.0/distance*a*log(1.0/maxForce));
%         end
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
end

grav = grav * (1- ((nPatternsClass- 1)/nPatterns));
%dgrav = dgrav * (1- ((nPatternsClass- 1)/nPatterns));

end

