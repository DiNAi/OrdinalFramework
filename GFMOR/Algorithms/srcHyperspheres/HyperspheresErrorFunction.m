function [error] = HyperspheresErrorFunction(W,X,Y,S,K,J)

% Extract the parameters
Omega = vec2mat(W(:,1:S*K),K);
Theta = W(:,S*K+1);
Paddings = W(:,S*K+2:end);
Thresholds = zeros(1,J-1);

% Obtain the real thresholds
Thresholds(1) = Theta^(2); 
for i=2:J-1,
       Thresholds(i) = Thresholds(i-1) + exp(Paddings(i-1));
end


% Compute the basis Function Space
BasisFunctionSpace = 1./(1.+exp(-Omega*X));
BasisFunctionSpaceSquare = repmat(sum(BasisFunctionSpace.^(2)),J-1,1);
% Compute the square of the thresholds (in a matricial form)
ThresholdsSquare = repmat(Thresholds,size(X,2),1).^(2)';

% Compute the f_{j}(x_{n}) function
F = BasisFunctionSpaceSquare - ThresholdsSquare;

% Cumulative probability
CumulativeProbability = 1./(1.+exp(F));
CumulativeProbability = [CumulativeProbability;ones(1,size(X,2))];

% Compute the posterior probability
PosteriorProbability = zeros(J,size(X,2));
PosteriorProbability(1,:) = CumulativeProbability(1,:);

for i=2:J
   PosteriorProbability(i,:) = CumulativeProbability(i,:)- CumulativeProbability(i-1,:);
end

error = sum(sum((PosteriorProbability-Y).^(2)));