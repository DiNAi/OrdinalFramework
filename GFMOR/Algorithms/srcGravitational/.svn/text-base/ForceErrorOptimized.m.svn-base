function [y,grad] = ForceErrorOptimized(W,X,Y,CTotal,multiplierClass,maxForce)

X entradas
Y Salidas
W cromosoma

1) Calcular H
2) calculas B
3) Calculas Yhat
4) error = rmse

% Define the main parameters of the problem
nClasses = size(Y,2);
nAttributes = size(X,2);
nPatterns = size(X,1);

ForceMatrix = zeros(nPatterns,nClasses);

if nargout > 1
    nopt = nAttributes*nClasses+nClasses;
    ForceGradientWeights = zeros(nPatterns,nClasses,nAttributes);
    ForceGradientForces = zeros(nPatterns,nClasses);
    grad = zeros(nopt,1);
end

for j = 1:nClasses,
    
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
    
    % Determine the a value (old tolerance)
    a = (1.0/maxForce).^(1/force);    

    Xsub = X(find(Y(:,j)),:);
    weuc = @(XI,XJ,W)(bsxfun(@minus,XI,XJ).^2 * W');
    Dwgt = pdist2(Xsub,X, @(Xi,Xj) weuc(Xi,Xj,weight));
    Dwgt2 = sqrt(Dwgt) + (a * (Dwgt ~= 0));
    Dwgt_power = (Dwgt2).^(force); 

    Sumatori = (1./Dwgt_power);
    ForceMatrix(:,j) = multiplierClass(j,:) .* nansum(Sumatori.*(~isinf(Sumatori)));
        
    if nargout > 1
        for k=1:nAttributes
            nj = size(Xsub,1);
            SumatoriGradientWeights = zeros(nj,nPatterns);
            SumatoriGradientWeights = pdist2(Xsub(:,k),X(:,k));
            SumatoriGradientWeights = (-force).*Sumatori.*1./Dwgt2.*Dwgt.^(-1.0/2.0).*SumatoriGradientWeights.^2.0*0.5;
            ForceGradientWeights(:,j,k) = multiplierClass(j,:) .* nansum(SumatoriGradientWeights.*(~isinf(SumatoriGradientWeights)));
        end
        SumatoriGradientForces = exp(-force*log(Dwgt2)).*(-1.0*log(Dwgt2)+1./(force*Dwgt2)*a*log(1/maxForce));
        ForceGradientForces(:,j)  = multiplierClass(j,:) .* nansum(SumatoriGradientForces.*(~isinf(SumatoriGradientForces)));
    end
    
end

Probabilities = exp(ForceMatrix)./repmat(sum(exp(ForceMatrix)')',1,nClasses);
ErrorMatrix = (Probabilities - Y).^(2) .* (CTotal + Y);

y = sum(sum(ErrorMatrix));
y = y/nPatterns;

if nargout > 1
    for k=1:nopt
        if(k<=nClasses*nAttributes) %derivative with respect the weights
            class = ceil(k/nAttributes);
            index = k-(class-1)*nAttributes;
        else %derivative with respect the exponents
            class = k-nClasses*nAttributes;
            index = nAttributes+1;
        end
        SUM = sum(exp(ForceMatrix)')';
        SUM2 = SUM.^2;
        
        %derivative of the softmax function
        if index<=nAttributes
            D = - exp(ForceMatrix).*repmat(exp(ForceMatrix(:,class)).*ForceGradientWeights(:,class,index),1,nClasses)./repmat(SUM2,1,nClasses);
        else
            D = - exp(ForceMatrix).*repmat(exp(ForceMatrix(:,class)).*ForceGradientForces(:,class),1,nClasses)./repmat(SUM2,1,nClasses);
        end
        if index<=nAttributes
            D(:,class) = D(:,class) + exp(ForceMatrix(:,class)).*ForceGradientWeights(:,class,index)./SUM;
        else
            D(:,class) = D(:,class) + exp(ForceMatrix(:,class)).*ForceGradientForces(:,class)./SUM;
        end
        
        DD = 2.*(Probabilities-Y).*(CTotal+Y).*D;
        grad(k) = sum(sum(DD));
    end
    grad = grad./nPatterns;
end


end
