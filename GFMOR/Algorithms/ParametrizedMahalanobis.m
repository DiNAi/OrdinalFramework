function D = ParametrizedMahalanobis(X,Y,Lambda)

nPatternsY = size(Y,1);
D = zeros(nPatternsY,1);

for i=1:nPatternsY,    
   Dist = (X-Y(i,:));
   D(i) = Dist*Lambda*Dist';
end


