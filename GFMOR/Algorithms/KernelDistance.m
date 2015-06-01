function D = KernelDistance(X,Y,S)

nPatternsY = size(Y,1);
D = zeros(nPatternsY,1);

for i=1:nPatternsY,    
   %Dist = norm(X-Y(i,:));
   %Kernel = exp(-Dist/S.^(2));
   Kernel = (1+dot(X,Y(i,:))).^(S);
   D(i) = 2*(1-Kernel);
end


