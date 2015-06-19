load('ionosphere.mat')

[nf nc] = size(X);
AY = zeros(size(Y));
for i = 1 : nf
    if(strcmp('g',Y(i))==1)
      AY(i)= 1;
    end
end
SOAP(X,AY)