function [ indices ] = SOAP( x,y )
%SOAP Summary of this function goes here
%   Detailed explanation goes here
[nf nc] = size(x);
D=[x y];
numclases = unique(y);
resp = zeros(nc,2);

for i = 1 : nc
    resp(i,2)=i;
    D=sortrows(D,i);
    cont =0;
    for j = 2 : nf
        if(D(j,nc+1)== D(j-1,nc+1))
            cont=cont-1;
        end
    end
    resp(i,1)=cont;
end
resp= sortrows(resp,1);
indice= resp(:,2);
%ordenados de mayor a menor
end


