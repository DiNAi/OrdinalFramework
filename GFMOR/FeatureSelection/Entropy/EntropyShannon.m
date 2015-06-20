
function [hy] = EntropyShannon(y)

classes = unique(y');
clasesI = classes';
numclases=size(clasesI,1);
hy = 0;

for i=1:numclases
    c=clasesI(i);
    py = sum(y'==c)/size(y,1);
    hy = hy + py *log2(py);
end
hy = -hy;

end