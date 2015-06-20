% Information Gain Calculator
% Jeffrey Jedele, 2011
%[gain, max_gain_feature] = infogain([1 1 2 3 3 3 2 1 1 3 1 2 2 3]',[1 1 2
%2 2 1 2 1 2 2 2 2 2 1]')
%http://www.cs.csi.cuny.edu/~imberman/ai/Entropy%20and%20Information%20Gain.htm
function [max_gain_feature  gain] = infogain(x,y)

max_gain_feature = 0;
info_gains = zeros(1, size(x,2));

yt=y';

% calculate H(y)
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

% iterate over all features (columns)
for col=1:size(x,2)
    
    features = unique(x(:,col))';
    
    % calculate entropy
    hyx = 0;
    numfeatures = size(features,2);
    for j=1 :numfeatures
        f=features(j);
        pf = sum(x(:,col)==f)/size(x,1);
        yf = yt(find(x(:,col)==f));
        
        % calculate h for classes given feature f
        yclasses = unique(yf)';
        hyf = 0;
        numyclasses = size(yclasses,1);
        for k= 1 : numyclasses
            yc= yclasses(k);
            pyf = sum(yf==yc)/size(yf,2);
            hyf = hyf + pyf*log2(pyf);
        end
        hyf = -hyf;
        hyx = hyx + pf * hyf;
    end
    
    info_gains(col) = hy - hyx;
end

[gain, max_gain_feature] = max(info_gains);

end