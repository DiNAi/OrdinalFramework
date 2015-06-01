classdef Fscore < Metric

    methods
        function obj = Fscore()
                obj.name = 'Fscore';
        end
    end
    
    methods(Static = true)
	    
        function fscore = calculateMetric(argum1,argum2)
            if nargin == 2,
                argum1 = confusionmat(argum1,argum2);
            end
            ccr_class = zeros(size(argum1,1),1);
            precision = zeros(size(argum1,1),1);
            
            n = sum(argum1,2);
            n2 = sum(argum1,1);
            
            for i=1:size(argum1,1),
                ccr_class(i) = argum1(i,i)/n(i);
                precision(i) = argum1(i,i)/n2(i);
                if isnan(precision(i))
                   precision(i) = 0; 
                end
                if isnan(ccr_class(i))
                   ccr_class(i) = 0; 
                end
            end
	    meanccr = mean(ccr_class,1);
        meanprec = mean(precision,1);
        fscore = (2*meanprec*meanccr)/(meanccr+meanprec);
        end


	function value = calculateCrossvalMetric(argum1,argum2)
                value = 1 - Fscore.calculateMetric(argum1,argum2);
        end
        
    end
            
    
end

