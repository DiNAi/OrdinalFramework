classdef Deviation  < Metric

    methods
        function obj = Deviation()
                obj.name = 'D';
        end
    end
    
    methods(Static = true)
	    
        function D = calculateMetric(argum1,argum2)
            if nargin == 2,
                argum1 = confusionmat(argum1,argum2);
            end
            ccr = sum(diag(argum1)) / sum(sum(argum1));
            ccr_class = zeros(size(argum1,1),1);
            deviation_class = zeros(size(argum1,1),1);
            n = sum(argum1,2);
            Ntotal = sum(sum(argum1,2));
            for i=1:size(argum1,1),
                if n(i) > 0,
                    ccr_class(i) = argum1(i,i)/n(i);
                    deviation_class(i) = ((ccr_class(i)-ccr)^(2))*(n(i)/Ntotal);
                else  
                    ccr_class(i) = 0;
                    deviation_class(i) = 0;
                    
                end;
                    
            end
            
	    D = sum(deviation_class);
        UpperBound = ccr - (ccr^2);
        
        D = sqrt(D/UpperBound);
        
        
        

        end


	function value = calculateCrossvalMetric(argum1,argum2)
                value = Deviation.calculateMetric(argum1,argum2);
        end
        
    end
            
    
end

