classdef GM  < Metric
    methods
        function obj = GM()
                obj.name = 'Geometric Mean';
        end
    end
    
    methods(Static = true)
	    
        function [gm] = calculateMetric(argum1,argum2)
            if nargin == 2,
                argum1 = confusionmat(argum1,argum2);
            end
            nC = size(argum1,1);
            gm = 1;

            for ii=1:nC
                gm = gm*argum1(ii,ii)/sum(argum1(ii,:));
            end
            gm = gm.^(1/nC);
            if isnan(gm)
               gm = 0;
            end
        end

	function value = calculateCrossvalMetric(argum1,argum2)
            if nargin == 2,
                value = 1 - GM.calculateMetric(argum1,argum2);
            else
                value = 1 - GM.calculateMetric(argum1);
            end
        end
        
    end
            
    
end

