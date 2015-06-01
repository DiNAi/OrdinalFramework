classdef SVORIMLIBSVMPLUS < Algorithm
    % Support Vector Machines for Ordinal Regression with Implicit
    % Constraints but using LIBSVM adapting thresholds 
    %   This class derives from the Algorithm Class and implements the
    %   SVORIMLIBSVM method. 
    %   Characteristics: 
    %               -Kernel functions: Yes
    %               -Ordinal: Yes
    %               -Parameters: 
    %                       -C: Penalty coefficient
    %                       -Others (depending on the kernel choice)
    
    properties
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Variable: parameters (Public)
        % Type: Struct
        % Description: This variable keeps the values for 
        %               the C penalty coefficient and the 
        %               kernel parameters
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        parameters
        name_parameters = {'C','k'}
    end
    
    methods
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: SVORIMLIBSVM (Public Constructor)
        % Description: It constructs an object of the class
        %               SVORIMLIBSVM and sets its characteristics.
        % Type: Void
        % Arguments: 
        %           kernel--> Type of Kernel function
        %           opt--> Type of optimization used in the method.
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function obj = SVORIMLIBSVMPLUS(kernel)
            obj.name = 'Support Vector Machines for Ordinal Regression with Implicit Constraints but using LIBSVM adapting thresholds (SVORIMLIBSVMPLUS)';
            if(nargin ~= 0)
                 obj.kernelType = kernel;
            else
                obj.kernelType = 'rbf';
            end
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: defaultParameters (Public)
        % Description: It assigns the parameters of the 
        %               algorithm to a default value.
        % Type: Void
        % Arguments: 
        %           No arguments for this function.
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function obj = defaultParameters(obj)
            obj.parameters.C = 10.^(-3:1:3);
            obj.parameters.k = 10.^(-3:1:3);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: runAlgorithm (Public)
        % Description: This function runs the corresponding
        %               algorithm, fitting the model, and 
        %               testing it in a dataset. It also 
        %               calculates some statistics as CCR,
        %               Confusion Matrix, and others. 
        % Type: It returns a set of statistics (Struct) 
        % Arguments: 
        %           Train --> Trainning data for fitting the model
        %           Test --> Test data for validation
        %           parameters --> Penalty coefficient C 
        %           for the SVORIMLIBSVMPLUS method and kernel parameters
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function [model_information] = runAlgorithm(obj,train, test, parameters)
            	addpath(fullfile('Algorithms','libsvm-rank-2.81','matlab'));
                param.C = parameters(1);
                param.k = parameters(2);
                
                c1 = clock;
                model = obj.train(train,param);
                [model_information.projectedTrain,model_information.predictedTrain] = obj.test(train,model);
		model.rho(2:size(model.rho,1)) = (SVORIMLIBSVMPLUS.adaptThresholds(train.targets,model_information.projectedTrain))';
                c2 = clock;
                model_information.trainTime = etime(c2,c1);
                
                c1 = clock;
                [model_information.projectedTest,model_information.predictedTest] = obj.test(test,model);
                c2 = clock;
                model_information.testTime = etime(c2,c1);

                model.algorithm = 'SVORIMLIBSVMPLUS';
                model.parameters = param;
                model_information.model = model;

            	rmpath(fullfile('Algorithms','libsvm-rank-2.81','matlab'));

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: train (Public)
        % Description: This function train the model for
        %               the SVORIMLIBSVM algorithm.
        % Type: [Array, Array]
        % Arguments: 
        %           trainPatterns --> Trainning data for 
        %                              fitting the model
        %           testTargets --> Training targets
        %           parameters --> Penalty coefficient C 
        %           for the SVORIMLIBSVM method and kernel parameters
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [model]= train( obj, train , param)
            options = ['-s 6 -t 2 -c ' num2str(param.C) ' -g ' num2str(param.k) ' -q'];
            model = svmtrain(train.targets, train.patterns, options);

        end
        
        function [projected, testTargets]= test(obj,test, model)
                [testTargets, acc, projected] = svmpredict(test.targets,test.patterns,model, '');

        end      
    end

    methods (Static)
        function [newThresholds] = adaptThresholds(targets, projections)
            
            numClasses = size(unique(targets),1);
            patternPerClass = sum(repmat(targets,1,numClasses-1) == repmat(1:(numClasses-1),size(targets,1),1));
            projectionsSorted = sort(projections);
            indexFrontiers = (cumsum(patternPerClass));
            frontierProjections = projectionsSorted(indexFrontiers);
            frontierProjectionsOneMore = projectionsSorted(indexFrontiers+1);
            newThresholds = (frontierProjections + frontierProjectionsOneMore)./ 2;
            
        end
        
    end
end
