classdef SVCSMOTE < Algorithm
    %SVCSMOTE Support Vector Classifier using 1Vs1 approach for Imbalanced
    % data
    %   This class derives from the Algorithm Class and implements the
    %   SVCSMOTE method with a preprocessing to handle imbalance. 
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
        name_parameters = {'C','k'}
        parameters
    end
    
    methods
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: SVCSMOTE (Public Constructor)
        % Description: It constructs an object of the class
        %               SVCSMOTE and sets its characteristics.
        % Type: Void
        % Arguments: 
        %           kernel--> Type of Kernel function
        %           opt--> Type of optimization used in the method.
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function obj = SVCSMOTE(kernel)
            obj.name = 'Support Vector Machine Classifier with 1vs1 paradigm, applying a preprocessing for handling imbalance';
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
        %           for the SVCSMOTE method and kernel parameters
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function [model_information] = runAlgorithm(obj,train, test, parameters)
            	addpath(fullfile('Algorithms','libsvm-weights-3.12','matlab'));
                param.C = parameters(1);
                param.k = parameters(2);
                
                c1 = clock;
                
                uniqueTargets = unique(train.targets);
                nOfPattPerClass = sum(repmat(train.targets,1,size(uniqueTargets,1))==repmat(uniqueTargets',size(train.targets,1),1));
            
                [minValue,indexMin] = min(nOfPattPerClass);
                [maxValue,indexMax] = max(nOfPattPerClass);
            
                sample = SMOTE((train.patterns(train.targets==uniqueTargets(indexMin),:))',maxValue-minValue);
                
                newTrain.patterns = [train.patterns; sample'];
                newTrain.targets = [train.targets; uniqueTargets(indexMin)*ones(maxValue-minValue,1)];                                               
                
                model = obj.train(newTrain,param);
                c2 = clock;
                
                model_information.trainTime = etime(c2,c1);
                
                c1 = clock;
                [model_information.projectedTrain,model_information.predictedTrain] = obj.test(newTrain,model);
                [model_information.projectedTest,model_information.predictedTest] = obj.test(test,model);
                c2 = clock;
                model_information.testTime = etime(c2,c1);

                model.algorithm = 'SVCSMOTE';
                model.parameters = param;
                model_information.model = model;

            	rmpath(fullfile('Algorithms','libsvm-weights-3.12','matlab'));

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: train (Public)
        % Description: This function train the model for
        %               the KDLOR algorithm.
        % Type: [Array, Array]
        % Arguments: 
        %           trainPatterns --> Trainning data for 
        %                              fitting the model
        %           testTargets --> Training targets
        %           parameters --> Penalty coefficient C 
        %           for the KDLOR method and kernel parameters
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [model]= train( obj, train , param)
            weights = ones(size(train.targets));
            options = ['-t 2 -c ' num2str(param.C) ' -g ' num2str(param.k) ' -q'];
            model = svmtrain(weights, train.targets, train.patterns, options);

        end
        
        function [projected, testTargets]= test(obj,test, model)
                [testTargets, acc, projected] = svmpredict(test.targets,test.patterns,model, '');

        end    
    end
end
