classdef SVCKernelSMOTE < Algorithm
    %SVCKernelSMOTE Support Vector Classifier using 1Vs1 approach for Imbalanced
    % data
    %   This class derives from the Algorithm Class and implements the
    %   SVCKernelSMOTE method with a preprocessing to handle imbalance. 
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
        % Function: SVCKernelSMOTE (Public Constructor)
        % Description: It constructs an object of the class
        %               SVCKernelSMOTE and sets its characteristics.
        % Type: Void
        % Arguments: 
        %           kernel--> Type of Kernel function
        %           opt--> Type of optimization used in the method.
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function obj = SVCKernelSMOTE(kernel)
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
        %           for the SVCKernelSMOTE method and kernel parameters
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function [model_information] = runAlgorithm(obj,train, test, parameters)
            	addpath(fullfile('Algorithms','libsvm-weights-3.12','matlab'));
                param.C = parameters(1);
                param.k = parameters(2);
                
                c1 = clock;
                                
                kernelMatrixTrain = computeKernelMatrix(train.patterns', train.patterns', 'rbf', param.k);
                kernelMatrixTest = computeKernelMatrix(train.patterns', test.patterns', 'rbf', param.k);
            
                [featureSpaceTrain featureSpaceTest] = obj.calculateEmpiricalFeatureSpace(kernelMatrixTrain,kernelMatrixTest);
                
                [model,newTrain,newKernelMatrix,sample] = obj.train(train,featureSpaceTrain,kernelMatrixTrain,param);
                c2 = clock;
                model_information.trainTime = etime(c2,c1);
                
                c1 = clock;
                [model_information.projectedTrain,model_information.predictedTrain] = obj.test(newKernelMatrix,newTrain,model);
                
                k1 = computeKernelMatrix(featureSpaceTest', sample', 'linear', param.k);
                newkernelMatrixTest = [kernelMatrixTest; k1'];     
                
                [model_information.projectedTest,model_information.predictedTest] = obj.test(newkernelMatrixTest,test,model);
                c2 = clock;
                model_information.testTime = etime(c2,c1);

                model.algorithm = 'SVCKernelSMOTE';
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
        
        function [model,newTrain,newKernelMatrix,sample]= train( obj,train,featureSpaceTrain,kernelMatrixTrain, param)
                        
            [newTrain,newKernelMatrix,sample] = obj.applySMOTE(train,featureSpaceTrain,kernelMatrixTrain,param);
            options = ['-t 4 -c ' num2str(param.C) ' -q']; 
            weights = ones(size(newTrain.targets));
            model = svmtrain(weights,newTrain.targets, [(1:size(newKernelMatrix,1))' newKernelMatrix], options); 

        end
        
        function [projected, testTargets]= test(obj,kernelMatrix,test,model)           
            [testTargets, acc, projected] = svmpredict(test.targets,[(1:numel(test.targets))' kernelMatrix'],model, ''); 

        end      
        
        function [newTrain,newKernelMatrix,sample] = applySMOTE(obj,train,featureSpace,kernelMatrix,param)
            
            uniqueTargets = unique(train.targets);
            nOfPattPerClass = sum(repmat(train.targets,1,size(uniqueTargets,1))==repmat(uniqueTargets',size(train.targets,1),1));
            
            [minValue,indexMin] = min(nOfPattPerClass);
            [maxValue,indexMax] = max(nOfPattPerClass);
            
            sample = SMOTE((featureSpace(train.targets==uniqueTargets(indexMin),:))',maxValue-minValue);
            k1 = computeKernelMatrix(featureSpace', sample, 'linear', param.k);
            k2 = computeKernelMatrix(sample, sample, 'linear', param.k);
            
            newKernelMatrix = [kernelMatrix, k1; k1', k2];
            sample = sample';
            newTrain.patterns = train.patterns;
            newTrain.targets = [train.targets; uniqueTargets(indexMin)*ones(maxValue-minValue,1)];
            
        end
        
        function [newTrain newTest] = calculateEmpiricalFeatureSpace(obj,kernelMatrixTrain,kernelMatrixTest)
                nPattern = size(kernelMatrixTrain,2);
                [V,D] = eig(kernelMatrixTrain);
                a = sum(D);
                %a = a((numel(a)-20):end);
                autovalpositivos = diag(1./sqrt(a(a>0)));
                autovectores = V(:,a>0);
                producto = autovalpositivos*autovectores';
                newTrain = zeros(nPattern,size(autovalpositivos,1));
                for j=1:nPattern
                       newTrain(j,:) = producto*kernelMatrixTrain(:,j);
                end
                nPatternTest = size(kernelMatrixTest,2);
                newTest = zeros(nPatternTest,size(autovalpositivos,1));
                for j=1:nPatternTest
                       newTest(j,:) = producto*kernelMatrixTest(:,j);
                end
%                  b = autovalpositivos * autovalpositivos;
%                  c = autovectores * b * autovectores';
%                k1 = computeKernelMatrix(new', new', 'linear', param);

        end 
    end
end
