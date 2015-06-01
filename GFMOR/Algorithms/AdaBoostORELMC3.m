classdef AdaBoostORELMC3 < Algorithm
    % Ordinal Neural Network Trained by AdaBoost.OR ELM
    %   This class derives from the Algorithm Class and implements the
    %   linear ORNNELM method. 
    %   Characteristics: 
    %               -Kernel functions: No
    %               -Ordinal: Yes
    
    properties
		
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Variable: parameters (Private)
        % Description: No parameters for this algorithm
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        parameters
        name_parameters = {'C', 'k'}
    end
    
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: POM (Public Constructor)
        % Description: It constructs an object of the class POM and sets its
        %               characteristics.
        % Type: Void
        % Arguments:
        %           No Parameters
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
        function obj = AdaBoostORELMC3(opt)
            obj.name = 'AdaBoost and ELM Ordinal Regression (C3 Cost)';
            % This method don't use kernel functions.
            obj.kernelType = 'no';
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
            obj.parameters.C =  10.^(-3:1:3);
            obj.parameters.k = 10.^(-3:1:3);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: runAlgorithm (Public)
        % Description: This function runs the corresponding algorithm, fitting the
        %               model, and testing it in a dataset. It also calculates some
        %               statistics as CCR, Confusion Matrix, and others. 
        % Type: It returns a set of statistics (Struct) 
        % Arguments: 
        %           train --> trainning data for fitting the model
        %           test --> test data for validation
        %           parameter --> No Parameters
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
        function model_information = runAlgorithm(obj,train, test,parameters)
            
                param.C = parameters(1);
                param.k = parameters(2);
                
                %Generate the ELM encodings label
                train.uniqueTargets = unique([test.targets ;train.targets]);
                test.uniqueTargets = train.uniqueTargets;
                train.nOfClasses = max(train.uniqueTargets);
                test.nOfClasses = train.nOfClasses;                
                train.nOfPatterns = length(train.targets);
                test.nOfPatterns = length(test.targets);
                
                train.dim = size(train.patterns,2);
                test.dim = train.dim;
                
                
                [train, test] = obj.labelToOrelm(train,test);
                
                train.uniqueTargetsOrelm = unique([test.targetsOrelm ;train.targetsOrelm],'rows');
                test.uniqueTargetsOrelm = train.uniqueTargetsOrelm;
                
                c1 = clock;
                [model]= obj.train( train,param );
                % Time information for training
                c2 = clock;
                model_information.trainTime = etime(c2,c1);
                
                
                c1 = clock;
                [model_information.projectedTrain,model_information.predictedTrain] = obj.test( train.patterns, model,train.patterns);
                [model_information.projectedTest,model_information.predictedTest] = obj.test( test.patterns, model,train.patterns);
                c2 = clock;
                
                % time information for testing
                model_information.testTime = etime(c2,c1);

                model_information.model = model;
                
        end
        
        function [model]= train( obj,train,parameters)
            
                    obj.parameters.C = parameters.C;
                    obj.parameters.k = parameters.k;
                    
                    nOfClasses = numel(unique(train.targets));
                    
                    T = train.targetsOrelm; %1 of J encoding training dataset (hat(Y))
                    
                    % Configure the hidden layer
                    Omega_train = kernel_matrix(train.patterns,'RBF_kernel', parameters.k);
                    n = size(train.patterns,1);
                    
                    nOfEnsembles = 25;
                    
                    weights = ones(train.nOfPatterns,1);
                    weights = weights./train.nOfPatterns;
                    alpha = zeros(nOfEnsembles,1);
                    
                    BetaEnsemble = zeros(n,nOfClasses,nOfEnsembles);
                    
                    for j = 1:nOfEnsembles,
                        % Estimation of the j-th component
                        % BetaEnsemble(:,:,j) = inv(H'*(diag(weights))*H)*H'*(diag(weights))* T;
                        %BetaEnsemble(:,:,j) = ((H'*(diag(weights))*H)\H')*(diag(weights))* T;
                        BetaEnsemble(:,:,j) = (((diag(weights))*Omega_train+speye(n)/parameters.C)\((diag(weights))*T)); 
                        
                        % Compute the error of the j-th component
                        
                        YHat = Omega_train*BetaEnsemble(:,:,j); %predicted labels by training model j
                        
                        [maxValue, locate] = max(YHat');
                        YHat = YHat - repmat(maxValue',1,nOfClasses); 
                        YHat(YHat==0) = 1.0;
                        YHat(YHat< 0) = 0.0; % 1 of J encoding
                        
                        % Cost Estimation
                        [maxVal, numTargets]  = max(T');
                        cost = (abs(numTargets - locate)+1); 
                        cost = cost./(nOfClasses);
                        
                        indicatorFunction = (sum((YHat.*T)'));
                        indicatorFunction(indicatorFunction ==1.0) = NaN;
                        indicatorFunction(indicatorFunction ==0.0) = 1.0;
                        indicatorFunction(isnan(indicatorFunction))= 0.0;
                        
                        %C3
                        errorELM = sum(indicatorFunction.*cost.*weights')/sum(cost.*weights');

                        % Alpha Estimation
                        alpha(j,:) = log((1-errorELM)/errorELM) + log(nOfClasses-1);
                         
                        %C3 
                        % Update weights
                        weights = weights.*(exp(cost.*alpha(j,:).*indicatorFunction))';   
                        
                        % Renormalize weights
                        weights = weights./sum(weights);
                    end
                 
                    % Store Information
                    model.C = obj.parameters.C;
                    model.k = obj.parameters.k;
                    model.alpha = alpha;
                    model.OutputWeight = BetaEnsemble;
                    model.algorithm = 'AdaBoost.OR(ELM)-C3';
                    model.parameters = parameters;

        end

    
        function [ projected,testTargets ]= test( obj, testPatterns, model,trainPatterns)

                
                Omega_test = kernel_matrix(trainPatterns,'RBF_kernel', model.k,testPatterns);
                
                nOfEnsembles = size(model.OutputWeight,3);
                nOfClasses = size(model.OutputWeight,2);
                nOfPatterns = size(testPatterns,1);
                
                YHat = zeros(nOfPatterns,nOfClasses);
                
                for j=1:nOfEnsembles,
                    
                    indicator = (Omega_test'*model.OutputWeight(:,:,j));
                    maxValue = max(indicator');
                    indicator = indicator - repmat(maxValue',1,nOfClasses);
                    indicator(indicator==0) = 1;
                    indicator(indicator< 0) = 0;
                    YHat = YHat + model.alpha(j,:).*indicator;
                end

                [maxVal,finalOutput] = max(YHat');
                
                testTargets = finalOutput';
                projected = finalOutput';

         end
    end
    
        methods(Access = private)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: orelmToLabel (Private)
        % Description: 
        % Type: 
        % Arguments: 
        %           trainSet--> Array of training patterns
        %           testSet--> Array of testing patterns
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



	function [trainSet, testSet] = labelToOrelm(obj,trainSet,testSet)

            %   newTargets = zeros(trainSet.nOfPatterns,trainSet.nOfClasses);
            trainSet.targetsOrelm = zeros(trainSet.nOfPatterns,trainSet.nOfClasses);
            testSet.targetsOrelm = zeros(testSet.nOfPatterns,trainSet.nOfClasses);
            
            for i=1:trainSet.nOfClasses,
                trainSet.targetsOrelm(trainSet.targets==trainSet.uniqueTargets(i),i) = 1;
                testSet.targetsOrelm(testSet.targets==trainSet.uniqueTargets(i),i) = 1;
            end
    end
   end
        
    
end

