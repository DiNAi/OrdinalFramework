classdef POMELM < Algorithm
    % POM Linear Proportional Odd Model for Ordinal Regression
    %   This class derives from the Algorithm Class and implements the
    %   linear POM method. 
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
        name_parameters = {'hiddenN'}
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
		
        function obj = POMELM(opt)
            obj.name = 'Linear Proportional Odds Model with ELM for Ordinal Regression';
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
            obj.parameters.hiddenN = {5,10,20,30,40,50,60,70,80,90,100};
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
            
                param.hiddenN = parameters(1);

                c1 = clock;
                [model]= obj.train( train,param );
                % Time information for training
                c2 = clock;
                model_information.trainTime = etime(c2,c1);
                
                
                
                c1 = clock;
                [model_information.projectedTrain,model_information.predictedTrain] = obj.test( train.patterns, model);
                [model_information.projectedTest,model_information.predictedTest] = obj.test( test.patterns, model);
                c2 = clock;
                
                % time information for testing
                model_information.testTime = etime(c2,c1);

                model_information.model = model;
                
        end
        
        function [model]= train( obj,train,parameters)
            
                    obj.parameters.hiddenN = parameters.hiddenN;
                    
                    nOfClasses = numel(unique(train.targets));
                    
                    % Configure the hidden layer
                    InputWeight=rand(obj.parameters.hiddenN,size(train.patterns,2))*2-1;

                    BiasofHiddenNeurons=rand(obj.parameters.hiddenN,1);
                    tempH=InputWeight*train.patterns';
                    ind=ones(1,size(train.patterns,1));
                    BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                    tempH=tempH+BiasMatrix;
                    H = 1 ./ (1 + exp(-tempH));
                    
                    % Training the ordinal linear model
                    % Obtain coefficients of the ordinal regression model
                    betaHatOrd = mnrfit(H',train.targets,'model','ordinal','interactions','off');
                    
                    model.thresholds = betaHatOrd(1:nOfClasses-1);
                    model.projection = betaHatOrd(nOfClasses:end);
                    model.inputWeight = InputWeight;
                    model.bias = BiasofHiddenNeurons;
                    model.algorithm = 'POM';
                    model.hiddenN = obj.parameters.hiddenN;
                    model.parameters = parameters;
                    % Estimated Probabilities
                    %pHatOrd = mnrval(betaHatOrd,trainPatterns,'model','ordinal','interactions','off');
        end

    
        function [ projected,testTargets ]= test( obj, testPatterns, model)
                numClasses = size(model.thresholds,1)+1;
                
                tempH=model.inputWeight*testPatterns';
                ind=ones(1,size(testPatterns,1));
                BiasMatrix=model.bias(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                tempH=tempH+BiasMatrix;
                H = 1 ./ (1 + exp(-tempH));
                projected =  - model.projection' * H;
                
                % We calculate the projected patterns - each thresholds, and then with
                % the following decision rule we can induce the class each pattern
                % belows.
                projected2 = repmat(projected, numClasses-1,1);
                projected2 = projected2 - model.thresholds*ones(1,size(projected2,2));

                % Asignation of the class
                % f(x) = max {Wx-bk<0} or Wx - b_(K-1) > 0
                wx=projected2;

                % The procedure for that is the following:
                % We assign the values > 0 to NaN
                wx(wx(:,:)>0)=NaN;

                % Then, we choose the bigger one.
                [maximum,testTargets]=max(wx,[],1);

                % If a max is equal to NaN is because Wx-bk for all k is >0, so this
                % pattern below to the last class.
                testTargets(isnan(maximum(:,:)))=numClasses;

                testTargets = testTargets';

         end
    end
    
end

