classdef LMNN < Algorithm
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
        name_parameters = {'S','k'}
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
		
        function obj = LMNN(opt)
            obj.name = 'learning the metric for k-Nearest Neighbors using NN';
            % This method don't use kernel functions.
            obj.kernelType = 'no';
        end
		

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: defaultParameters (Public)
        % Description: It assigns the parameters of the algorithm to a default value.
        % Type: Void
        % Arguments: 
        %           No arguments for this function.
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
        function obj = defaultParameters(obj)
            obj.parameters.k = {1,2,3,4,5};
            obj.parameters.S = {5,10,15,20};
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
		
        function model_information = runAlgorithm(obj,training, testing,parameters)

                param.k = parameters(1);
                param.S = parameters(2);
                
                
                c1 = clock;
                [model]= obj.train(training,param );
                % Time information for training
                c2 = clock;
                model_information.trainTime = etime(c2,c1);
                
                
                
                c1 = clock;
                [model_information.projectedTrain,model_information.predictedTrain] = obj.test( training.patterns, model);
                [model_information.projectedTest,model_information.predictedTest] = obj.test( testing.patterns, model);
                c2 = clock;
                
                % time information for testing
                model_information.testTime = etime(c2,c1);

                model_information.model = model;
                
        end
        
        function [model]= train( obj,training,parameters)
            
            Linitial = rand(parameters.S,size(training.patterns,2));
            
            [Omega,Det]=lmnn2(training.patterns',(training.targets-1)',1,Linitial,'outdim',parameters.S,'quiet',1,'maxiter',500,'validation',0.3,'checkup',0);
            
            Lambda = Omega'*Omega;
            
            model.mdl = ClassificationKNN.fit(training.patterns, training.targets,'NumNeighbors',parameters.k,'Distance',@(Xi,Xj)ParametrizedMahalanobis(Xi,Xj,Lambda));
            model.parameters = parameters;
        end

    
        function [ projected,testTargets ]= test( obj, testPatterns, model)
                testTargets = predict(model.mdl,testPatterns);
                projected = testTargets;

         end
    end
    
    
end

