classdef DGCplus < Algorithm
    % GMOR Gravitational Model for Ordinal Regression
    %   This class derives from the Algorithm Class and implements the
    %   GMOR method. 
    
    properties
		
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Variable: parameters (Private)
        % Description: No parameters for this algorithm
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        parameters = []
        name_parameters = {}
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
		
        function obj = DGCplus(algorithm,cost)
            addpath('Algorithms/srcGravitational');
            obj.name = 'Data Gravitational Classification+';
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
            obj.parameters = [];
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
		
        function model_information = runAlgorithm(obj,train, test)
                
                addpath('Algorithms/srcGravitational');
                
                % Get the number of classes and attributes
                nOfClasses = numel(unique(train.targets));
                ClassType = 1:nOfClasses;
                nOfAttributes = size(train.patterns,2);
                nPatterns = size(train.patterns,1);
                
                % Initialization of the parameter
                W = ones(1,(nOfAttributes*nOfClasses));
            
                % First convert the outputs to 0-1 class
                trainingTargets = (LabelFormatConvertion(train.targets',ClassType,1))';
                multiplierClass = zeros(nOfClasses,1);
                
                for j = 1:nOfClasses,
                    multiplierClass(j,:) = (1- ((size(train.patterns(find(trainingTargets(:,j)),:),1)- 1)/nPatterns));
                end

                c1 = clock;
                [model_information] = obj.train(W,train.patterns,train.targets,multiplierClass);
                % Time information for training
                c2 = clock;
                
                model_information.trainTime = etime(c2,c1);
                
                c1 = clock;
                [model_information.predictedTrain] = obj.test( train.patterns, train.patterns, train.targets, model_information.model);
                [model_information.predictedTest] = obj.test( test.patterns, train.patterns, train.targets, model_information.model);
                c2 = clock;
                model_information.projectedTrain = model_information.predictedTrain;
                model_information.projectedTest = model_information.predictedTest;
                
                % time information for testing
                model_information.testTime = etime(c2,c1);
                
        end
        
        function [model_information] = train(obj, Winit,tPatterns,tTargets,multiplierClass)
           addpath('Algorithms/srcGravitational');           
           
           nOfClasses = numel(unique(tTargets));
           nOfAttributes = size(tPatterns,2);
                    
           opts.CMA.active = 1;
           opts.Noise.on = 1; 
           opts.DiagonalOnly = 1;
           opts.MaxIter = 800;
           opts.Seed = 100;
           opts.UBounds = zeros((nOfAttributes*nOfClasses),1);
           opts.UBounds(1:nOfAttributes*nOfClasses,1) = 1;
           opts.LBounds = zeros((nOfAttributes*nOfClasses),1);
           opts.LBounds(1:nOfAttributes*nOfClasses,1) = 0;
                    
           [model_information.model,fval] = cmaes('GravitationalErrorOptimized', Winit, 0.3, opts ,tPatterns,tTargets,multiplierClass); 
           
           
        end
    
        function [predictedPatterns] = test(obj,x,X,Y,W)
           addpath('Algorithms/srcGravitational');
           nEvaluationPatterns = size(x,1);
           nClasses = numel(unique(Y));
           gravitationalVector = zeros(nEvaluationPatterns,nClasses);

           for n = 1:nEvaluationPatterns,
              for j = 1:nClasses,
                 gravitationalVector(n,j) = gravitation(x(n,:),j,X,Y,W); 
              end
           end

           [maxValues predictedPatterns] = max(gravitationalVector');
           predictedPatterns = predictedPatterns';

        end
    end
    
end

