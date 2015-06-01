classdef HyperspheresModel < Algorithm
    % GMOR Gravitational Model for Ordinal Regression
    %   This class derives from the Algorithm Class and implements the
    %   GMOR method. 
    
    properties
		
        optimizer = 'local';
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Variable: parameters (Private)
        % Description: No parameters for this algorithm
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        parameters = []
        name_parameters = {'S'}
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
		
        function obj = HyperspheresModel(algorithm)
            addpath('Algorithms/srcHyperspheres');
            obj.name = 'Hyperspheres Ordinal Neural Network Model';
            if(nargin > 0)
                obj.optimizer = algorithm;
            else
                obj.optimizer = 'local';
            end
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
		
        function model_information = runAlgorithm(obj,train, test,parameters)
                
                addpath('Algorithms/srcHyperspheres');
                
                % Get the number of classes and attributes
                nOfClasses = numel(unique(train.targets));
                ClassType = 1:nOfClasses;
                nOfAttributes = size(train.patterns,2);
                param.S = parameters(1);
                nPatterns = size(train.patterns,1);
                
                % Initialization of the parameter vector
                W = rand(1,(param.S *nOfAttributes) + (nOfClasses-1));
                
                % First convert the outputs to 0-1 class
                trainingTargets = (LabelFormatConvertion(train.targets',ClassType,1))';
                testTargets = (LabelFormatConvertion(test.targets',ClassType,1))';
                
                c1 = clock;
                [model] = obj.train(W,train.patterns',trainingTargets',param,nOfAttributes,nOfClasses);
                % Time information for training
                c2 = clock;
                
                model_information.trainTime = etime(c2,c1);

                c1 = clock;                
                [model_information.predictedTrain] = obj.test( train.patterns', trainingTargets',model.W,param.S,nOfAttributes,nOfClasses);
                [model_information.predictedTest] = obj.test( test.patterns', testTargets', model.W,param.S,nOfAttributes,nOfClasses);
                c2 = clock;
                model_information.projectedTrain = model_information.predictedTrain;
                model_information.projectedTest = model_information.predictedTest;
                
                % time information for testing
                model_information.testTime = etime(c2,c1);
                model_information.model = model;
                
        end
        
        function [model] = train(obj, Winit,tPatterns,tTargets,parame,K,J)
           addpath('Algorithms/srcHyperspheres');           
           switch lower(obj.optimizer)
                case {'local'}
                    optsLocal.Display = 'iter';
                    optsLocal.GradObj = 'on';
                    optsLocal.MaxIter = 100;
                    optsLocal.TolX = 1e-10;
                    optsLocal.Algorithm = 'interior-point';
                    %optsLocal.Algorithm = 'trust-region-reflective';
                    f = @(Winit)HyperspheresErrorFunction(Winit,tPatterns,tTargets,parame.S,K,J);
                    model.W = Winit;
                    model.parameters = parame;
                    [model.W,fval] = fminunc(f,Winit);
                case {'global'}
                    %opts.LogModulo = 1;
                    nOfClasses = size(tTargets,2);
                    nOfAttributes = size(tPatterns,2);
                    
                    opts.CMA.active = 1;
                    opts.Noise.on = 1; 
                    opts.DiagonalOnly = 1;
                    opts.MaxIter = 800;
                   opts.Seed = 100;
                    %opts.DispFinal  = 0;
                    %opts.DispModulo = 0;
                    %opts.SaveVariables = 0;
                    %opts.SaveFilename = 0; 
                    %opts.LogModulo = 0;
                    %opts.LogTime   = 0;

                    
                    [model_information.model,fval] = cmaes('HyperspheresErrorFunction', Winit, 0.3, opts ,tPatterns',tTargets',parame.S,K,J); 
                    model.W = Winit;
                    model.parameters = parame;
           end
           

           
        end
    
        function [predictedPatterns] = test(obj,X,Y,W,S,K,J)
           addpath('Algorithms/srcHyperspheres');
           % Extract the parameters
            Omega = vec2mat(W(:,1:S*K),K);
            Theta = W(:,S*K+1);
            Paddings = W(:,S*K+2:end);
            Thresholds = zeros(1,J-1);

            % Obtain the real thresholds
            Thresholds(1) = Theta^(2); 
            for i=2:J-1,
                Thresholds(i) = Thresholds(i-1) + exp(Paddings(i-1));
            end


            % Compute the basis Function Space
            BasisFunctionSpace = 1./(1.+exp(-Omega*X));
            BasisFunctionSpaceSquare = repmat(sum(BasisFunctionSpace.^(2)),J-1,1);
            % Compute the square of the thresholds (in a matricial form)
            ThresholdsSquare = repmat(Thresholds,size(X,2),1).^(2)';

            % Compute the f_{j}(x_{n}) function
            F = BasisFunctionSpaceSquare - ThresholdsSquare;

            % Cumulative probability
            CumulativeProbability = 1./(1.+exp(F));
            CumulativeProbability = [CumulativeProbability;ones(1,size(X,2))];

            % Compute the posterior probability
            PosteriorProbability = zeros(J,size(X,2));
            PosteriorProbability(1,:) = CumulativeProbability(1,:);

            for i=2:J
                PosteriorProbability(i,:) = CumulativeProbability(i,:)- CumulativeProbability(i-1,:);
            end
            
            [maxValues predictedPatterns] = max(PosteriorProbability);
           predictedPatterns = predictedPatterns';

        end
    end
    
end

