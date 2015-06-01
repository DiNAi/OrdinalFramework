classdef GFMOR < Algorithm
    % GMOR Gravitational Model for Ordinal Regression
    %   This class derives from the Algorithm Class and implements the
    %   GMOR method. 
    
    properties
		
        optimizer = 'local';
        cost = 'quadratic';
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
		
        function obj = GFMOR(algorithm,cost)
            addpath('Algorithms/srcGravitational');
            obj.name = 'Generalized Force-based Model for Ordinal Regression';
            if(nargin > 0)
                obj.optimizer = algorithm;
                if(nargin > 1)
                  obj.cost = cost;
                else
                  obj.cost = 'quadratic';
                end
            else
                obj.optimizer = 'local';
                obj.cost = 'quadratic';
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
                W = ones(1,(nOfAttributes*nOfClasses)+nOfClasses);
                W(1,1:nOfAttributes*nOfClasses) = 0.9;
                W(1,(nOfAttributes*nOfClasses)+1:end) = 2.0;
                
                % First convert the outputs to 0-1 class
                trainingTargets = (LabelFormatConvertion(train.targets',ClassType,1))';
                multiplierClass = zeros(nOfClasses,1);
                
                maxForce = 10;
                
                for j = 1:nOfClasses,
                    multiplierClass(j,:) = (1- ((size(train.patterns(find(trainingTargets(:,j)),:),1)- 1)/nPatterns));
                end
                
                switch lower(obj.cost)
                   case {'zero-one'}
                      alpha = 0;
                   case {'absolute'}
                      alpha = 1;
                   case {'quadratic'}
                      alpha = 2;
                end
                % Generate the cost matrix and the total cost matrix
                CMatrix = abs(repmat(1:nOfClasses,nOfClasses,1) - repmat((1:nOfClasses)',1,nOfClasses));
                CTotal = (trainingTargets * CMatrix).^(alpha);
                
                c1 = clock;
                [model_information] = obj.train(W,train.patterns,trainingTargets,CTotal,multiplierClass,maxForce);
                % Time information for training
                c2 = clock;
                
                model_information.trainTime = etime(c2,c1);
                
                c1 = clock;
                [model_information.predictedTrain] = obj.test( train.patterns, train.patterns, trainingTargets, model_information.model,maxForce);
                [model_information.predictedTest] = obj.test( test.patterns, train.patterns, trainingTargets, model_information.model,maxForce);
                c2 = clock;
                model_information.projectedTrain = model_information.predictedTrain;
                model_information.projectedTest = model_information.predictedTest;
                
                % time information for testing
                model_information.testTime = etime(c2,c1);
                
        end
        
        function [model_information] = train(obj, Winit,tPatterns,tTargets,CTotal,multiplierClass,maxForce)
           addpath('Algorithms/srcGravitational');           
           switch lower(obj.optimizer)
                case {'none'}
                    % Base Model
                    model_information.model = Winit;
                case {'local'}
                    optsLocal.Display = 'iter';
                    optsLocal.GradObj = 'on';
                    optsLocal.MaxIter = 100;
                    optsLocal.TolX = 1e-10;
                    %optsLocal.Algorithm = 'interior-point';
                    optsLocal.Algorithm = 'trust-region-reflective';
                    f = @(Winit)ForceErrorOptimized(Winit,tPatterns,tTargets,CTotal,multiplierClass,maxForce);
                    nOfClasses = size(tTargets,2);
                    nOfAttributes = size(tPatterns,2);
                    lb = ones(1,(nOfAttributes*nOfClasses)+nOfClasses);
                    ub = ones(1,(nOfAttributes*nOfClasses)+nOfClasses);
                    lb(1,:) = 0.0;
                    lb(1,(nOfAttributes*nOfClasses)+1:end) = 1.0;
                    ub(1,1:(nOfAttributes*nOfClasses)) = 1.0;
                    ub(1,(nOfAttributes*nOfClasses)+1:end) = 4.0;
                    [model_information.model,fval] = fmincon(f,Winit,[],[],[],[],lb,ub,[],optsLocal);
                case {'global'}
                    %opts.LogModulo = 1;
                    nOfClasses = size(tTargets,2);
                    nOfAttributes = size(tPatterns,2);
                    
                    opts.CMA.active = 1;
                    opts.Noise.on = 1; 
                    opts.DiagonalOnly = 1;
                    opts.MaxIter = 800;
                    opts.Seed = 100;
                    opts.UBounds = zeros((nOfAttributes*nOfClasses)+nOfClasses,1);
                    opts.UBounds(1:nOfAttributes*nOfClasses,1) = 1;
                    opts.UBounds((nOfAttributes*nOfClasses)+1:end,1) = 4;
                    opts.LBounds = zeros((nOfAttributes*nOfClasses)+nOfClasses,1);
                    opts.LBounds(1:nOfAttributes*nOfClasses,1) = 0;
                    opts.LBounds((nOfAttributes*nOfClasses)+1:end,1) = 1;
                    %opts.DispFinal  = 0;
                    %opts.DispModulo = 0;
                    %opts.SaveVariables = 0;
                    %opts.SaveFilename = 0; 
                    %opts.LogModulo = 0;
                    %opts.LogTime   = 0;

                    
                    [model_information.model,fval] = cmaes('ForceErrorOptimized', Winit, 0.3, opts ,tPatterns,tTargets,CTotal,multiplierClass,maxForce); 
           end
           
           
        end
    
        function [predictedPatterns] = test(obj,x,X,Y,W,maxForce)
           addpath('Algorithms/srcGravitational');
           nEvaluationPatterns = size(x,1);
           nClasses = size(Y,2);
           forceVector = zeros(nEvaluationPatterns,nClasses);

           for n = 1:nEvaluationPatterns,
              for j = 1:nClasses,
                 forceVector(n,j) = force(x(n,:),j,X,Y,W,maxForce); 
              end
           end

           [maxValues predictedPatterns] = max(forceVector');
           predictedPatterns = predictedPatterns';

        end
    end
    
end

