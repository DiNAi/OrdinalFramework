classdef SVMRank < Algorithm
    % SVOR Support Vector for Ordinal Regression (Implicit constraints)
    %   This class derives from the Algorithm Class and implements the
    %   SVORIM method.
    %   Characteristics:
    %               -Kernel functions: Yes
    %               -Ordinal: Yes
    
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
    end
    
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: SVORIM (Public Constructor)
        % Description: It constructs an object of the class
        %               SVORIM and sets its characteristics.
        % Type: Void
        % Arguments:
        %           kernel--> Type of Kernel function
        %           opt--> Type of optimization used in the method.
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function obj = SVORIM(kernel)
            obj.name = 'Support Vector for Ordinal Regression (Implicit constraints)';
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
            obj.parameters.C =  10.^(-3:1:3);
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
        %           for the SVORIM method and kernel parameters
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [model_information] = runAlgorithm(obj,train, test, parameters)
            
            param.C = parameters(1);
            param.k = parameters(2);
            data_information.folder = 'tmp';
            data_information.dataset = 'blabla';
            data_information.holdout = 1;
            tic;
            train_file = [data_information.folder '/train-' data_information.dataset '-' num2str(data_information.holdout)];
                    test_file = [data_information.folder '/test-' data_information.dataset '-' num2str(data_information.holdout)];
                    model_file = [data_information.folder '/model-' data_information.dataset '-' num2str(data_information.holdout)];
                    output_file = [data_information.folder '/output-' data_information.dataset '-' num2str(data_information.holdout)];
                    guess_file = [data_information.folder '/guess-' data_information.dataset '-' num2str(data_information.holdout)];

            libsvmwrite(train_file, train.targets, sparse(train.patterns));
                    obj.train( train_file, model_file, param);

                    [train.projected, train.predicted] = obj.test( train_file, model_file, output_file, guess_file);
                    trainTime = toc;

                    tic;
                    libsvmwrite(test_file, test.targets, sparse(test.patterns));
                    [test.projected, test.predicted] = obj.test( test_file, model_file,output_file, guess_file);            
                    testTime = toc;

                    b = lee_modelo(model_file, numel(unique([train.targets; test.targets])));
                    dataSetStatistics.thresholds = b(2:end);
                     
                    system(['rm ' train_file]);
                    system(['rm ' test_file]);
                    system(['rm ' output_file]);
                    system(['rm ' model_file]);
                    system(['rm ' guess_file]);
            model_information.predictedTrain = train.predicted;
            model_information.predictedTest = test.predicted;
            
            model_information.projection = b;          
            model_information.trainTime = trainTime;
            model_information.testTime = testTime;
            model_information.parameters = parameters;
            model_information.projectedTest = test.projected';
            model_information.projectedTrain = train.projected';
            model_information.thresholds = b;
            
        end
        
        
        
      function train(obj,train_file, model_file , parameters)
            options = ['-s 5 -t 2 -c ' num2str(parameters.C) ' -g ' num2str(parameters.k) ];
            execute_train = ['./libsvm-2.81/svm-train ' options ' ' train_file ' ' model_file ];
            system(execute_train);
        end
        
        function [projected, testTargets]= test(obj, test_file, model_file, output_file, guess_file)
                execute_test = ['./libsvm-2.81/svm-predict ' test_file ' ' model_file ' ' output_file ' ' guess_file];
                system(execute_test);
                projected = load(guess_file);
                testTargets = load(output_file);
        end  
        
        
    end
   
        
    end

