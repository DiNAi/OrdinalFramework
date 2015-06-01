classdef oNN < Algorithm
    %ELM Extreme Learning Machine
    %   This class derives from the Algorithm Class and implements the
    %   ELM method with some alternatives
    %   Characteristics: 
    %               -TODO
    %               -Parameters: 
    %                       -hiddenNC: number of networks in the hidden
    %                       layer
    %
    
    properties
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Variable: parameters (Public)
        % Type: Struct
        % Description: This variable keeps the values for 
        %               the C penalty coefficient, the 
        %               kernel parameters and epsilon
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        parameters
        name_parameters = {'hiddenN'}
    end

    
    methods
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: oNN (Public Constructor)
        % Description: It constructs an object of the class
        %               oNN and sets its characteristics.
        % Type: Void
        % Arguments: 
        %           classifier--> Type of ANN: classifier or regressor
        %           activationFunction--> 
        %           hiddenN--> Number of neurons in the hidden layer
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function obj = oNN()%, opt)
            obj.name = 'Ordinal Neural Networks (The Replication Method)';
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
            obj.parameters.hiddenN = {5,10,25,40};
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
        %           for the oNN method and kernel parameters
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function [model_information] = runAlgorithm(obj,training, test, parameters)
            
                param.hiddenN = parameters(1);
                
                p               = size(training.patterns, 2);
                features        = training.patterns;
                size(features)
                classes         = training.targets;

                K               = size(unique(classes),1);
                
                s = K;      %ceil(K/2); %we fix K at the highest value but it can be less
                H = 1;


                net = network;
                net.numInputs = 2;
                net.numLayers = 2;
                net.biasConnect = [1; 1];
                net.inputConnect = [1 0; 0 1];
                net.layerConnect = [0 0; 1 0];

                net.outputConnect = [0 1];

                net.inputs{1}.range = minmax (features');
                net.inputs{2}.range = ones(K-2, 2)*diag([0 H]);

                net.outputConnect = [0 1];

                net.layers{1}.size = param.hiddenN;
                net.layers{1}.transferFcn = 'logsig';
                net.layers{1}.initFcn = 'initnw';

                net.layers{2}.size = 1;
                net.layers{2}.transferFcn = 'tansig'; 
                net.layers{2}.initFcn = 'initnw';

                net.initFcn = 'initlay';
                net.performFcn = 'mse';
                net.trainFcn = 'trainlm';

                c1 = clock;
                [model] = obj.train(net,param, features,classes,K,H,s,p);
                predClasses = zeros(size(training.targets),1);
                for i=1:size(training.targets)
                    feature    = training.patterns(i,1:p);
                    [NewFeatures, ~] = xreplicateData(feature, [], K, H, K-1);
                    P = {NewFeatures(:,1:p)'; NewFeatures(:,p+1:p+K-2)'};
                    pred = sim(model.net, P);
                    pred = pred{1}';
                    pred = sign(pred);
                    pred = 1+size(find(pred==1),1);
                    predClasses(i) = pred;
                end
                
                model_information.predictedTrain = predClasses;
                model_information.projectedTrain = predClasses;
                


                c2 = clock;
                model_information.trainTime = etime(c2, c1);
                
                c1 = clock;
                predClasses = obj.test(model.net,test.patterns,test.targets,K,H,p);
                c2 = clock;
                model_information.testTime = etime(c2,c1);
                model_information.predictedTest = predClasses;
                model_information.projectedTest = predClasses;
                model_information.model = model;


           
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: train (Public)
        % Description: This function train the model for
        %               the oNN algorithm.
        % Type: [Structure]
        % Arguments: 
        %           train.patterns --> Trainning data for 
        %                              fitting the model
        %           testTargets --> Training targets
        %           parameters --> 
        % ,  wMin, wMax
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %[InputWeight,BiasofHiddenNeurons,OutputWeight,Y,TrainingTime]
        function [model,P] = train( obj,net,parameters, features,classes,K,H,s,p)
            
                net = init(net);
                net.trainParam.epochs = 2000;
                net.trainParam.show = 4000;
                net.trainParam.goal = 1e-10;

                [NewFeatures, NewClasses] = xreplicateData(features, classes, K, H, s);
                P = {NewFeatures(:,1:p)'; NewFeatures(:,p+1:p+K-2)'};
                net = train(net, P, {NewClasses(:)'});
                model.net = net;
                model.hiddenN = obj.parameters.hiddenN;
                model.algorithm = 'oNN';
                model.parameters = parameters;

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: test (Public)
        % Description: This function test a model given
        %               a set of test patterns.
        % Type: [Array, Array]
        % Arguments: 
        %           test.patterns --> Testing data
        %           projection --> Projection previously 
        %                       calculated fitting the model
        %           thresholds --> Thresholds previously 
        %                       calculated fitting the model
        %           train.patterns --> Trainning data (needed
        %                              for the gram matrix)
        %           kernelParam --> kernel parameter for SVRPCDOC
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [predClasses]= test(obj,net, features,targets,K,H,p)
                predClasses = zeros(size(targets),1);
                for i=1:size(targets)
                    feature    = features(i,1:p);
                    [NewFeatures, NewClasses] = xreplicateData(feature, [], K, H, K-1);
                    P = {NewFeatures(:,1:p)'; NewFeatures(:,p+1:p+K-2)'};
                    pred = sim(net, P);
                    pred = pred{1}';
                    pred = sign(pred);
                    pred = 1+size(find(pred==1),1);
                    predClasses(i) = pred;
                end
                

        end


    end
        
end

