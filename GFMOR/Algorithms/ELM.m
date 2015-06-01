classdef ELM < Algorithm
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
        
        classifier = 'nominal';
        activationFunction = 'sig';
        % Input Weights range 
        wMin = -1;
        wMax = 1;
        
        
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
    end
    
    properties(Access = private)
        model
    end

    
    methods
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: ELM (Public Constructor)
        % Description: It constructs an object of the class
        %               ELM and sets its characteristics.
        % Type: Void
        % Arguments: 
        %           classifier--> Type of ANN: classifier or regressor
        %           activationFunction--> 
        %           hiddenN--> Number of neurons in the hidden layer
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function obj = ELM(classifier,activationFunction)%, opt)
            obj.name = 'Extreme Learning Machine';
            if(nargin ~= 0)
                obj.classifier = classifier;
                obj.activationFunction = activationFunction;
            else
                %disp('Using default values for ELM.');
                obj.classifier = 'nominal';
                obj.activationFunction = 'sig';
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
            obj.parameters.hiddenN = 20;
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
        %           for the SVRPCDOC method and kernel parameters
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function [model_information] = runAlgorithm(obj,train, test, parameters)
            
                % <Mover a una función >
                train.uniqueTargets = unique([test.targets ;train.targets]);
                test.uniqueTargets = train.uniqueTargets;
                train.nOfClasses = max(train.uniqueTargets);
                test.nOfClasses = train.nOfClasses;                
                train.nOfPatterns = length(train.targets);
                test.nOfPatterns = length(test.targets);
                
                train.dim = size(train.patterns,2);
                test.dim = train.dim;
                % </Mover a una función >
                
                param.hiddenN = parameters(1);
                
                switch(obj.classifier)
                    case {'nominal'}
                        [train, test] = obj.labelToBinary(train,test);
                    case {'ordinal'}
                        [train, test] = obj.labelToOrelm(train,test);
                        train.uniqueTargetsOrelm = unique([test.targetsOrelm ;train.targetsOrelm],'rows');
                        test.uniqueTargetsOrelm = train.uniqueTargetsOrelm;
                
                    case {'ordinalRegressScaled'}
                        [train, test] = obj.scaleLabel(train,test, 0, 1);
                        
%                                 case {'LatentTriEqual'}
%                                     [trainSet, testSet] = obj.LatentTriEqual(trainSet,testSet);
%                                 case {'LatentTriProportional'}
%                                     [trainSet, testSet] = obj.LatentTriProportional(trainSet,testSet);
%                                 case {'no'} % default value
                        % Do nothing...
                end
                
                tic;
                model = obj.train( train, param);
                trainTime = toc;
                
                tic;
                [model_information.projectedTrain, model_information.predictedTrain] = obj.test( train,model );
                [model_information.projectedTest, model_information.predictedTest] = obj.test( test,model );
                testTime = toc;

                % Dummy values
                %model.thresholds = -1;

                model_information.trainTime = trainTime;
                model_information.testTime = testTime;
                model_information.model = model;

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: train (Public)
        % Description: This function train the model for
        %               the SVRPCDOC algorithm.
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
        function ELMmodel = train( obj,train, parameters)
            
            % TODO: UNHACK A NIVEL DE CROSSVALIDACIÓN
            if( strcmp(obj.activationFunction,'rbf') && parameters.hiddenN > train.nOfPatterns)
                    %disp(['User''s number of hidden neurons ' num2str(parameters.hiddenN) ... 
                     %   ' was too high and has been adjusted to the number of training patterns']);
               obj.parameters.hiddenN = train.nOfPatterns;
            else
               obj.parameters.hiddenN = parameters.hiddenN;
            end
            
            P = train.patterns';

             switch(obj.classifier)
                case {'nominal'}
                    T = train.targetsBinary;
                case {'ordinal'}
                    T = train.targetsOrelm;
                case {'ordinalRegress'}
                    T = train.targets;
                case {'ordinalRegressScaled'}
                    T = train.targetsScaled;
                case {'regression'}
                    T = train.targets;
                case {'regressionLatent'}
                    T = train.targetsLatent;
             end

            T = T';

            %%%%%%%%%%% Calculate weights & biases

            %------Perform log(P) calculation once for UP
            % The calculation is done here for including it into the validation time
            if strcmp(obj.activationFunction, 'up')
                P = log(P);
            end

            %%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons

            switch lower(obj.activationFunction)
                case {'sig','sigmoid'}
                    InputWeight=rand(obj.parameters.hiddenN,train.dim)*2-1;

                    BiasofHiddenNeurons=rand(obj.parameters.hiddenN,1);
                    tempH=InputWeight*P;
                    ind=ones(1,train.nOfPatterns);
                    BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                    tempH=tempH+BiasMatrix;
                case {'up'}
                    InputWeight = obj.wMin + (obj.wMax-obj.wMin).*rand(obj.parameters.hiddenN,train.dim);
                case {'rbf'}
                    P = P';
                    if (train.nOfPatterns>2000)
                        TY=pdist(P(randperm(2000),:));
                    else
                        TY=pdist(P);
                    end
                    a10=prctile(TY,20);
                    a90=prctile(TY,60);
                    MP=randperm(train.nOfPatterns);
                    W1=P(MP(1:obj.parameters.hiddenN),:);
                    W10=rand(1,obj.parameters.hiddenN)*(a90-a10)+a10;
                    W10 = W10';
                    InputWeight = [W1 W10];
                    clear TY;
                case {'krbf'}
                    P = P';
                    opts = statset('MaxIter',200);
                    [IDX, C, SUMD, D] = kmeans(P,obj.parameters.hiddenN,'Options',opts);
                    MC = squareform(pdist(C));
                    MCS = sort(MC);
                    MCS(1,:)=[];
                    radii = sqrt(MCS(1,:).*MCS(2,:));
                    InputWeight = [C radii'];

                    W1 = C;
                    W10 = radii;
                case {'grbf'}
                    MP = randperm(train.nOfPatterns);
                    InputWeight = P(:,MP(1:obj.parameters.hiddenN))';
            end


            %%%%%%%%%%% Calculate hidden neuron output matrix H
            switch lower(obj.activationFunction)
                case {'sig','sigmoid'}
                    %%%%%%%% Sigmoid 
                    H = 1 ./ (1 + exp(-tempH));
                case {'sin','sine'}
                    %%%%%%%% Sine
                    H = sin(tempH);    
                case {'hardlim'}
                    %%%%%%%% Hard Limit
                    H = double(hardlim(tempH));
                case {'tribas'}
                    %%%%%%%% Triangular basis function
                    H = tribas(tempH);
                case {'radbas'}
                    %%%%%%%% Radial basis function
                    H = radbas(tempH);
                    %%%%%%%% More activation functions can be added here
                case {'up'}
                    %PU_j(X) = productorio_{i=0}^n (x_i^{w_{ji}})
                    %P = log(P);

                    H = zeros(obj.parameters.hiddenN,train.nOfPatterns);
                    for i = 1 : train.nOfPatterns
                        for j = 1 : obj.parameters.hiddenN
                            temp = zeros(train.dim,1);
                            for n = 1: train.dim
                                temp(n) = InputWeight(j,n)*P(n,i);
                            end
                            H(j,i) =  sum(temp);
                        end
                    end
                    clear temp;
                case {'rbf','krbf'}
                    % TODO: Un hack
                    H = zeros(train.nOfPatterns, obj.parameters.hiddenN);
                    for j=1:obj.parameters.hiddenN
                        H(:,j)=gaussian_func(P,W1(j,:),W10(j,:));
                        %KM.valueinit(:,j)=gaussian_func(x,W1(j,:),W10(1,j));
                    end
                    H = H';        
                case {'grbf'}
                    % Compute Pairwise Euclidean distance
                    EuclideanDistanceArray = pdist(InputWeight);
                    EuclideanDistanceMatrix = squareform(EuclideanDistanceArray);
                    EuclideanDistanceSorted = sort(EuclideanDistanceMatrix);
                    % Larges distances and nearest distances
                    dF = EuclideanDistanceSorted(2,:);
                    %dN = (dF*0.05)/0.95;
                    dN = ones(size(dF)) * sqrt((0.001^2) * train.dim);
                    % Determine Tau and radii values
                    %taus = 4.0674 ./ (log(dF./dN));
                    taus = 5.6973 ./ (log(dF./dN));

                    taus = ones(1,obj.parameters.hiddenN)*2;
                    %radii = dF ./(-log(0.95)).^(1 ./taus);
                    radii = dF ./(-log(0.99)).^(1 ./taus);
                    % Obtain denominator
                    denominator = radii .^taus;    
                    denominator_extended = repmat(denominator,train.nOfPatterns,1)';
                    % Obtain Numerator
                    EuclideanDistance = pdist2(InputWeight,P','euclidean');
                    taus_extended = repmat(taus,train.nOfPatterns,1)';
                    numerator = EuclideanDistance.^taus_extended;
                    % Calculate Hidden Node outputs
                    H = exp(-(numerator./denominator_extended));
            end
            %COMENTADO clear P;

            clear tempH;                                        %   Release the temnormMinrary array for calculation of hidden neuron output matrix H


            %%%%%%%%%%% Calculate output weights OutputWeight (beta_i)

            % Hack Maria
            % ARREGLAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! tRASPONER
            % MATRIZ EN REGRESSION
            if strcmp(obj.classifier,'regression'),
                T = T';
            end
            OutputWeight=pinv(H') * T';                        % slower implementation
            % OutputWeight=inv(H * H') * H * T';                         % faster implementation


            ELMmodel.classifier = obj.classifier;
            ELMmodel.activationFunction = obj.activationFunction;
            ELMmodel.hiddenN = obj.parameters.hiddenN;
            ELMmodel.InputWeight = InputWeight;
            if strcmpi(obj.activationFunction, 'sig')
                ELMmodel.BiasofHiddenNeurons = BiasofHiddenNeurons;
            end

            if strcmp(obj.activationFunction, 'rbf') || strcmp(obj.activationFunction, 'krbf')
                ELMmodel.W1 = W1;
                ELMmodel.W10 = W10;
            end

            ELMmodel.OutputWeight = OutputWeight;

            ELMmodel.parameters = parameters;
            obj.model = ELMmodel;

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
        
        function [TY, TestPredictedY]= test(obj, test, model)

            if nargin < 3 
                model = obj.model;
            end
            
            TV.P = test.patterns';

            %------Perform log(P) calculation once for UP
            % The calculation is done here for including it into the validation time
            if strcmp(model.activationFunction, 'up')
                TV.P = log(TV.P);
            end

            %%%%%%%%%%% Calculate the output of testing input

            if strcmpi(model.activationFunction, 'sig')
                tempH_test=model.InputWeight*TV.P;
                %Movido abajo 
                %clear TV.P;             %   Release input of testing data             
                ind=ones(1,test.nOfPatterns);

                BiasMatrix=model.BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                tempH_test=tempH_test + BiasMatrix;
            end

            switch lower(model.activationFunction)
                case {'sig','sigmoid'}
                    %%%%%%%% Sigmoid 
                    H_test = 1 ./ (1 + exp(-tempH_test));
                case {'sin','sine'}
                    %%%%%%%% Sine
                    H_test = sin(tempH_test);
                case {'hardlim'}
                    %%%%%%%% Hard Limit
                    H_test = hardlim(tempH_test);
                case {'tribas'}
                    %%%%%%%% Triangular basis function
                    H_test = tribas(tempH_test);        
                case {'radbas'}
                    %%%%%%%% Radial basis function
                    H_test = radbas(tempH_test);        
                    %%%%%%%% More activation functions can be added here
                case {'up'}

                    %TV.P = log(TV.P);
                    H_test = zeros(model.hiddenN, test.nOfPatterns);

                    for i = 1 : test.nOfPatterns
                        for j = 1 : model.hiddenN
                            temp = zeros(test.dim,1);
                            for n = 1: test.dim
                                %temp(n) = TV.P(n,i)^InputWeight(j,n);
                                temp(n) = model.InputWeight(j,n)*TV.P(n,i);
                            end
                            %H_test(j,i) =  prod(temp);
                            H_test(j,i) =  sum(temp);
                        end
                    end

                    clear temp;
                case {'rbf','krbf'}
                    H_test = zeros(test.nOfPatterns,model.hiddenN);
                    TV.P = TV.P';

                    for j=1:model.hiddenN
                        H_test(:,j)=gaussian_func(TV.P,model.W1(j,:),model.W10(j,:));
                    end
                    H_test = H_test';

                case {'grbf'}
                    % Repmat denominator to Testing data
                    denominator_extended = repmat(denominator,nOfPatterns,1)';
                    % Recalculate Euclidean Distance
                    EuclideanDistanceTest = pdist2(InputWeight,TV.P','euclidean');
                    taus_extended = repmat(taus,nOfPatterns,1)';
                    numerator = EuclideanDistanceTest.^taus_extended;
                    % Calculate Hidden Node outputs
                    H_test = exp(-(numerator./denominator_extended));
            end

            clear TV.P;             %   Release input of testing data


            TY=(H_test' * model.OutputWeight)';                       %   TY: the actual output of the testing data
            
            clear H_test;

            switch (model.classifier)
                case {'nominal'}
                    [FOO, TestPredictedY] = max(TY);
                    %TY = -1*ones(size(TestPredictedY)); % Dummy value
                case {'ordinal'}
					% Otra alternativa
					% Q = size(model.OutputWeight,2);
                    % uniqueTargets = tril(2*ones(Q)) + -1*ones(Q);
                    % TestPredictedY = obj.orelmToLabel(TY',uniqueTargets);

                    TestPredictedY = obj.orelmToLabel(TY', test.uniqueTargetsOrelm);
                    %TY = -1*ones(size(TestPredictedY)); % Dummy value
                case {'ordinalRegress'}
                    TestPredictedY = obj.regressToLabel(TY', test.uniqueTargets);
                case {'ordinalRegressScaled'}
                    TestPredictedY = obj.regressToLabel(TY',test.uniqueTargets);
                case {'regression','regressionLatent'}
                    TestPredictedY = TY;
            end
            
            TestPredictedY = TestPredictedY';

        end

    end
    
    methods(Access = private)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: labelToBinary (Private)
        % Description: 
        % Type: 
        % Arguments: 
        %           trainSet--> Array of training patterns
        %           testSet--> Array of testing patterns
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [trainSet, testSet] = labelToBinary(obj,trainSet,testSet)
            % Adapt labels to Matlab's newff / ELM / etc. 

            % NOTE: This option based on full + ind2vec is not valid since 
            % it have problems if a datasets do not have patterns of a
            % given class
            %DataSet.TrainTT = full(ind2vec(DataSet.TrainT'));
            %DataSet.TestTT = full(ind2vec(DataSet.TestT'));

            trainN = size(trainSet.targets,1);
            TT = zeros(trainN,trainSet.nOfClasses);
            for ii=1:trainN
                TT(ii,trainSet.targets(ii,1)) = 1;
            end

            trainSet.targetsBinary = TT;

            testN = size(testSet.targets,1);
            TT = zeros(testN,testSet.nOfClasses);
            for ii=1:testN
                TT(ii,testSet.targets(ii,1)) = 1;
            end

            testSet.targetsBinary = TT;

            clear trainN testN TT;
        end

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: scaleLabel (Private)
        % Description: 
        % Type: 
        % Arguments: 
        %           trainSet--> Array of training patterns
        %           testSet--> Array of testing patterns
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [trainSet, testSet] = scaleLabel(obj,trainSet,testSet, min, max)

            trainSet.targetsScaled = scale(trainSet.targets, min, max);
            testSet.targetsScaled = scale(testSet.targets, min, max);
            
        end
        
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
        
        function [finalOutput] = orelmToLabel(obj,predictions,uniqueNewTargets)

            finalOutput = zeros(1,size(predictions,1));
            distancias = zeros(1,size(predictions,2));

            for i=1:size(predictions,1),
                for j=1:size(predictions,2),
                    distancias(j) = pdist([predictions(i,:);uniqueNewTargets(j,:)]);
                end
                [FOO,finalOutput(i)] = min(distancias);
            end

        end
        


	function [trainSet, testSet] = labelToOrelm(obj,trainSet,testSet)
            %uniqueTargets = unique([trainSet.targets; testSet.targets]);

            %   newTargets = zeros(trainSet.nOfPatterns,trainSet.nOfClasses);
            trainSet.targetsOrelm = ones(trainSet.nOfPatterns,trainSet.nOfClasses);
            testSet.targetsOrelm = ones(testSet.nOfPatterns,trainSet.nOfClasses);
            
            for i=1:trainSet.nOfClasses,
                trainSet.targetsOrelm(trainSet.targets<trainSet.uniqueTargets(i),i) = -1;
                testSet.targetsOrelm(testSet.targets<trainSet.uniqueTargets(i),i) = -1;
            end
           
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: regressToLabel (Private)
        % Description: 
        % Type: 
        % Arguments: 
        %           trainSet--> Array of training patterns
        %           testSet--> Array of testing patterns
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [finalOutput] = regressToLabel(obj,predictions,uniqueNewTargets)

            finalOutput = zeros(1,size(predictions,1));
            distancias = zeros(1,numel(uniqueNewTargets));

            for i=1:size(predictions,1),
                for j=1:numel(distancias),
                    distancias(j) = pdist([predictions(i,:);uniqueNewTargets(j,:)]);
                end
                [FOO,finalOutput(i)] = min(distancias);
            end

        end
        
    end
    
end

