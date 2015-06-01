classdef ELM_LVOR < Algorithm
    %ELM_LVOR ELM with Latent Variable for Ordinal Regression
    %   This class derives from the Algorithm Class and implements the
    %   KLDOR method. 
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
        %               the C penalty coefficient, the 
        %               kernel parameters and epsilon
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        parameters
        activationFunction = 'sig';
        lambda = 1;
        latentModel = 'triEqual';

    end
    
    properties(Access = private)
        model;
        modelELMr;
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
        
        function obj = ELM_LVOR(activationFunction, latentModel)%, opt)
            obj.name = 'ELM with Latent Variable for Ordinal Regression';
            
            if(nargin ~= 0)
                obj.activationFunction = activationFunction;
                obj.latentModel = latentModel;
            else
                obj.activationFunction = 'sig';
                obj.latentModel = 'triEqual';
            end
            
            obj.modelELMr = ELM('regressionLatent', obj.activationFunction);

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
                                
                tic;
                obj.model = obj.train(train, parameters);
                [projectedTrain, train.predicted] = obj.test( train );
                trainTime = toc;
                
                tic;
                %[p2, test.predicted] = obj.test( test );
                [projectedTest, test.predicted] = obj.test( test );
                testTime = toc;

                % Dummy values
                obj.model.thresholds = -1;
      
                model_information.predictedTrain = train.predicted;
                model_information.predictedTest = test.predicted;
                model_information.projection = obj.model.thresholds;
                model_information.trainTime = trainTime;
                model_information.testTime = testTime;
                model_information.parameters = parameters;
                model_information.projectedTest = projectedTest;
                model_information.projectedTrain = projectedTrain;
                model_information.thresholds = obj.model.thresholds;
                model_information.model = obj.model;

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
        function ELM_LVORmodel = train( obj,train, parameters)
            
            ELMparams.hiddenN = parameters(1);
           
            % Add latent variable 
            switch(obj.latentModel)
                case {'triEqual'}
                    ELM_LVORmodel.PDF = ELM_LVOR.adjusttripdfsTriEqual(train.nOfClasses, obj.lambda);
                    [train.targetsLatent, foo] = ELM_LVOR.addlatentvariableTriEqual(train.targets, train.nOfClasses, ELM_LVORmodel.PDF);
                case {'triProportional'}
                    % Calculate a priori per-class probability
                    PQ = ELM_LVOR.aPrioriProbabilities(train.targets,train.nOfClasses);
                    ELM_LVORmodel.PDF = ELM_LVOR.adjusttripdfsTriProportional(PQ, obj.lambda);
                    [train.targetsLatent, foo] = ELM_LVOR.addlatentvariableTriProportional(train.targets, train.nOfClasses, ELM_LVORmodel.PDF);
            end
            clear foo;

            % Train ELM regressor and save the model
            ELM_LVORmodel.ELMModel = obj.modelELMr.train( train, ELMparams);
            
            obj.model = ELM_LVORmodel;
            
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
        
        function [projectedTest, TestPredictedLabel]= test(obj, testSet)
            
            % Predict the regression value
            [projectedTest] = obj.modelELMr.test( testSet );
            
            % Map regression predicted values to labels according to
            % thressholds
            switch(obj.latentModel)
                case {'triEqual'}
                    TestPredictedLabel = ELM_LVOR.zToClassTri(projectedTest, obj.model.PDF, testSet.nOfClasses);
                case {'triProportional'}
                    TestPredictedLabel = ELM_LVOR.zToClassTri(projectedTest, obj.model.PDF, testSet.nOfClasses);
            end
            
            TestPredictedLabel = TestPredictedLabel';

        end      

    end
    
    methods(Static, Access = private)
        
        function [Z,ZC] = addlatentvariableTriEqual(T, Q, PDF)
            % Latent variable
            Z = zeros(size(T));
            ZC = cell(Q,1);

            %PDF = adjusttripdfsV3(k,overlap);

            for ii=1:Q
                ind = (T==ii);
                ZC{ii,1} = ELM_LVOR.trirnd(PDF{ii,1}.a,PDF{ii,1}.c,PDF{ii,1}.b,size(Z(ind),1));
                ZC{ii,1} = ZC{ii,1}';
                Z(ind) = ZC{ii,1};
            end

        end
        
        function PDF = adjusttripdfsTriEqual(Q, lambda)

            % Adjusted probability densities functions
            PDF = cell(Q,1);

            % Calculate 'c' for all the pdfs
            PDF{1,1}.c = 0;

            Max = 1;

            % Adjust the overlap 'ov' considering the number of classes
            % and the 'Max' value
            lambda = lambda * (Max/(Q-1));

            % Calculate 'a', 'c' and 'b'
            PDF{1,1}.a = 0;
            PDF{1,1}.b = (1/Q) + lambda;
            PDF{1,1}.c = (PDF{1,1}.b - PDF{1,1}.a) / 2;

            for jj=2:Q-1
                PDF{jj,1}.a = (jj-1)*(1/Q) - lambda;
                PDF{jj,1}.b = (jj)*(1/Q) + lambda;
                PDF{jj,1}.c = PDF{jj,1}.a + (PDF{jj,1}.b - PDF{jj,1}.a) / 2;
            end

            PDF{Q,1}.a = (Q-1)*(1/Q) - lambda;
            PDF{Q,1}.b = Max;
            PDF{Q,1}.c = PDF{Q,1}.a + (PDF{Q,1}.b - PDF{Q,1}.a) / 2;

            % Intersection points
            % This  V2 method should be suitable for V3
            for jj=1:Q-1    
                [PDF{jj,1}.x PDF{jj,1}.y] = ELM_LVOR.intersection(PDF{jj,1},PDF{jj+1,1});
            end

        end
        
        
        function Y = zToClassTri(Z, PDF, k)

            Y = zeros(size(Z));

            % calculate thresholds and assign labels
            %threshold = zeros(k-1,1);

            % Intermediate classes thresholds
            for ii=1:k-1
                %threshold(ii,1) = PDF{ii}.x;

                if (ii == 1)
                    indx = Z<=PDF{ii}.x;
                else
                    indx = and(Z>PDF{ii-1}.x, Z<=PDF{ii}.x);
                end

                Y(indx) = ii;
            end

            %last class
            indx = Z>PDF{k-1}.x;
            Y(indx) = k;

        end
        
        
        %%% Latent variable with proportional widths
        function [Z,ZC] = addlatentvariableTriProportional(T, Q, PDF)
            % Latent variable
            Z = zeros(size(T));
            ZC = cell(Q,1);

            % Calculate a priori per-class probability
            %PQ = aPrioriProbabilities(T,Q);

            %PDF = adjusttripdfsTriProportional(PQ,overlap);

            for ii=1:Q
                ind = (T==ii);
                ZC{ii,1} = trirnd(PDF{ii,1}.a,PDF{ii,1}.c,PDF{ii,1}.b,size(Z(ind),1));
                ZC{ii,1} = ZC{ii,1}';
                Z(ind) = ZC{ii,1};
            end

        end
        
        function PDF = adjusttripdfsTriProportional(PQ, lambda)

            Q = size(PQ,1);
    
            % Adjusted probability densities functions
            PDF = cell(Q,1);

            % Calculate 'c' for all the pdfs
            PDF{1,1}.c = 0;

            Max = 1;

            % Adjust the overlap 'ov' considering the width and class position
            ov = zeros(Q,1);

            ov(1,1) = PQ(1,1)*lambda;

            for jj=2:Q-1 
                ov(jj,1) = PQ(jj,1)*(lambda/2);
            end

            ov(Q,1) = PQ(Q,1)*lambda;

            % Calculate the intersection points based on the PQs
            PDF{1,1}.x = PQ(1,1);
            PDF{1,1}.y = 0;

            for jj=2:Q-1
                PDF{jj,1}.x = PDF{jj-1,1}.x + PQ(jj,1);
                PDF{jj,1}.y = 0;
            end

            PDF{Q,1}.x = Max;
            PDF{Q,1}.y = 0;

            % Calculate 'a', 'c' and 'b'
            PDF{1,1}.a = 0;
            PDF{1,1}.b = PDF{1,1}.x + ov(1,1);
            PDF{1,1}.c = (PDF{1,1}.b - PDF{1,1}.a) / 2;

            for jj=2:Q-1
                PDF{jj,1}.a = PDF{jj-1,1}.x - ov(jj,1);
                PDF{jj,1}.b = PDF{jj,1}.x + ov(jj,1);
                PDF{jj,1}.c = PDF{jj,1}.a + (PDF{jj,1}.b - PDF{jj,1}.a) / 2;
            end

            PDF{Q,1}.a = PDF{Q-1,1}.x - ov(Q,1);
            PDF{Q,1}.b = Max;
            PDF{Q,1}.c = PDF{Q,1}.a + (PDF{Q,1}.b - PDF{Q,1}.a) / 2;

            % Intersection points
            % This  V2 method should be suitable for V3
            %     for jj=1:k-1    
            %         [PDF{jj,1}.x PDF{jj,1}.y] = intersectionV2(PDF{jj,1},PDF{jj+1,1});
            %     end

        end
        
        
        function PQ = aPrioriProbabilities(T,Q)
            PQ = zeros(Q,1);
            N = size(T,1);

            for ii=1:Q
                PQ(ii,1) = sum(T==ii) / N;
            end
        end

        
        % Algorithm http://paulbourke.net/geometry/lineline2d/
        function [x,y] = intersection( PDF1, PDF2)

            x1 = PDF1.c;
            x2 = PDF1.b;
            y1 = 2/(x2-PDF1.a);
            y2 = 0;
            x3 = PDF2.a;
            y3 = 0;
            x4 = PDF2.c;
            y4 = 2/(PDF2.b-PDF2.a);

            ua = ( (x4-x3)*(y1-y3) - (y4-y3)*(x1-x3) ) / ...
                ( (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1) );
            %ub = ( (x2-x1)*(y1-y3) - (y2-y1)*(x1-x3) ) / ...
            %    ( (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1) );

            x = x1 + ua*(x2-x1);
            y = y1 + ua*(y2-y1);

        end

        
        %Script by Dr.Mongkut Piantanakulchai
        %To simulate the triangular distribution
        %Return a vector of random variable
        %The range of the value is between (a,b)
        %The mode is c (most probable value)
        %n is to tatal number of values generated
        %Example of using
        %X = trirnd(1,5,10,100000);
        % this will generate 100000 random numbers between 1 and 10 (where most probable
        % value is 5)
        % To visualize the result use the command
        % hist(X,50); 

        function X = trirnd(a,c,b,n)
            X=zeros(n,1);
            for i=1:n
                %Assume a<X<c
                z=rand;
                if sqrt(z*(b-a)*(c-a))+a<c
                    X(i)=sqrt(z*(b-a)*(c-a))+a;
                else
                    X(i)=b-sqrt((1-z)*(b-a)*(b-c));
                end
            end %for
            %hist(X,50); Remove this comment % to look at histogram of X
        end %function
        
        
    end
    
end

