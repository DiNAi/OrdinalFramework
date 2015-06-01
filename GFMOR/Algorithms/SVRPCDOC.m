classdef SVRPCDOC < Algorithm
    %SVRPCDOC Kernel Discriminant Learning for Ordinal Regression
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
        % Variable: optimizationMethod (Public)
        % Type: String
        % Description: It specifies the method used for 
        %              optimizing the discriminant funcion
        %              of the model. It can be quadprog, 
        %               qp, or cvx.
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
%         optimizationMethod = 'quadprog'
        
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
        name_parameters = {'C','k','e'}
    end
    
    properties(Access = private)
        model
    end

    
    methods
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: SVRPCDOC (Public Constructor)
        % Description: It constructs an object of the class
        %               SVRPCDOC and sets its characteristics.
        % Type: Void
        % Arguments: 
        %           kernel--> Type of Kernel function
        %           opt--> Type of optimization used in the method.
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function obj = SVRPCDOC(kernel)%, opt)
            obj.name = 'Pairwise Class Distances Ordinal Classifier';
            if(nargin ~= 0)
                 obj.kernelType = kernel;
            else
                obj.kernelType = 'rbf';
            end
%             if(nargin > 1)
%                 obj.optimizationMethod = opt;
%             else
%                 obj.optimizationMethod = 'quadprog';
%             end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: set.optimizationMethod (Public)
        % Description: It verifies if the value for the 
        %               variable optimizationMethod 
        %                   is correct.
        % Type: Void
        % Arguments: 
        %           value--> Value for the variable optimizationMethod.
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%         function obj = set.optimizationMethod(obj, value)
%             if ~(strcmpi(value,'quadprog') || strcmpi(value,'qp') || strcmpi(value,'cvx'))
%                    error('Invalid value for optimizer');
%             else
%                    obj.optimizationMethod = value;
%             end 
%         end

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
            obj.parameters.C = 10.^(3:-1:-3);
            obj.parameters.k = 10.^(3:-1:-3);
            obj.parameters.e = 10.^(-3:1:0);
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
            	%addpath libsvm-weights-3.12/matlab
                addpath Algorithms/libsvm-weights-3.12/matlab
                
                param.C = parameters(1);
                param.k = parameters(2);
                param.e = parameters(3); 
                
                train.nOfClasses = length(unique(train.targets));
                test.nOfClasses = train.nOfClasses;
                
                tic;
                obj.model = obj.train( train, param);  
                [p, train.predicted] = obj.test( train );
                trainTime = toc;
                
                tic;
                [p2, test.predicted] = obj.test( test );
                testTime = toc;

                model_information.predictedTrain = train.predicted;
                model_information.predictedTest = test.predicted;
                model_information.projection = obj.model.thresholds;
                model_information.trainTime = trainTime;
                model_information.testTime = testTime;
                model_information.parameters = parameters;
                model_information.projectedTest = p2;
                model_information.projectedTrain = p;
                
                obj.model.algorithm = 'SVRPCDOC';
                obj.model.parameters = param;

                model_information.model = obj.model;
                
                
                % trainZ = obj.pcdprojection(train.patterns,train.targets,train.nOfClasses);

                rmpath Algorithms/libsvm-weights-3.12/matlab

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: train (Public)
        % Description: This function train the model for
        %               the SVRPCDOC algorithm.
        % Type: [Structure]
        % Arguments: 
        %           trainPatterns --> Trainning data for 
        %                              fitting the model
        %           testTargets --> Training targets
        %           parameters --> Penalty coefficient C 
        %           for the SVRPCDOC method and kernel parameters
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function pcdocModel = train( obj,trainSet, parameters)
                %[dim,numTrain] = size(trainPatterns);

                if(nargin < 1)
                    error('Patterns and targets are needed.\n');
                end
                               
                pcdocModel.svrhyperparam.C = parameters.C;
                pcdocModel.svrhyperparam.k = parameters.k;
                pcdocModel.svrhyperparam.e = parameters.e;
                    
                % PCD projection
                % width 
                width = 1/trainSet.nOfClasses;
                % maximum Z value
                maxZ = 1;
                % Thressholds used for classification purposes when predicting
                % new values of Z for unseen data. 
                pcdocModel.thresholds = width:width:maxZ;

                nc = trainSet.nOfClasses;
                % Calculate the projection
                trainZ = obj.pcdprojection(trainSet.patterns,trainSet.targets,nc);

                switch(obj.kernelType)
                    case {'lin'}
                        kernelTypeNum = 0;
                    case {'rbf'}
                        kernelTypeNum = 2;
                end
                % e-SVR training
                svrParameters = ...
                    ['-s 3 -t ' num2str(kernelTypeNum) ' -c '  num2str(pcdocModel.svrhyperparam.C) ...
                    ' -p ' num2str(pcdocModel.svrhyperparam.e)  num2str(pcdocModel.svrhyperparam.k) ' -q'];

                weights = ones(size(trainSet.targets));

                pcdocModel.svrmodel = svmtrain(weights, trainZ, trainSet.patterns, svrParameters);
                pcdocModel.svrParameters =svrParameters;
                %pcdocModel.svrmodel = svmtrain(trainZ, trainPatterns, svrParameters);
                
                obj.model = pcdocModel;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: test (Public)
        % Description: This function test a model given
        %               a set of test patterns.
        % Type: [Array, Array]
        % Arguments: 
        %           testPatterns --> Testing data
        %           projection --> Projection previously 
        %                       calculated fitting the model
        %           thresholds --> Thresholds previously 
        %                       calculated fitting the model
        %           trainPatterns --> Trainning data (needed
        %                              for the gram matrix)
        %           kernelParam --> kernel parameter for SVRPCDOC
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [TestPredictedZ, TestPredictedY]= test(obj, test)
            
                    DummyTestT = rand(size(test.patterns,1),1);

                    % svmpredict needs TestT
                    TestPredictedZ = ...
                                    svmpredict(DummyTestT, test.patterns, obj.model.svrmodel);

                    TestPredictedY = obj.PCDOC_classify(TestPredictedZ, obj.model.thresholds, test.nOfClasses);

        end      

    end
    
    methods(Access = private)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % [Z,ZC] = pcdprojection(X,Y,Q)
        %
        % DESCRIPTION: 
        %   This function implements the Pairwise Class Distances projection.
        %   The distances calculation code is not optimized, however, it is more
        %   understable in the way it is writed. 
        % INPUT: 
        %   - X: patterns attributes
        %   - Y: patterns class labels
        %   - Q: number of classes
        % OUTPUT: 
        %   - Z: the PCD projection (latent representation). 
        %   - ZC: the PCD projection separated for each class
        %   - Dmin,DminIdx,Ztemp,NC: are returning for teaching purposes, such as
        %   for the projection analysis. 
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function [Z,ZC,Dmin,DminIdx,NC] = pcdprojection(obj,X,Y,Q)

            % Number of pattern per class
            NC = obj.orderedClassDistribution(Y,Q);

            % width 
            width = 1/Q;

            % center for each class
            center = zeros(Q,1);

            % Get distances between adjacent classes using pdist() with the default
            % distance (Euclidean)
            % Dmin is \kappa in the Equations
            Dmin = cell(Q,Q);
            DminIdx = cell(Q,Q);

            for i = 1:Q
                 j = i + 1;
                 if j<=Q
                    % i+1 class
                    % Class 1
                    Ni=size(X(Y==i,:),1);
                    Nj=size(X(Y==j,:),1);
                    Xsub = [X(Y==i,:); X(Y==j,:)];
                    XsubDist = squareform(pdist(Xsub));

                    % 'Supress' distances between elements of the same class for
                    % ignoring them when calculating the minimum distances
                    XsubDist(1:Ni,1:Ni) = inf;
                    XsubDist(Ni+1:end,Ni+1:end) = inf;

                    % Get the shortest distances
                    [XdubDistMin XdubDistMinIdx] = min(XsubDist,[],1);

                    % Save the pairwise distances for classes i and j
                    Dmin{i,j} = XdubDistMin';
                    DminIdx{i,j} = XdubDistMinIdx';
                end

            end


            Ztemp=cell(Q,1);

            for i=1:Q
                Ztemp{i,1} = ones(NC(i),1)*inf;
            end

            % class 1,2
            i = 1;
            center(i) = 0;

            DminJr = Dmin{i,i+1};
            DminJr = DminJr(1:NC(i),:);

            W = DminJr / max(DminJr);
            Ztemp{i,1} = center(i) + width*(1-W)*(1/2);

            clear DminJr W;

            for i=2:Q-1   
                % Get distances only for the element on i class

                % Look to the class on the right
                DminJr = Dmin{i,i+1};
                DminJr = DminJr(1:NC(i),:);

                % Look to the class on the left
                DminJl = Dmin{i-1,i};
                DminJl = DminJl(NC(i-1)+1:end,:);

                W = (DminJl+DminJr)/(max(DminJl+DminJr));

                center(i) = i*width-width/2;
                ZWtemp = zeros(NC(i),1);

                for n = 1:size(DminJl)
                     if DminJl(n)<=DminJr(n) 
                         ZWtemp(n,1) = center(i) - ((width/2)*(1-W(n,1)));
                     else
                         ZWtemp(n,1) = center(i) + ((width/2)*(1-W(n,1)));
                     end
                end
                Ztemp{i,1} = ZWtemp;
                clear ZWtemp;
            end

            %class Q-1,Q
            i=Q;
            center(Q) = 1;
            DminJl = Dmin{Q-1,Q};
            DminJl = DminJl(NC(Q-1)+1:end,:);

            W = DminJl / max(DminJl);
            Ztemp{i,1} = center(i) - width*(1-W)*(1/2);

            clear DminJr;

            % Join ZC for each class in Z
            ZC = cell(Q,1);
            poffset = 0;
            for i=1:Q
                temp = Ztemp{i,1};

                if i==1
                    Z = temp;
                else
                    Z = [Z;temp];
                end

                ZC{i,1} = temp;

                poffset=poffset+NC(i);
            end

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Y = PCDOC_classify(Z, pcdocModel.T, Q)
        %
        % DESCRIPTION: 
        %   This function uses Z latent representation and thressholds for classifying 
        %   patterns according to they latent representation.
        % INPUT: 
        %   - Z: patterns latent representation
        %   - T: thressholds
        %   - Q: number of classes
        % OUTPUT: 
        %   - Y: predicted patterns labels
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function Y = PCDOC_classify(obj,Z, T, Q)

            Y = zeros(size(Z));

            for ii=1:Q-1

                if (ii == 1)
                    indx = Z<=T(ii);
                else
                    indx = and(Z>T(ii-1), Z<=T(ii));
                end

                Y(indx) = ii;
            end

            % last class
            indx = Z>T(Q-1);
            Y(indx) = Q;

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: orderedClassDistribution (Private)
        % Description: This function returns the number of pattenrs per 
        %       class. 
        % Type: [Array]
        % Arguments: 
        %           T --> Tags
        %           Q --> Number of classes
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function Pq = orderedClassDistribution(obj,T,Q)
            Pq = zeros(Q,1);

            for ii=1:Q
                Pq(ii,1) = sum(T==ii);
            end
        end
        
    end
    
end

