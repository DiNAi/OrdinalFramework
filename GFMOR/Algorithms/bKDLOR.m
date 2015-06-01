classdef bKDLOR < Algorithm
    %KDLOR Kernel Discriminant Learning for Ordinal Regression
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
        
        optimizationMethod = 'quadprog'
        
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

	base_algorithm = KDLOR
    end
    
    methods
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: KDLOR (Public Constructor)
        % Description: It constructs an object of the class
        %               KDLOR and sets its characteristics.
        % Type: Void
        % Arguments: 
        %           kernel--> Type of Kernel function
        %           opt--> Type of optimization used in the method.
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function obj = KDLOR(kernel, opt)
            obj.name = 'Product Ensemble Discriminant Analysis';
            obj.ordinal = 1;
            obj.numParameters = 2;
            obj.determinist = 1;
            if(nargin ~= 0)
                 obj.kernelType = kernel;
            else
                obj.kernelType = 'rbf';
            end
            if(nargin > 1)
                obj.optimizationMethod = opt;
            else
                obj.optimizationMethod = 'quadprog';
            end
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

        function obj = set.optimizationMethod(obj, value)
            if ~(strcmpi(value,'quadprog') || strcmpi(value,'qp') || strcmpi(value,'cvx'))
                   error('Invalid value for optimizer');
            else
                   obj.optimizationMethod = value;
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
            obj.parameters.C = [0.1,1,10,100];
            obj.parameters.k = [0.001,0.01,0.1,1,10,100,1000];
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
        %           for the KDLOR method and kernel parameters
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function [model_information] = runAlgorithm(obj,train, test, parameters)
                
                param.C = parameters(1);
                param.k = parameters(2);
                
                if strcmp(obj.kernelType, 'sigmoid'),
                    param.k = [parameters(2),parameters(3)];
                    if numel(parameters)>3,
                        param.u = parameters(4);
                    else
                        param.u = 0.01;
                    end
                else
                    if numel(parameters)>2,
                        param.u = parameters(3);
                    else
                        param.u = 0.01;
                    end
               
                end
                
                tic;

                
                classes = unique(train.targets);
                nOfClasses = numel(classes);
               n = zeros(1,nOfClasses);
                for i=1:nOfClasses,
                    n(i) = sum(train.targets == i);
                end
               
              prob_pertenencia_claseTrain = ones(nOfClasses, size(train.patterns,1));

              prob_pertenencia_claseTest = ones(nOfClasses, size(test.patterns,1));
              
              patrones = train.patterns(train.targets==1,:);
              etiq = train.targets(train.targets == 1);
              for i = 2:nOfClasses,
                  patrones = [patrones ; train.patterns(train.targets==i,:)];
                  etiq = [etiq ; train.targets(train.targets == i)];
              end
             patrones= patrones';
             train.targets = etiq;
		imbalanced = false;
             equalDistributed = true;
 
              
             pesos = zeros(nOfClasses,1);
             
               for i = 1:nOfClasses,
                   
                   nmenor = sum(classes<i);
                   nmayor = sum(classes>i);
                   
                   clase_anterior = numel(train.targets(train.targets < i));
                   %clase_actual = numel(train.targets(train.targets == i));
                   clase_posterior = numel(train.targets(train.targets > i));
                   
                   if clase_anterior == 0, 
                        etiquetas = [ train.targets(train.targets==1) ; ones(size(train.targets(train.targets>1)))*2];
                   elseif clase_posterior ==0,
                        etiquetas = [ ones(size(train.targets(train.targets<i))) ;  ones(size(train.targets(train.targets==i)))*2];
                   else
                        etiquetas = [ ones(size(train.targets(train.targets<i))) ; ones(size(train.targets(train.targets==i)))*2; ones(size(train.targets(train.targets>i)))*3];
                   end

                   
                   %% Train

                   [projection, thresholds]= obj.train( patrones, etiquetas, param);  

                   [projected] = obj.test( patrones, projection, thresholds, patrones, param.k);
                   probTrain = obj.calculateProbabilities(projected, thresholds);
                   
                   [projected] = obj.test( test.patterns', projection, thresholds, patrones, param.k);
                   
                    probTest = obj.calculateProbabilities(projected, thresholds);


                    if equalDistributed==true,
                       for j = 1: nOfClasses,
                        if nmayor~= 0 && nmenor ~=0,
                            if(j<i)
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* (probTrain(1,:)/nmenor);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* (probTest(1,:)/nmenor);
                                pesos(j) = pesos(j) + 1/nmenor;
                            elseif (j>i)
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* (probTrain(3,:)/nmayor);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* (probTest(3,:)/nmayor);
                                pesos(j) = pesos(j) + 1/nmayor;
                            else
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* probTrain(2,:);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* probTest(2,:);
                                pesos(j) = pesos(j) + 1;
                            end
                        elseif i==j,
                            if nmenor == 0,
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .*  probTrain(1,:);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .*  probTest(1,:);
                                pesos(j) = pesos(j) + 1;
                            else
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* probTrain(2,:);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* probTest(2,:);
                                pesos(j) = pesos(j) + 1;
                            end
                        else
                            if nmenor == 0,
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* (probTrain(2,:)/nmayor);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* (probTest(2,:)/nmayor);
                                pesos(j) = pesos(j) + 1/nmayor;
                            else
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* (probTrain(1,:)/nmenor);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* (probTest(1,:)/nmenor);
                                pesos(j) = pesos(j) + 1/nmenor;
                            end
                        end
                           
                        end
                   elseif imbalanced == true,
                       for j = 1: nOfClasses,
                        if nmayor~= 0 && nmenor ~=0,
                            if(j<i)
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* (probTrain(1,:)*(1-(n(j)/clase_anterior)));
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* (probTest(1,:)*(1-(n(j)/clase_anterior)));
                                pesos(j) = pesos(j) + (1-(n(j)/clase_anterior));
                            elseif (j>i)
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* (probTrain(3,:)*(1-(n(j)/clase_posterior)));
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* (probTest(3,:)*(1-(n(j)/clase_posterior)));
                                pesos(j) = pesos(j) + (1-(n(j)/clase_posterior));
                            else
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* probTrain(2,:);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* probTest(2,:);
                                pesos(j) = pesos(j) + 1;
                            end
                        elseif i==j,
                            if nmenor == 0,
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .*  probTrain(1,:);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .*  probTest(1,:);
                                pesos(j) = pesos(j) + 1;
                            else
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* probTrain(2,:);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* probTest(2,:);
                                pesos(j) = pesos(j) + 1;
                            end
                        else
                            if nmenor == 0,
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* (probTrain(2,:)*(1-(n(j)/clase_posterior)));
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* (probTest(2,:)*(1-(n(j)/clase_posterior)));
                                pesos(j) = pesos(j) +(1- (n(j)/clase_posterior));
                            else
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* (probTrain(1,:)*(1-(n(j)/clase_anterior)));
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* (probTest(1,:)*(1-(n(j)/clase_anterior)));
                                pesos(j) = pesos(j) + (1-(n(j)/clase_anterior));
                            end
                        end
                           
                        end
                           
                   else
                       for j = 1: nOfClasses,
                        if nmayor~= 0 && nmenor ~=0,
                            if(j<i)
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* (probTrain(1,:)*((n(j)/clase_anterior)));
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* (probTest(1,:)*((n(j)/clase_anterior)));
                                pesos(j) = pesos(j) + n(j)/clase_anterior;
                            elseif (j>i)
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* (probTrain(3,:)*((n(j)/clase_posterior)));
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* (probTest(3,:)*((n(j)/clase_posterior)));
                                pesos(j) = pesos(j) + n(j)/clase_posterior;
                            else
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* probTrain(2,:);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* probTest(2,:);
                                pesos(j) = pesos(j) + 1;
                            end
                        elseif i==j,
                            if nmenor == 0,
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .*  probTrain(1,:);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .*  probTest(1,:);
                                pesos(j) = pesos(j) + 1;
                            else
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* probTrain(2,:);
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* probTest(2,:);
                                pesos(j) = pesos(j) + 1;
                            end
                        else
                            if nmenor == 0,
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* (probTrain(2,:)*((n(j)/clase_posterior)));
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* (probTest(2,:)*((n(j)/clase_posterior)));
                                pesos(j) = pesos(j) + n(j)/clase_posterior;
                            else
                                prob_pertenencia_claseTrain(j,:) = prob_pertenencia_claseTrain(j,:) .* (probTrain(1,:)*((n(j)/clase_anterior)));
                                prob_pertenencia_claseTest(j,:) = prob_pertenencia_claseTest(j,:) .* (probTest(1,:)*((n(j)/clase_anterior)));
                                pesos(j) = pesos(j) + n(j)/clase_anterior;
                            end
                        end
                           
                        end
                   end
                   
 
               end
 
                 
                 trainTime = toc;
                 prob_pertenencia_claseTrain = prob_pertenencia_claseTrain ./ (pesos*ones(1,size(prob_pertenencia_claseTrain,2)));
                 prob_pertenencia_claseTest = prob_pertenencia_claseTest ./ (pesos*ones(1,size(prob_pertenencia_claseTest,2)));
                 
                 [aux, clasetrain] = max(prob_pertenencia_claseTrain);

                 [aux, clasetest] = max(prob_pertenencia_claseTest);

                tic;

                testTime = toc;

                
                train.predicted = clasetrain;
                test.predicted = clasetest;
                %train.targets = etiq3;
                %test.targets = etiqtest3;
               % dataSetStatistics = obj.calculateResults(train,test);
                
                
                model_information.predictedTrain = train.predicted';
                model_information.predictedTest = test.predicted';
                model_information.projection = projection;
                model_information.trainTime = trainTime;
                model_information.testTime = testTime;
                model_information.parameters = parameters;
                model_information.projectedTest = projected;
                model_information.projectedTrain = projected;
                model_information.thresholds = thresholds;
                
                %dataSetStatistics.projectedTest = p2;
                %dataSetStatistics.projectedTrain = p;
                %dataSetStatistics.thresholds = thresholds;

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: train (Public)
        % Description: This function train the model for
        %               the KDLOR algorithm.
        % Type: [Array, Array]
        % Arguments: 
        %           trainPatterns --> Trainning data for 
        %                              fitting the model
        %           testTargets --> Training targets
        %           parameters --> Penalty coefficient C 
        %           for the KDLOR method and kernel parameters
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [projection, thresholds]= train( obj,trainPatterns, trainTargets , parameters)
                [dim,numTrain] = size(trainPatterns);

                if(nargin < 2)
                    error('Patterns and targets are needed.\n');
                end

                if length(trainTargets) ~= size(trainPatterns,2)
                    error('Number of patterns and targets should be the same.\n');
                end

                    if(nargin < 4)
                            % Default parameters
                            d=10;
                            u=0.001;

                            switch obj.kernelType
                                case 'rbf'
                                    kernelParam = 1;
                                case 'sigmoid'
                                    kernelParam = [1,2];
                                case 'linear'
                                    kernelParam = 1;
                            end
                    else
                            d = parameters.C;
                            u = parameters.u;
                            kernelParam = parameters.k;
                    end



                % Compute the Gram or Kernel matrix
                kernelMatrix = computeKernelMatrix(trainPatterns, trainPatterns,obj.kernelType, kernelParam);
                dim2 = numTrain; 
                numClasses = length(unique(trainTargets));
                meanClasses = zeros(numClasses,dim2);

                Q=zeros(numClasses-1, numClasses-1);
                c=zeros(numClasses-1,1);
                A=ones(numClasses-1,numClasses-1);
                A=-A;
                b=zeros(numClasses-1,1);
                E=ones(1,numClasses-1);

                aux=zeros(1,dim2);
                N=hist(trainTargets,1:numClasses);

                H = sparse(dim2,dim2); 

                % Calculate the mean of the classes and the H matrix
                for currentClass = 1:numClasses,
                  meanClasses(currentClass,:) = mean(kernelMatrix(:,( trainTargets == currentClass )),2);
                  
                  H = H + kernelMatrix(:,( trainTargets == currentClass ))*(eye(N(1,currentClass),N(1,currentClass))-ones(N(1,currentClass),N(1,currentClass))/sum( trainTargets == currentClass ))*kernelMatrix(:,( trainTargets == currentClass ))';
                end

                % Avoid ill-posed matrixes
                H = H +  u*eye(dim2,dim2);

                % Calculate the Q matrix for the optimization problem
                for i = 1:numClasses-1,
                    for j = i:numClasses-1,
                        Q(i,j) = (meanClasses(i+1,:)-meanClasses(i,:))*inv(H)*(meanClasses(j+1,:)-meanClasses(j,:))';
                        % Force the matrix to be symmetric
                        Q(j,i)=Q(i,j);
                    end
                end

             vlb = zeros(numClasses-1,1);    % Set the bounds: alphas and betas >= 0
             vub = Inf*ones(numClasses-1,1); %                 alphas and betas <= Inf
             x0 = zeros(numClasses-1,1);     % The starting point is [0 0 0 0]

             % Choice the optimization method

                switch upper(obj.optimizationMethod)
                    case 'QUADPROG'
                        alpha = quadprog2(Q,c,A,b,E,d,vlb,vub,[],[]);
                    case 'CVX'

                        cvx_begin
                        cvx_quiet(true)
                        variables alpha(numClasses-1)
                        minimize( 0.5*alpha'*Q*alpha );
                        subject to
                            (ones(1,numClasses-1)*alpha) == d;
                            alpha >= 0;
                        cvx_end
                    case 'QP'
                        alpha = qp(Q, c, E, d, vlb, vub,x0,1,0);
                    otherwise
                        error('Invalid value for optimizer\n');
                end

                % Calculate Sum_{k=1}^{K-1}(alpha_{k}*(M_{k+1}-M_{k}))
                for currentClass = 1:numClasses-1,
                    aux = aux + alpha(currentClass)*(meanClasses(currentClass+1,:)-meanClasses(currentClass,:));
                end
                % W = 0.5 * H^{-1} * aux
                projection = 0.5*inv(H)*aux';
                thresholds = zeros(numClasses-1, 1);

                % Calculate the threshold for each couple of classes
                for currentClass = 1:numClasses-1,
                    thresholds(currentClass) = (projection'*(meanClasses(currentClass+1,:)+meanClasses(currentClass,:))')/2;
                end
            
            
            
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
        %           kernelParam --> kernel parameter for KDLOR
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [projected1] = test(obj, testPatterns, projection,thresholds, trainPatterns, kernelParam)
            

                kernelMatrix2 = computeKernelMatrix(trainPatterns, testPatterns,obj.kernelType, kernelParam);

                projected1 = projection'*kernelMatrix2;

        end    
        
        
         function [g] = calculateProbabilities(obj, projected, thresholds)
            nOfClasses = size(thresholds,1)+1;
	    if (numel(thresholds)==2)
                deseada=4.0;
                actual=abs(thresholds(2) - thresholds(1));
                if actual<4,
                    projected = projected*(deseada/actual);
                    thresholds = thresholds*(deseada/actual);
                end
            end


            f = zeros(nOfClasses, size(projected',1));
            g = zeros(nOfClasses, size(projected',1));
            

               for i=1:(nOfClasses-1)
                    f(i,:) = cummulativeProb(projected',thresholds(i));
               end
               f(nOfClasses,:) = ones(1, size(projected',2));
              
               g(1,:) = f(1,:);
               for i=2:nOfClasses
                    g(i,:)=f(i,:)-f(i-1,:);
               end


            end      
            
    end
    
end

