classdef POVASVM < Algorithm
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
        % Variable: parameters (Public)
        % Type: Struct
        % Description: This variable keeps the values for 
        %               the C penalty coefficient and the 
        %               kernel parameters
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        parameters
        combiner = 'sum'
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
        
        function obj = POVASVM(kernel)
            obj.name = 'Probabilistic One vs All Support Vector Machine model';
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
                addpath libsvm-weights-3.12/
                param.C = parameters(1);
                param.k = parameters(2);
                
                tic;
                options = ['-t 2 -c ' num2str(param.C) ' -g ' num2str(param.k) ' -q'];
                model = obj.train(train.targets, train.patterns, options);
                trainTime = toc;
                
                tic;
                [model_information.projectedTrain, model_information.predictedTrain] = obj.test(train.targets,train.patterns,model);
                [model_information.projectedTest,model_information.predictedTest ] = obj.test(test.targets,test.patterns,model);
                testTime = toc;
                           
                
                 model_information.trainTime = trainTime;
                 model_information.testTime = testTime;
                 model_information.parameters = parameters;

                 model_information.model.SVMModel = model;
                 model_information.model.parameters = parameters;
                 model_information.model.algorithm = 'POVASVM';		
                 model_information.model.kernel = obj.kernelType;

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
        
        function [model]= train( obj, y , x, cmd)   

            labelSet = unique(y);
            labelSetSize = length(labelSet);
            models = cell(labelSetSize,1);

            for i=1:labelSetSize,
                etiquetas = double(y == labelSet(i));
                etiquetas(etiquetas==1) = -1;
                etiquetas(etiquetas==0) = 1;
                weights = ones(size(etiquetas));
                models{i} = svmtrain(weights,etiquetas, x, cmd);
            end

            model = struct('models', {models}, 'labelSet', labelSet);

        end
        
        function [decv, pred]= test(obj, y,x, model)

            labelSet = model.labelSet;
            labelSetSize = length(labelSet);
            models = model.models;
            %decv= zeros(size(y, 1), labelSetSize);
            nclasses = numel(unique(y));
            npatterns = size(x,1);
            if strcmpi('product', obj.combiner),
                probs = ones(npatterns,nclasses);
            else
                probs = zeros(npatterns,nclasses);
            end

            for i=1:labelSetSize
                etiquetas = double(y == labelSet(i));
                [l,a,d] = svmpredict(etiquetas, x, models{i});
                bias = -models{i}.rho;
                decv(:,i) = d * (2 * models{i}.Label(1) - 1);
                projected = d * (2 * models{i}.Label(1) - 1);
                probTest = obj.calculateProbabilities(projected', bias);

                if strcmpi('product', obj.combiner),
                    'Producto'
                    probs(:,i) = probs(:,i).*probTest(1,:)';
                    for j = 1:i-1,
                        probs(:,j) = probs(:,j).*(probTest(2,:)'/(nclasses-1));
                    end
                    for j = i+1:nclasses,  
                        probs(:,j) = probs(:,j).*(probTest(2,:)'/(nclasses-1));                        
                    end
                else
                    probs(:,i) = probs(:,i)+probTest(1,:)';
                    for j = 1:i-1,
                        probs(:,j) = probs(:,j)+(probTest(2,:)'/(nclasses-1));
                    end
                    for j = i+1:nclasses,  
                        probs(:,j) = probs(:,j)+(probTest(2,:)'/(nclasses-1));                        
                    end
                end
                 
            end
            [tmp,pred] = max(probs, [], 2);
            pred = labelSet(pred);

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
