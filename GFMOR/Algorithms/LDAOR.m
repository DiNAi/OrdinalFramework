classdef LDAOR < Algorithm
    % LDAOR Linear Discriminant Analysis for Ordinal Regression
    %   This class derives from the Algorithm Class and implements the
    %   LDAOR method. 
    %   Characteristics: 
    %               -Kernel functions: No
    %               -Ordinal: Yes
    %               -Parameters: 
    %                       -C: Penalty coefficient
    
    properties
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Variable: optimizationMethod (Private)
        % Description: It specifies the method used for optimize the
        % discriminant funcion of the model. It can be quadprog, qp, or svx
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        optimizationMethod = 'quadprog'
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Variable: parameters (Private)
        % Description: This variable keeps the values for the C penalty
        %               coefficient
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        parameters
        name_parameters = {'C', 'u'}
    end
    
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: LDAOR (Public Constructor)
        % Description: It constructs an object of the class LDAOR and sets its
        %               characteristics.
        % Type: Void
        % Arguments: 
        %           opt--> Type of optimization used in the method.
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = LDAOR(opt)
            obj.name = 'Linear Discriminant Analysis for Ordinal Regression';
            % This method don't use kernel functions.
            obj.kernelType = 'no';
            if(nargin > 1)
                obj.optimizationMethod = opt;
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: set.optimizationMethod (Public)
        % Description: It verifies if the value for the variable optimizationMethod 
        %           is correct.
        % Type: Void
        % Arguments: 
        %           value--> Value for the variable optimizationMethod.
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = set.optimizationMethod(obj, value)
            if ~(strcmpi(lower(value),'quadprog') || strcmpi(lower(value),'qp') || strcmpi(lower(value),'cvx'))
                   fprintf('Invalid value for optimizer\n');
            else
                   obj.optimizationMethod = value;
            end 
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: defaultParameters (Public)
        % Description: It assigns the parameters of the algorithm to a default value.
        % Type: Void
        % Arguments: 
        %           No arguments for this function.
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = defaultParameters(obj)
                obj.parameters.C = [0.1,1,10,100];
                obj.parameters.u = [0.01,0.001,0.0001,0.00001,0.000001];
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: runAlgorithm (Public)
        % Description: This function runs the corresponding algorithm, fitting the
        %               model, and testing it in a dataset. It also calculates some
        %               statistics as CCR, Confusion Matrix, and others. 
        % Type: It returns a set of statistics (Struct) 
        % Arguments: 
        %           train --> trainning data for fitting the model
        %           test --> test data for validation
        %           parameter --> Penalty coefficient C for the LDAOR method
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function model_information = runAlgorithm(obj,train, test, parameter)
                param.C = parameter(1);
                
                if length(parameter)>1,
                    param.u = parameter(2);
                else
                    param.u = 0.01;
                end
                
                c1 = clock;
                [model]= obj.train( train.patterns', train.targets', param);  
                % Time information for training
                c2 = clock;
                model_information.trainTime = etime(c2,c1);

                c1 = clock;
                [model_information.projectedTrain, model_information.predictedTrain] = obj.test( train.patterns', model);
                [model_information.projectedTest, model_information.predictedTest] = obj.test( test.patterns', model);
                c2 = clock;
                % time information for testing
                model_information.testTime = etime(c2,c1);

                model_information.model = model;
                
        end
        
        function [model]= train( obj,trainPatterns, trainTargets, parameters)
                % Analisis discriminante lineal para casos ordinales
                % Entrada:
                %  train [struct] Conjunto de entrenamiento:
                %   .patrones [dim x numTrain] Caracterisiticas del ejemplo.
                %   .etiquetas [1 x numTrain] Etiquetas de clases.
                %   
                %   test [dim x num_test] Conjunto de prueba
                %   optimizer: Variable que nos indica el tipo de optimizer a utilizar
                %       -optimizer = 1 --> quadprog
                %       -optimizer = 2 --> cvx
                %   parametro_C: Variable que nos indica el valor que tomar� el par�metro C
                %   (determinante en la clasificaci�n)
                %   parametro_u: Variable para evitar la singularidad de la matrix Sw
                % Salida:
                %   resultado [struct]
                %     .W mejor proyecci�n
                %     .b 
                %     .proyectados
                %     .test_targets


                    [dim,numTrain] = size(trainPatterns);
                    numClasses = length(unique(trainTargets));
                    classMean = zeros(numClasses,dim);

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
                    else
                            d = parameters.C;
                            u = parameters.u;
                    end

                    % We initialize some matrixes
                    Sw=zeros(dim,dim);
                    Q=zeros(numClasses-1, numClasses-1);
                    c=zeros(numClasses-1,1);
                    A=ones(numClasses-1,numClasses-1);
                    A=-A;
                    b=zeros(numClasses-1,1);
                    E=ones(1,numClasses-1);
                    u=0.001;

                    aux=zeros(1,dim);

                     % Calculate the mean of the classes and the covarianze matrix
                    for currentClass = 1:numClasses,
                      %calcula la media de todas las caracteristicas para una clase en concreto
                      classMean(currentClass,:) = mean(trainPatterns(:,( trainTargets == currentClass )),2);
                      Sw = Sw + (1/numTrain)*cov( (trainPatterns(:,( trainTargets == currentClass )))', 1);
                    end

                    % Avoid ill-posed matrixes
                    Sw = Sw + u*eye(size(Sw));
                    Sw_inv = inv(Sw);
                    % Calculate the Q matrix for the optimization problem
                    for i = 1:numClasses-1,
                        for j = i:numClasses-1,
                            Q(i,j) = (classMean(i+1,:)-classMean(i,:))*Sw_inv*(classMean(j+1,:)-classMean(j,:))';
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
%                             rmpath ../cvx/sets
%                             rmpath ../cvx/keywords
%                             addpath ../cvx
%                             addpath ../cvx/structures
%                             addpath ../cvx/lib
%                             addpath ../cvx/functions
%                             addpath ../cvx/commands
%                             addpath ../cvx/builtins

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
                        aux = aux + alpha(currentClass)*(classMean(currentClass+1,:)-classMean(currentClass,:));
                    end

                    % W = 0.5 * H^{-1} * aux
                    projection = 0.5*Sw_inv*aux';
                    thresholds = zeros(numClasses-1, 1);

                    % Calculate the threshold for each couple of classes
                    for currentClass = 1:numClasses-1,
                        thresholds(currentClass) = (projection'*(classMean(currentClass+1,:)+classMean(currentClass,:))')/2;
                    end
                    
                    model.projection = projection;
                    model.thresholds = thresholds;
                    model.parameters = parameters;
                    model.algorithm = 'LDAOR';
                    
                end

    
        function [projected, testTargets]= test( obj,testPatterns, model)

                numClasses = size(model.thresholds,1)+1;

                projected = model.projection'*testPatterns;  

                % We calculate the projected patterns - each thresholds, and then with
                % the following decision rule we can induce the class each pattern
                % belows.
                projected2 = repmat(projected, numClasses-1,1);
                projected2 = projected2 - model.thresholds*ones(1,size(projected2,2));

                % Asignation of the class
                % f(x) = max {Wx-bk<0} or Wx - b_(K-1) > 0
                wx=projected2;

                % The procedure for that is the following:
                % We assign the values > 0 to NaN
                wx(wx(:,:)>0)=NaN;

                % Then, we choose the bigger one.
                [maximum,testTargets]=max(wx,[],1);

                % If a max is equal to NaN is because Wx-bk for all k is >0, so this
                % pattern below to the last class.
                testTargets(isnan(maximum(:,:)))=numClasses;
                testTargets = testTargets';


         end
    end
    
end

