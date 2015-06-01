classdef EORELM < Algorithm
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
        % Variable: parameters (Private)
        % Description: This variable keeps the values for the C penalty
        %               coefficient
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        parameters = struct('C', [])
        f_fitness = MAE;
        nGen = 2;
        numIndividuos = 10;
        nNeuronas = 100;
        crossoverRate = 0.8;
        activationFunction = 'rbf'
        
        algorithm = ELM
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
        function obj = EORELM(opt)
            obj.name = 'Evolutive Ordinal Regression Extreme Learning Machine';
            % This method has one parameter, the penalty coefficient C.
            % This method is deterministic.
            % This method don't use kernel functions.
            obj.kernelType = 'no';
            if(nargin > 1)
                obj.optimizationMethod = opt;
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
        function dataSetStatistics = runAlgorithm(obj,train, test, parameter)

                obj.algorithm.classifier = 'regression';
                obj.algorithm.activationFunction = obj.activationFunction;
                
                tic;
                [model]= obj.train( train); 

                [train.projected, train.predicted] = obj.test( train, model);
                % Time information for training

                trainTime = toc;

                tic;
                [test.projected, test.predicted] = obj.test( test, model);
                % time information for testing
                testTime = toc;

                dataSetStatistics.predictedTrain = train.predicted';
                dataSetStatistics.predictedTest = test.predicted';
                dataSetStatistics.trainTime = trainTime;
                dataSetStatistics.testTime = testTime;
                dataSetStatistics.parameters = parameter;
                dataSetStatistics.projectedTest = test.projected;
                dataSetStatistics.projectedTrain = train.projected;
                dataSetStatistics.thresholds = -1;
                dataSetStatistics.projection = -1;
        end
        
        function [model, mejorIndividuo]= train( obj,train)
                   
                poblacion = obj.generaPoblacion(numel(unique(train.targets)));
                poblacion = obj.reparaPoblacion(poblacion);

                for i =1:obj.nGen,
                    disp(i);
                    fitness = obj.evaluaPoblacion(poblacion,train);
                    poblacion_auxiliar = obj.mutaPoblacion(poblacion);
                    poblacion_auxiliar = obj.cruzaPoblacion(poblacion,poblacion_auxiliar);
                    poblacion_auxiliar = obj.reparaPoblacion(poblacion_auxiliar);
                    fitness_auxiliar = obj.evaluaPoblacion(poblacion_auxiliar,train);
                    poblacion = obj.seleccion(poblacion,poblacion_auxiliar,fitness,fitness_auxiliar);
                end
                
                fitness = obj.evaluaPoblacion(poblacion,train);
                
                [minimo_fitness, indice] = min(fitness);
                mejorIndividuo = poblacion(indice,:);

                auxTrain = train;
                etiquetas = obj.individualToTargets(mejorIndividuo);
                [auxTrain.targets] = obj.relabel(auxTrain.targets,etiquetas);
                param.hiddenN = (fix(abs(mejorIndividuo(end))));
                model = obj.algorithm.train( auxTrain, param);
                model.bestIndividual = mejorIndividuo; 
		
        end

        function poblacion = reparaPoblacion(obj,poblacion)
            poblacion(:,end) = fix(abs(poblacion(:,end)))+1;
            poblacion(:,poblacion(:,end)>obj.nNeuronas) = obj.nNeuronas;
        end
        
        function [poblacion] = generaPoblacion(obj,nOfClasses)
            etiquetas = (rand(obj.numIndividuos,nOfClasses-1)*2) - 1;
            neuronas = rand(obj.numIndividuos,1)*obj.nNeuronas;
            poblacion = [etiquetas, neuronas];
            
        end
        
        function [poblacion_auxiliar] = mutaPoblacion(obj,poblacion)
            poblacion_auxiliar=zeros(size(poblacion));
            for i = 1:obj.numIndividuos,
                seleccionados = randi(obj.numIndividuos,3,1);
                while(numel(unique(seleccionados))~=3)
                    seleccionados = randi(obj.numIndividuos,3,1);
                end
                poblacion_auxiliar(i,:) = poblacion(seleccionados(1),:) + ((rand(1,size(poblacion,2))*2) - 1).*(poblacion(seleccionados(2),:)-poblacion(seleccionados(3),:));
            end
        end
        
        function [poblacion_nueva] = cruzaPoblacion(obj,poblacion,poblacion_auxiliar)
            poblacion_nueva=poblacion_auxiliar;
            rnbr = randi(size(poblacion,2),obj.numIndividuos,1);
            randbj = rand(size(poblacion));
            
            for i = 1:obj.numIndividuos,
                for j=1:size(poblacion,2),
                    if(randbj(i,j)>obj.crossoverRate && j~=rnbr(i))
                        poblacion_nueva(i,j) = poblacion(i,j);
                    end
                end
            end
        end
        
        function [poblacion_final] = seleccion(obj, poblacion,poblacion_auxiliar,fitness,fitness_auxiliar)
            poblacion_final = poblacion; 
            for i=1:obj.numIndividuos,
                if(fitness_auxiliar(i)<fitness(i))
                    poblacion_final(i,:) = poblacion_auxiliar(i,:);
                end
            end
        end

	function [newTargets] = relabel(obj,targets,etiquetas)
                for j=1:numel(unique(etiquetas)),
                      newTargets(targets==j) = etiquetas(j);
                end
	end        

	function [targets] = individualToTargets(obj,indidivual)
		targets = [0, cumsum(abs(indidivual(1:end-1)),2)];
	end

	function [targets] = poblationToTargets(obj,poblacion)
		targets = [zeros(obj.numIndividuos,1), cumsum(abs(poblacion(:,1:end-1)),2)];
	end

        function [fitness] = evaluaPoblacion(obj,poblacion,train)
            % Hay que redondear el numero de neuronas al entero superior
            % (también hacerle valor absoluto, primero)
         
            etiquetas = obj.poblationToTargets(poblacion);
      
            flag = false;
            for i=1:size(train.uniqueTargets,1),
                if(train.nOfPattPerClass(i)==1)
                    train.patterns = [train.patterns; train.patterns(train.targets==train.uniqueTargets(i),:)];
                    train.targets = [train.targets; train.targets(train.targets==train.uniqueTargets(i),:)];
                    flag = true;
                end
            end
            
            if flag 
                [trainSet.nOfPatterns trainSet.dim] = size(train.patterns);
                trainSet.nOfPattPerClass = sum(repmat(train.targets,1,size(train.uniqueTargets,1))==repmat(train.uniqueTargets',size(train.targets,1),1));
            end
            
            
            for i=1:obj.numIndividuos,
                CVO = cvpartition(train.targets,'k',5);
                for ff = 1:CVO.NumTestSets,
                        % Build fold dataset
                        trIdx = CVO.training(ff);
                        teIdx = CVO.test(ff);

                        auxTrain = train;
                        auxTest = train;
                        
                        auxTrain.targets = train.targets(trIdx,:);
                        auxTrain.patterns = train.patterns(trIdx,:);
                        auxTest.targets = train.targets(teIdx,:);
                        auxTest.patterns = train.patterns(teIdx,:);
                        
                        [auxTrain.nOfPatterns auxTrain.dim] = size(auxTrain.patterns);
                        [auxTest.nOfPatterns auxTest.dim] = size(auxTest.patterns);
                
                        auxTrain.nOfPattPerClass = sum(repmat(auxTrain.targets,1,size(auxTrain.uniqueTargets,1))==repmat(auxTrain.uniqueTargets',size(auxTrain.targets,1),1));
                        auxTest.nOfPattPerClass = sum(repmat(auxTest.targets,1,size(auxTest.uniqueTargets,1))==repmat(auxTest.uniqueTargets',size(auxTest.targets,1),1));
                
                        [auxTrain.targets] = obj.relabel(auxTrain.targets,etiquetas(i,:));
                        
                        [model_information] = obj.algorithm.runAlgorithm(auxTrain,auxTest,(fix(abs(poblacion(i,end)))));
                        model_information.predictedTrain = obj.projectedToTest(model_information.predictedTrain,etiquetas(i,:));
                        model_information.predictedTest = obj.projectedToTest(model_information.predictedTest,etiquetas(i,:));
                        value(ff) = obj.f_fitness.calculateMetric(model_information.predictedTest',auxTest.targets);
                end

                fitness(i) = mean(value);
            end
        end
    
        function [projected, testTargets]= test( obj, test, model)

                auxTest = test;
                etiquetas = obj.individualToTargets(model.bestIndividual);
                [auxTest.targets] = obj.relabel(auxTest.targets,etiquetas);
                [projected] = obj.algorithm.test(auxTest,model);

                testTargets = obj.projectedToTest(projected,etiquetas);

        end

	function [testTargets] = projectedToTest(obj,predictions,uniqueTargets)
	
            testTargets = zeros(1,numel(predictions));
            distancias = zeros(1,numel(uniqueTargets));

            for i=1:numel(predictions),
                for j=1:numel(distancias),
                    distancias(j) = abs( predictions(i)-uniqueTargets(j) );
                end
                [falsa,testTargets(i)] = min(distancias);
            end

        end

    end
    
end

