classdef ORNNELM < Algorithm
    % Ordinal Neural Network Trained by ELM
    %   This class derives from the Algorithm Class and implements the
    %   linear ORNNELM method. 
    %   Characteristics: 
    %               -Kernel functions: No
    %               -Ordinal: Yes
    
    properties
		
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Variable: parameters (Private)
        % Description: No parameters for this algorithm
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        parameters
        name_parameters = {'hiddenN'}
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
		
        function obj = ORNNELM(opt)
            obj.name = 'Ordinal Neural Network with ELM for Ordinal Regression';
            % This method don't use kernel functions.
            obj.kernelType = 'no';
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
            obj.parameters.hiddenN = {5,10,20,30,40,50,60,70,80,90,100};
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
		
        function model_information = runAlgorithm(obj,train, test,parameters)
            
                param.hiddenN = parameters(1);
                
                %Generate the ELM encodings label
                train.uniqueTargets = unique([test.targets ;train.targets]);
                test.uniqueTargets = train.uniqueTargets;
                train.nOfClasses = max(train.uniqueTargets);
                test.nOfClasses = train.nOfClasses;                
                train.nOfPatterns = length(train.targets);
                test.nOfPatterns = length(test.targets);
                
                train.dim = size(train.patterns,2);
                test.dim = train.dim;
                
                
                [train, test] = obj.labelToOrelm(train,test);
                
                train.uniqueTargetsOrelm = unique([test.targetsOrelm ;train.targetsOrelm],'rows');
                test.uniqueTargetsOrelm = train.uniqueTargetsOrelm;
                
                c1 = clock;
                [model]= obj.train( train,param );
                % Time information for training
                c2 = clock;
                model_information.trainTime = etime(c2,c1);
                
                
                c1 = clock;
                [model_information.projectedTrain,model_information.predictedTrain] = obj.test( train.patterns, model);
                [model_information.projectedTest,model_information.predictedTest] = obj.test( test.patterns, model);
                c2 = clock;
                
                % time information for testing
                model_information.testTime = etime(c2,c1);

                model_information.model = model;
                
        end
        
        function [model]= train( obj,train,parameters)
            
                    obj.parameters.hiddenN = parameters.hiddenN;
                    
                    nOfClasses = numel(unique(train.targets));
                    
                    T = train.targetsOrelm.*1000000;
                    
                    % Determination of the cost matrix
                    costDimension = nOfClasses;
                    cost = (repmat(1:costDimension,costDimension,1) - repmat((1:costDimension)',1,costDimension));
                    cost(cost>0) = 0;
                    cost(cost<0) = 1;
                    cost(logical(eye(size(cost))))=1;
                    
                    % Configure the hidden layer
                    InputWeight=rand(obj.parameters.hiddenN,size(train.patterns,2))*2-1;

                    BiasofHiddenNeurons=rand(obj.parameters.hiddenN,1);
                    tempH=InputWeight*train.patterns';
                    ind=ones(1,size(train.patterns,1));
                    BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                    tempH=tempH+BiasMatrix;
                    H = (1 ./ (1 + exp(-tempH)))';
                    
                    % Initialize the padding parameters
                    paddings = zeros(obj.parameters.hiddenN,nOfClasses);
                    
                    % Optimization procedure
                    Tnew = T * inv(cost'); % Cost transpose? Future question
                    
                    % Paddings estimation through pseudoinverse matrix
                    paddings=pinv(H) * Tnew;
                    
                    for i = 1:nOfClasses,
                        A = eye(obj.parameters.hiddenN);
                        index = find(paddings(:,i)<0);
                        
                        %if (isempty(index))
                        if (size(index)  == size(paddings(:,i),1))
                            % All negatives
                            firstPart = inv(H'*H)*A;
                            secondPart = inv(A'*inv(H'*H)*A);
                            thirdPart = -A'*paddings(:,i);
                            additionalICLS = firstPart*secondPart*thirdPart;
                            paddings(:,i)= paddings(:,i)+ additionalICLS;
                            
                        elseif all((size(index) < size(paddings(:,i),1)) & (~isempty(index)))
                            % Some negatives some positives
                            A(:,index)=[];
                            firstPart = inv(H'*H)*A;
                            secondPart = inv(A'*inv(H'*H)*A);
                            thirdPart = -A'*paddings(:,i);
                            additionalICLS = firstPart*secondPart*thirdPart;
                            paddings(:,i)= paddings(:,i)+ additionalICLS;
                        end     
                        
                    end
                    
                    
                                      
                    % Obtain coefficients of the ordinal regression model
                    OutputWeight = cost * paddings';
                    
                    model.hiddenN = obj.parameters.hiddenN;
                    model.InputWeight = InputWeight;
                    model.BiasofHiddenNeurons = BiasofHiddenNeurons;
                    model.OutputWeight = OutputWeight;
                    model.algorithm = 'ELMOR';
                    model.parameters = parameters;

        end

    
        function [ projected,testTargets ]= test( obj, testPatterns, model)

                
                tempH=model.InputWeight*testPatterns';
                ind=ones(1,size(testPatterns,1));
                BiasMatrix=model.BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                tempH=tempH+BiasMatrix;
                H = (1 ./ (1 + exp(-tempH)))';
                
                projected=(H * model.OutputWeight');

                TestPredictedY = obj.orelmToLabel(projected);

                
                testTargets = TestPredictedY;
                projected = testTargets';

         end
    end
    
        methods(Access = private)
        
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
        
        function [finalOutput] = orelmToLabel(obj,predictions)
            
            probabilities = zeros(size(predictions,1),size(predictions,2));
            probabilities(:,1) = predictions(:,1);
            % probabilities
            for i = 2:size(predictions,2)
               probabilities(:,i) = predictions(:,i)-predictions(:,i-1); 
            end
            
            [maxVal,finalOutput] = max(probabilities');
            finalOutput = finalOutput';
            
        end
        


	function [trainSet, testSet] = labelToOrelm(obj,trainSet,testSet)

            %   newTargets = zeros(trainSet.nOfPatterns,trainSet.nOfClasses);
            trainSet.targetsOrelm = zeros(trainSet.nOfPatterns,trainSet.nOfClasses);
            testSet.targetsOrelm = zeros(testSet.nOfPatterns,trainSet.nOfClasses);
            
            for i=1:trainSet.nOfClasses,
                trainSet.targetsOrelm(trainSet.targets<=trainSet.uniqueTargets(i),i) = 1;
                testSet.targetsOrelm(testSet.targets<=trainSet.uniqueTargets(i),i) = 1;
            end
    end
   end
        
    
end

