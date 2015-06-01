classdef Utilities < handle
    % Algorithm Abstract interface class
    % Abstract class which defines Machine Learning algorithms.
    % It describes some common methods and variables for all the
    % algorithMs.
    
    properties
        
        
    end
    
    
    methods (Static = true)
        
        function runExperiments(ficheroExperimentos)
            c = clock;
            %addpath libsvm-2.81/
            %addpath libsvm-weights-3.12/matlab
            addpath('Measures');
            addpath('Algorithms');
            dirSuffix = [num2str(c(1)) '-' num2str(c(2)) '-'  num2str(c(3)) '-' num2str(c(4)) '-' num2str(c(5)) '-' num2str(uint8(c(6)))];
            disp('Setting up experiments...');
            logsDir = Utilities.configureExperiment(ficheroExperimentos,dirSuffix);
            
            ficheros_experimentos = dir([logsDir '/' 'exp-*']);
            
            
            for i=1:numel(ficheros_experimentos),
                if ~strcmp(ficheros_experimentos(i).name(end), '~')
                    auxiliar = Experiment;
                    disp(['Running experiment ', ficheros_experimentos(i).name]);
                    auxiliar.launch([logsDir '/' ficheros_experimentos(i).name]);
                end
            end
            
            disp('Calculating results...');
            Utilities.results([logsDir '/' 'Results']);
            rmpath('Measures');
            rmpath('Algorithms');
            
        end
        
        
        function results(experiment_folder)
            addpath('Measures');
            addpath('Algorithms');
            
            experimentos = dir([experiment_folder '/' '*-*']);
            
            % Recorremos las carpetas de experimentos ejecutados para sacar los
            % resultados
            for i=1:numel(experimentos),
                fid = fopen([experiment_folder '/' experimentos(i).name '/' 'dataset'],'r');
                ruta_dataset = fgetl(fid);
                fclose(fid);
                
                predicted_files = dir([experiment_folder '/' experimentos(i).name '/' 'Predictions' '/' 'test_*']);
                time_files = dir([experiment_folder '/' experimentos(i).name '/' 'Times' '/' '*.*']);
                
                hyp_files = dir([experiment_folder '/' experimentos(i).name '/' 'OptHyperparams' '/' '*.*']);
                guess_files = dir([experiment_folder '/' experimentos(i).name '/' 'Guess' '/' 'test_*']);
                
                % Discard "." and ".."
                time_files = time_files(3:numel(time_files));
                hyp_files = hyp_files(3:numel(hyp_files));
                real_files = dir([ruta_dataset '/' 'test_*']);
                
                act = cell(1, numel(predicted_files));
                pred = cell(1, numel(predicted_files));
                proj = cell(1, numel(guess_files));
                
                times = [];
                param = [];
                % Recorremos cada fichero de test de la carpeta de resultados
                for j=1:numel(predicted_files)
                    pred{j} = importdata([experiment_folder '/' experimentos(i).name '/' 'Predictions' '/' predicted_files(j).name]);
                    times(:,j) = importdata([experiment_folder '/' experimentos(i).name '/' 'Times' '/' time_files(j).name]);
                    
                    proj{j} = importdata([experiment_folder '/' experimentos(i).name '/' 'Guess' '/' guess_files(j).name]);
                    if length(hyp_files)~=0
                        struct_hyperparams(j) = importdata([experiment_folder '/' experimentos(i).name '/' 'OptHyperparams' '/' hyp_files(j).name],',');
                        for z = 1:numel(struct_hyperparams(j).data)
                            param(z,j) = struct_hyperparams(j).data(z);
                        end
                    end
                    actual = importdata([ruta_dataset '/' real_files(j).name]);
                    act{j} = actual(:,end);
                end
                
                
                names = {'Dataset', 'Deviations','Fmeasure', 'Acc', 'MAcc', 'GM', 'MS', 'AUC', 'MAE', 'AMAE', 'MMAE', 'MinMAE','RSpearman', 'Tkendall', 'Wkappa', 'TrainTime', 'TestTime', 'CrossvalTime'};
                
		if length(hyp_files)~=0
                    for j=1:numel(struct_hyperparams(1).textdata),
                        names{numel(names)+1} = struct_hyperparams(1).textdata{j};
                    end
        end
                
                deviations = cell2mat(cellfun(@Deviation.calculateMetric, act, pred, 'UniformOutput', false));
                fscoress = cell2mat(cellfun(@Fscore.calculateMetric, act, pred, 'UniformOutput', false));
                accs = cell2mat(cellfun(@CCR.calculateMetric, act, pred, 'UniformOutput', false));
                mccrs = cell2mat(cellfun(@MCCR.calculateMetric, act, pred, 'UniformOutput', false));
                gms = cell2mat(cellfun(@GM.calculateMetric, act, pred, 'UniformOutput', false));
                mss = cell2mat(cellfun(@MS.calculateMetric, act, pred, 'UniformOutput', false));
                aucs = cell2mat(cellfun(@AUC.calculateMetric, act, proj, 'UniformOutput', false));
                maes = cell2mat(cellfun(@MAE.calculateMetric, act, pred, 'UniformOutput', false));
                amaes = cell2mat(cellfun(@AMAE.calculateMetric, act, pred, 'UniformOutput', false));
                maxmaes = cell2mat(cellfun(@MMAE.calculateMetric, act, pred, 'UniformOutput', false));
                minmaes = cell2mat(cellfun(@MinMAE.calculateMetric, act, pred, 'UniformOutput', false));
                spearmans = cell2mat(cellfun(@Spearman.calculateMetric, act, pred, 'UniformOutput', false));
                kendalls = cell2mat(cellfun(@Tkendall.calculateMetric, act, pred, 'UniformOutput', false));
                wkappas = cell2mat(cellfun(@Wkappa.calculateMetric, act, pred, 'UniformOutput', false));
                results_matrix = [deviations;fscoress;accs; mccrs; gms; mss;aucs; maes; amaes; maxmaes; minmaes; spearmans; kendalls; wkappas; times(1,:); times(2,:); times(3,:)];
                if length(hyp_files)~=0
                    for j=1:numel(struct_hyperparams(1).textdata),
                        results_matrix = [results_matrix ; param(j,:) ];
                    end
                end
                
                results_matrix = results_matrix';
                
                % Results for the independent dataset
                fid = fopen([experiment_folder '/' experimentos(i).name '/' 'results.csv'],'w');
                for h = 1:numel(names),
                    fprintf(fid, '%s,', names{h});
                end
                fprintf(fid,'\n');
                
                for h = 1:size(results_matrix,1),
                    fprintf(fid, '%s,', real_files(h).name);
                    for z = 1:size(results_matrix,2),
                        fprintf(fid, '%f,', results_matrix(h,z));
                    end
                    fprintf(fid,'\n');
                end
                fclose(fid);
                
                % Confusion matrices
                fid = fopen([experiment_folder '/' experimentos(i).name '/' 'matrices.txt'],'w');
                
                for h = 1:size(results_matrix,1),
                    fprintf(fid, '%s\n----------\n', real_files(h).name);
                    cm = confusionmat(act{h},pred{h});
                    for ii = 1:size(cm,1),
                        for jj = 1:size(cm,2),
                            fprintf(fid, '%d ', cm(ii,jj));
                        end
                        fprintf(fid, '\n');                        
                    end
                end
                fclose(fid);
                
                medias = mean(results_matrix,1);
                stdev = std(results_matrix,0,1);
                
                fid = fopen([experiment_folder '/' 'mean-results.csv'],'at');
                
                if i==1,
                    fprintf(fid, 'Dataset-Experiment,');
                    
                    for h = 2:numel(names),
                        fprintf(fid, 'Mean%s,Std%s,', names{h},names{h});
                    end
                    fprintf(fid,'\n');
                end
                
                fprintf(fid, '%s,', experimentos(i).name);
                for h = 1:numel(medias),
                    fprintf(fid, '%f,%f,', medias(h), stdev(h));
                end
                fprintf(fid,'\n');
                fclose(fid);
                
            end
            rmpath('Measures');
            rmpath('Algorithms');
            
            
        end
        
        
        function logsDir = configureExperiment(ficheroExperimentos,dirSuffix)
            
            if( ~(exist(ficheroExperimentos,'file')))
                fprintf('The file %s does not exist!!!\n',ficheroExperimentos);
                return;
            end
            
            logsDir = ['Experiments' '/' 'exp-' dirSuffix];
            resultados = [logsDir '/' 'Results'];
            mkdir(logsDir);
            mkdir(resultados);
            fid = fopen(ficheroExperimentos,'r+');
            num_experiment = 0;
            
            while ~feof(fid),
                nueva_linea = fgetl(fid);
                if strncmpi(nueva_linea,'%',1),
                    %Doing nothing!
                elseif strcmpi('new experiment', nueva_linea),
                    num_experiment = num_experiment + 1;
                    id_experiment = num2str(num_experiment);
                    auxiliar = '';
                elseif strcmpi('name', nueva_linea),
                    id_experiment = [fgetl(fid) num2str(num_experiment)];
                elseif strcmpi('dir', nueva_linea),
                    directory = fgetl(fid);
                elseif strcmpi('datasets', nueva_linea),
                    datasets = fgetl(fid);
                elseif strcmpi('end experiment', nueva_linea),
                    fichero_ini = [logsDir '/' 'exp-' id_experiment];
                    [matchstart,matchend,tokenindices,matchstring,tokenstring,tokenname,splitstring] = regexpi(datasets,',');
                    if( ~(exist(directory,'dir')))
                        fprintf('The directory %s does not exist!!!\n',directory);
                        return;
                    end
                    [train, test] = Utilities.processDirectory(directory,splitstring);
                    for i=1:numel(train)
                        aux_directory = [resultados '/' splitstring{i} '-' id_experiment];
                        mkdir(aux_directory);
                        mkdir([aux_directory '/' 'OptHyperparams']);
                        mkdir([aux_directory '/' 'Times']);
                        mkdir([aux_directory '/' 'Models']);
                        mkdir([aux_directory '/' 'Predictions']);
                        mkdir([aux_directory '/' 'Guess']);
                        fichero = [resultados '/' splitstring{i} '-' id_experiment '/' 'dataset'];
                        fich = fopen(fichero,'w');
                        fprintf(fich, [directory '/' splitstring{i} '/' 'gpor']);
                        fclose(fich);
                        for j=1:numel(train{i}),
                            fichero = [fichero_ini '-' splitstring{i} '-' num2str(j)];
                            fich = fopen(fichero,'w');
                            fprintf(fich, ['directory\n' directory '/' splitstring{i} '/' 'gpor' '\n']);
                            fprintf(fich, ['train\n' train{i}(j).name '\n']);
                            fprintf(fich, ['test\n' test{i}(j).name '\n']);
                            fprintf(fich, ['results\n' resultados '/' splitstring{i} '-' id_experiment '\n']);
                            fprintf(fich, auxiliar);
                            fclose(fich);
                        end
                    end
                else
                    auxiliar = [auxiliar nueva_linea '\n'];
                end
                
            end
            fclose(fid);
            
        end
        
        
        function [trainFileNames, testFileNames] = processDirectory(directory, dataSetNames)
            dbs = dir(directory);
            dbs(2) = [];
            dbs(1) = [];
            validDataSets = 1;
            
            if strcmpi(dataSetNames{1}, 'all')
                for dd=1:size(dbs,1)
                    % get directory
                    if dbs(dd).isdir,
                        ejemplo = [directory '/' dbs(dd).name '/' 'gpor' '/' 'train_' dbs(dd).name '.*'];
                        trainFileNames{validDataSets, :} = dir(ejemplo);
                        ejemplo = [directory '/' dbs(dd).name '/' 'gpor' '/' 'test_' dbs(dd).name '.*'];
                        testFileNames{validDataSets, :} = dir(ejemplo);
                        validDataSets = validDataSets + 1;
                    end
                    
                end
            else
                for j=1:numel(dataSetNames),
                    isdirectory = [directory '/' dataSetNames{j}];
                    if(isdir(isdirectory)),
                        ejemplo = [isdirectory '/' 'gpor' '/' 'train_' dataSetNames{j} '.*'];
                        trainFileNames{validDataSets, :} = dir(ejemplo);
                        ejemplo = [isdirectory '/' 'gpor' '/' 'test_' dataSetNames{j} '.*'];
                        testFileNames{validDataSets, :} = dir(ejemplo);
                        validDataSets = validDataSets + 1;
                    end
                end
                
                
            end
        end
        
        function runExperiment(fichero)
            addpath('Measures');
            addpath('Algorithms');
            
            auxiliar = Experiment;
            auxiliar.launch(fichero);

            rmpath('Measures');
            rmpath('Algorithms');
            
        end
        
        
    end
    
    
    
end


