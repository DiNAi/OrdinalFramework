experimentos=dir(fullfile(pwd, '/Experiments/'))
datasets=[];
fileID = fopen('resumen.csv','w');

for k=1:length(experimentos)
    FileNames=experimentos(k).name;
    if( strcmp(FileNames,'.')~=1 &&  strcmp(FileNames,'..')~=1)
        ruta=fullfile(pwd, '/Experiments/',experimentos(k).name,'/Results/mean-results.csv');
        cont = dir(fullfile(pwd, '/Experiments/',experimentos(k).name,'/Results/'));
        for m=1:length(cont)
            if(isdir(fullfile(pwd, '/Experiments/',experimentos(k).name,'/Results/',cont(m).name)))
                %fullfile(pwd, '/Experiments/',experimentos(k).name,'/Results/',cont(m).name)
                name=cont(m).name;
                if( strcmp(name,'.')~=1 &&  strcmp(name,'..')~=1)
                    nombre=name;
                end
            end
        end
        
        
        ii = strfind(nombre,'-');
        indice = ii(1);
        dataset=substring(nombre, 0, indice-2);
        ruta;
        %csvread(ruta);
        fid = fopen(ruta);
        %fgets();
        tline = fgets(fid);
        if(k==3)
            tline = strrep(tline, ',', ';');
            fprintf(fileID,'%s',tline);
        end
        
        tline = fgets(fid);
        fclose(fid);
        % escibir tline
        tline = strrep(tline, ',', ';');
        fprintf(fileID,'%s',tline);
    end
    
    
end


fclose(fileID);

