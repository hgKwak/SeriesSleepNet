%% Dataset Loading %%
opts = delimitedTextImportOptions("NumVariables", 22);
  
% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ","; 

% Specify column names and types
opts.VariableNames = ["nsrrid", "pptid", "oahi", "overall_shhs1", "PreRDI", "slptime", "gender", "race", "MStat", "age_s1", "smokstat_s1", "ethnicity", "bmi_s1", "educat", "weight", "waist", "height", "weight20", "lang15", "age_category_s1", "LightOff", "rcrdtime"];
opts.VariableTypes = ["double", "double", "double", "double" , "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "datetime"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
% Specify variable properties
opts = setvaropts(opts, "rcrdtime", "InputFormat", "HH:mm:ss");
% Import the data
shhs1dataset = readtable("./raws/shhs/shhs1-dataset-0.15.0.csv", opts);
clear opts
%% Variable Setting %%
% data_dir MUST include both psg and hypnogram file!
data_dir = "./raws/shhs/polysomnography/edfs/shhs1/";
annot_dir = "./raws/shhs/polysomnography/annotations-events-nsrr/shhs1/";

mannual_starting_point = 0;
female_target_idx = (find((shhs1dataset.oahi < 5 & shhs1dataset.overall_shhs1 == 7 & shhs1dataset.gender == 2)));
%%
target_files = string(shhs1dataset.nsrrid(female_target_idx));

tmp = struct2cell(dir(data_dir + "/*.edf"));
j=1;
for i = 1:length(tmp)
    if (contains(tmp{1,i}, target_files))
        psg_n{:,j} = tmp{:,i};
        j = j+1;
    end
end
tmp = struct2cell(dir(annot_dir + "/*.xml"));  
j=1;
for i = 1:length(tmp)
    if (contains(tmp{1,i}, target_files))
        hyp_n{:,j} = tmp{:,i};
        j = j+1;
    end
end
psg_path = './female_SHHS/psg/';
hyp_path = './female_SHHS/hyp/';

if(exist(psg_path, 'dir'))
    end_file = dir(psg_path + '*.mat');
    end_file = end_file(end).name(1:end-4);
    starting_point = find(contains(psg_n, end_file)==1)+1;
    if mannual_starting_point > starting_point
       starting_point = mannual_starting_point
    end
else
    starting_point = 1;
end
psg_n = psg_n(starting_point: length(target_files));

if(exist(hyp_path, 'dir'))
    end_file = dir(hyp_path + '*.mat');
    end_file = end_file(end).name(1:end-4);
    starting_point = find(contains(hyp_n, end_file)==1)+1;
    if mannual_starting_point > starting_point
       starting_point = mannual_starting_point
    end
else
    starting_point = 1;
end
hyp_n = hyp_n(starting_point: length(target_files));  
 
class = ["Wake|0", "Stage 1 sleep|", "Stage 2 sleep|2", "Stage 3 sleep|3", "Stage 4 sleep|4", "REM sleep|5"];
len=length(psg_n);
     
% 0: Wake
% 1: Stage 1
% 2: Stage 2
% 3: Stage 3/4
% 4: Stage 3/4
% 5: REM stage
% 9: Movement/Wake

% EEG C4 loc: 8
% EEG(sec) C3 loc: 3

% Preprocessing data 
cd("./eeglab")
eeglab;
cd("..")

if(~exist(psg_path, 'dir'))
    mkdir(psg_path);
end

if(~exist(hyp_path, 'dir'))
    mkdir(hyp_path);
end

disp('Number of files to preprocess:' + string(length(target_files) - starting_point + 1))
disp('Starting from file number' + string(starting_point)) 

for i = 1:len
    %[ , chan, signal] = blockEdfLoad(cat(2, char(data_dir), psg_n{i}));
    %sleep_h(i).data{1,:} = transpose(signal{:,3});
    %sleep_h(i).data{2,:} = transpose(signal{:,8});
    % (i).data = cell2mat(sleep_h(i).data);
    [annotation] = parseXML(cat(2, char(annot_dir), hyp_n{i}));
    sleepstages=[];
    disp(string(i+starting_point-1)+' / '+string(length(target_files))+') Data loading... '+string(psg_n{i}))      
    k = 1;
    annot = annotation.Children(6).Children;
    for j = 1:length(annot)
        if(~isstruct(annot(j).Children))
            continue;
        else
            label = annot(j).Children(4).Children.Data;
            start_time = annot(j).Children(6).Children.Data;
            duration = annot(j).Children(8).Children.Data;
            if(contains(label, class))
                sleepstages(k).label = label;
                sleepstages(k).start_time = str2num(start_time);
                sleepstages(k).duration = str2num(duration);
                k = k+1;
            end
        end
    end
    for j = 1:length(sleepstages)
        if (sleepstages(j).label == "Wake|0")
            sleepstages(j).label = 0;
        elseif (sleepstages(j).label == "Stage 1 sleep|1")
            sleepstages(j).label = 2;
        elseif (sleepstages(j).label == "Stage 2 sleep|2")
            sleepstages(j).label = 3;
        elseif (sleepstages(j).label == "Stage 3 sleep|3")
            sleepstages(j).label = 4;
        elseif (sleepstages(j).label == "Stage 4 sleep|4")
            sleepstages(j).label = 4;
        elseif (sleepstages(j).label == "REM sleep|5")
            sleepstages(j).label = 1;
        else
            sleepstages(j).label = 5;
        end
    end
    sleep_h(i).annotation = sleepstages;

% Bandpass filter
    disp(string(i+starting_point-1)+' / '+string(length(target_files))+') Preprocessing... '+string(psg_n{i}))
    psg = [];
    EEG = [];
    EEG = pop_biosig(cat(2, char(data_dir), psg_n{i}),'channels',8, 'importevent', 'off');
    psg(1,:) = EEG.data;
    psg = psg(:, 1:(sleep_h(i).annotation(end).start_time + sleep_h(i).annotation(end).duration) * EEG.srate);
    total_len= (sleep_h(i).annotation(end).start_time + sleep_h(i).annotation(end).duration);
    psg = single(permute(reshape(psg, 1, 30*EEG.srate, total_len/30), [3, 1, 2]));
% Segmenting hyp
    clear hyp;
    for j = 1:length(sleep_h(i).annotation)
        start_epo = sleep_h(i).annotation(j).start_time / 30;
        end_epo = start_epo + (sleep_h(i).annotation(j).duration / 30);
        hyp([start_epo+1:end_epo]) = sleep_h(i).annotation(j).label;
    end
% Removing movement / ? samples
    if find(hyp == 5) 
        psg(find(hyp == 5), :, :) = [];
        hyp(find(hyp == 5)) = [];
    end
    

    long_stages{i} = [];
    
    s_stages = find(hyp ~= 0);
    k = 1;
    for j = 1:(length(s_stages) - 1)
        if s_stages(j + 1) - s_stages(j) > 90 
            long_stages{i}(k) = j;
            k = k + 1;
        end
    end
    if ~isempty(long_stages{i})
        a = s_stages(long_stages{i} + 1) - 1;
        st_end(i, 1) = a(1);
    else
        st_end(i, 1) = s_stages(1) - 1;
    end
    
    st_end(i, 2) = s_stages(end) + 1;
    
   if (st_end(i, 1) - 60 <= 0)
        st = 1;
    else
        st = st_end(i, 1) - 60;
    end
    
    if (st_end(i, 2) + 60 > length(hyp))
        st_ed = length(hyp);
    else
        st_ed = st_end(i, 2) + 60;
    end
    
    psg = psg(st: st_ed, :, :);
    hyp = hyp(st: st_ed);
    
    long_stages{i} = [];
    disp(string(i+starting_point-1)+' / '+string(length(target_files))+') Saving preprocessed '+string(psg_n{i})+'...')
    save(cat(2, psg_path, char(psg_n{i}(1:end - 4) + ".mat")), 'psg', '-v7')
    save(cat(2, hyp_path, char(hyp_n{i}(1:end - 4) + ".mat")), 'hyp', '-v7')
    disp('Saved... '+string(length(target_files)-i-starting_point+1)+' files left')
end