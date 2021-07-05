%% Clear #1
clear; close all; clc;
%% Path #2
addpath(genpath("./edfread/"))

% data_dir MUST include both PSG and Hypnogram file!
data_dir = "data/path/";

tmp = struct2cell(dir(data_dir + "/*PSG.edf"));
psg_n = tmp(1, :);
tmp = struct2cell(dir(data_dir + "/*Hypnogram.edf"));
hyp_n = tmp(1, :);
class = ["W", "R", "1", "2", "3"];
%% Loading data #3
for i = 1:length(psg_n)
    [sleep_h{i}, ~] = edfread(cat(2, char(data_dir), psg_n{i}));
    [~, annotation] = readEDF(cat(2, char(data_dir), hyp_n{i}));
    for j = 1:length(annotation.annotation.event)
        annotation.annotation.event{j} = annotation.annotation.event{j}(end);
    end
    
    annotation.annotation.event(find(cellfun(@(x) x == 'W', annotation.annotation.event) == 1)) = {0};
    annotation.annotation.event(find(cellfun(@(x) x == 'R', annotation.annotation.event) == 1)) = {1};
    annotation.annotation.event(find(cellfun(@(x) x == '1', annotation.annotation.event) == 1)) = {2};
    annotation.annotation.event(find(cellfun(@(x) x == '2', annotation.annotation.event) == 1)) = {3};
    annotation.annotation.event(find(cellfun(@(x) x == '3', annotation.annotation.event) == 1)) = {4};
    annotation.annotation.event(find(cellfun(@(x) x == '4', annotation.annotation.event) == 1)) = {4};
    annotation.annotation.event(find(cellfun(@(x) x == '?', annotation.annotation.event) == 1)) = {5};
    annotation.annotation.event(find(cellfun(@(x) x == 'e', annotation.annotation.event) == 1)) = {5};
    annotation.annotation.event = cell2mat(annotation.annotation.event);

    sleep_h{i}.annotation = annotation.annotation;
end
%% Pre-processing #4
cd("./eeglab")
eeglab
cd("..")
psg_path = './SleepEDF/psg/';
if(~exist(psg_path, 'dir'))
    mkdir(psg_path);
end
hyp_path = './SleepEDF/hyp/';
if(~exist(hyp_path, 'dir'))
    mkdir(hyp_path);
end
%%
for i = 1:length(psg_n) 
% Bandpass filter
    psg = [];
    EEG = pop_biosig(cat(2, char(data_dir), psg_n{i}), 'channels', 1);
    psg(1, :) = EEG.data;
    psg = single(permute(reshape(psg(:, 1:sleep_h{i}.annotation.starttime(end) * 100), 1, 3000, sleep_h{i}.annotation.starttime(end) / 30), [3, 1, 2]));
  
% Segmenting hyp
    clear hyp;
    for j = 1:length(sleep_h{i}.annotation.event) - 1 
        for k = sleep_h{i}.annotation.starttime(j)/30 + 1:sleep_h{i}.annotation.starttime(j + 1) / 30
            hyp(k) = sleep_h{i}.annotation.event(j);
        end
    end
%%
% Removing movement / ? samples
    if find(hyp == 5) 
        psg(find(hyp == 5), :, :) = [];
        hyp(find(hyp == 5)) = [];
    end
    
    tt{i} = [];
    
    t = find(hyp ~= 0);
    tt_k = 1;
    for ii = 1:(length(t) - 1)
        if t(ii + 1) - t(ii) > 90
            tt{i}(tt_k) = ii;
            tt_k = tt_k + 1;
        end
    end
    if ~isempty(tt{i})
        st_end(i, 1) = t(tt{i} + 1) - 1;
    else
        st_end(i, 1) = t(1) - 1;
    end
    
    st_end(i, 2) = t(end) + 1;
    
    psg = psg(st_end(i, 1) - 60: st_end(i, 2) + 60, :, :);
    hyp = hyp(st_end(i, 1) - 60: st_end(i, 2) + 60);
    
    save(cat(2, psg_path, char(psg_n{i}(1:end - 4) + ".mat")), 'psg', '-v7')
    save(cat(2, hyp_path, char(hyp_n{i}(1:end - 4) + ".mat")), 'hyp', '-v7')
end
