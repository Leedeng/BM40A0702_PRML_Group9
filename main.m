clc;clear;close all;
%% This is a example of digit_classify

% define classes
classes = {'0','1','2','3','4','5','6','7','8','9'};
% import pretrained model
net = importKerasNetwork('Pretrained/model_CL.h5',Classes=classes);

% for a single sample
test_data = load('dataset/digits_3d_raw/validation_data/stroke_1_0045');
digit_classify(test_data,net,classes)

% for multiple samples
% dataset_dir = 'dataset/digits_3d_raw/validation_data/';
% files = dir(strcat(dataset_dir,'*.mat'));
% predict = [];
% gt = [];
% for i = 1:length(files)
%     name = files(i).name;
%     class_label = strsplit(name,'_');
%     gt(end+1) = str2double(class_label{1,2});
%     file_dir = append(dataset_dir, name);
%     test_data = load(file_dir);
%     pred = digit_classify(test_data,net,classes);
%     predict(end+1) = pred;
% end
% num_correct = sum(predict==gt);
% fprintf('Accuracy: %.2f %%', 100*(num_correct/length(files)))
