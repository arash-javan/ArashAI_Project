close all;
clear all;
clc;

%% Initializing parameters and data required.
numRight=0;
wrong=0;

load MOCA1

X =data(:,3:end-1);
Y = data(:,end);

[R C]=size(data);

% produce  Train data 
numTrain=ceil(80*R/100);
XTrainData=X(1:numTrain,:);
YTrainData=Y(1:numTrain,:);
 
% produce  Test data
XTestData=X(numTrain:end,:); 
YTestData=Y(numTrain:end,:);
nytst=size(YTestData,1);

%% Make LibSVM
libsvm_options='t 1';
model = svmtrain(YTrainData, XTrainData,libsvm_options );
[predicted_label,accuracy,prob_estimates] = svmpredict(YTestData, XTestData, model,libsvm_options)

%Calculate Accuracy
Accuracy=accuracy;

