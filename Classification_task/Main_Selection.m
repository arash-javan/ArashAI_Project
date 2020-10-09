clc;
clear;
close all;
    No_of_folds=5;
NUM_File=5;
for i=1:NUM_File
    q=i-1;
   
    dirnamehh=['Inputdata_fiveyears_With_Dat' ];
    mkdir(dirnamehh);
    ffname=strcat(dirnamehh,'/Input_data_Year_',num2str(q), '.mat');
    load(ffname);
    
    DataNum = size(input,1);
    selectData=randperm(DataNum);
    
    data=input(selectData,:);
    
    data=data(:,2:end);
    
    X=data(:,1:end-1);
    Y=data(:,end);
   
    [TestResult,TrainResult]=ClassificationECOC_class(X,Y,No_of_folds,1,1,5);
    Decision(1,i)=mean(TestResult.Accurtest);
    
 
    
end

