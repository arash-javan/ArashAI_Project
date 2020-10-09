function [TestResult,TrainResult]=SetRF(X,Y,No_of_folds,q,p,o)

%Q-Learning Algorithm in order to find the best configuration



dirname=['SetParameters'];
mkdir(dirname);
ffname=strcat(dirname,'/SettingRF.mat');

% filename_setting=(dirname 'seting.mat');
%save(filename);
setting=[];
if exist(ffname,'file')==0
    MinError=1000.0;
    LastBestNN=4;
    NN_Temp=LastBestNN;
    LastChange=LastBestNN;
    Coeff=1;
   for count=1:3
        close all;
        [TestResult,TrainResult]=RandomForest(X,Y,No_of_folds,i,p,o,NN_Temp,0)
%   AbsErr_Temp=cell2mat(TestResult.Accurtest);
        AbsErr_Temp=mean(TestResult.Accurtest);
        LastChange=ceil(LastChange/2);
        if AbsErr_Temp<MinError
            
            LastBestNN=NN_Temp;
            MinError=AbsErr_Temp;
        else 
            Coeff=(-1)*Coeff;
        end %if
        
            NN_Temp=LastBestNN+(LastChange*Coeff);

    end %for
    
    setting.RightNumberOfNeurons=LastBestNN;

else
    load(ffname);
end %if

SaveResults=1;

[TestResult,TrainResult]=RandomForest(X,Y,No_of_folds,i,p,o,setting.RightNumberOfNeurons,SaveResults);
save(ffname,'setting');
end