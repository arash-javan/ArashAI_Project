function [TestResult,TrainResult]=SetLolimat(X,Y,No_of_folds,q,p,o)

%Q-Learning Algorithm in order to find the best configuration

%  mkdir('ResultLOLIMOT');
%  dirname=strcat(ResultLOLIMOT,'/Setting.mat');
dirname=['SetParameters'];
mkdir(dirname);
ffname=strcat(dirname,'/SettingLOLIMOT.mat');

setting=[];
if exist(ffname,'file')==0
    MinError=1000.0;
    LastBestNN=6;
    NN_Temp=13;
    LastChange=7;
    Coeff=1;
    for count=1:5
        close all;
        Save=0;
        [TestResult,TrainResult]=loli(X,Y,No_of_folds,q,p,o,NN_Temp,Save);
%   AbsErr_Temp=cell2mat(TestResult.Accurtest);
        AbsErr_Temp=mean(TestResult.Accurtest)
        LastChange=ceil(LastChange/1.3)
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

[TestResult,TrainResult]=loli(X,Y,No_of_folds,q,p,o,setting.RightNumberOfNeurons,1)
save(ffname,'setting');

end