function [TestResult,TrainResult]=SET_Fit_linear_regression_model(X,Y,No_of_folds,q,p,o)

%Q-Learning Algorithm in order to find the best configuration


dirname=['SetParameters'];
mkdir(dirname);
filename_setting=strcat(dirname,'/SetFitLinear.mat');

setting=[];
if exist(filename_setting,'file')==0
    MinError=1000.0;
    LastBestNN=20;
    NN_Temp=LastBestNN;
    LastChange=LastBestNN;
    Coeff=1;
    for count=1:15
        close all;
        [TestResult,TrainResult]=Fit_linear_regression_model(X,Y,No_of_folds,q,p,o,NN_Temp,0);
%   AbsErr_Temp=cell2mat(TestResult.Accurtest);
        AbsErr_Temp=mean(TestResult.Accurtest)
        LastChange=ceil(LastChange/1.2)
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
    load(filename_setting);
end %if

[TestResult,TrainResult]=Fit_linear_regression_model(X,Y,No_of_folds,q,p,o,setting.RightNumberOfNeurons,1)
save(filename_setting,'setting');

end