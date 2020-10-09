function  [TestResult,TrainResult]=SetANFISS(X,Y,i,No_of_folds,q,p,o)

%Q-Learning Algorithm in order to find the best configuration


dirname=['ResultANFISS/Setting'];
mkdir(dirname);

filename_setting=strcat(dirname,'/seting_',num2str(q), num2str(p), num2str(o),'.mat');
%save(filename);
setting=[];
if exist(filename_setting,'file')==0
    MinError=1000.0;
    LastBestNN=6;
    NN_Temp=LastBestNN;
    LastChange=LastBestNN;
    Coeff=1;
    for count=1:5
        close all;
        [TestResult,TrainResult]=ANFIS_fun(X,Y,No_of_folds,q,p,o,NN_Temp,0);
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

saveresult=1;
[TestResult,TrainResult]=ANFIS_fun(X,Y,No_of_folds,q,p,o,setting.RightNumberOfNeurons,saveresult);
save(filename_setting,'setting');

end