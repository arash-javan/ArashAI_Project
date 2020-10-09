function [TestResult,TrainResult,ExtTestResult]=Lolimot_Fun(X,Y,X_ext,Y_ext,No_of_folds,nn)
addpath('C:\Users\ajavanmardi\Desktop\ArashAI_Project_new\Classification_task')

No_of_class=max(Y);
InputNum=size(X,2);
data=[X Y];
MAXY=max(Y);


[test_data,train_data] = KFoldCrossValidation(data,No_of_folds);
for K =1 : No_of_folds
    
    Train_Validedata=train_data(K);
    Train_Validedatase=cell2mat(Train_Validedata);
    [r c]=size(Train_Validedatase);
    numTrain=ceil(90*r/100);
    Xtrn=Train_Validedatase(1:numTrain,1:end-1);
    Ytrn=Train_Validedatase(1:numTrain,end);
    Xvld= Train_Validedatase(numTrain:end,1:end-1);
    yvld=Train_Validedatase(numTrain:end,end);
    
    test_datatest=cell2mat(test_data(K));
    Xtst= test_datatest(:,1:end-1);
    Ytst=test_datatest(:,end);
    
    
   ctr=size(Xtrn,2);
    
    for l=1:ctr
        if std( Xtrn(:,l))==0
            Xtrn(1,l)=Xtrn(1,l)+(1);
        end
    end
    
    for l=1:ctr
        if std( Xvld(:,l))==0
            Xvld(1,l)=Xvld(1,l)+(1);
        end
    end
    
    cte=size(Xtst,2);
    
    
    for l=1:cte
        if std( Xtst(:,l))==0
            Xtst(1,l)=Xtst(1,l)+(1);
        end
    end




    [C I LB M UB V W] = FLoLiMoT(Xtrn,Ytrn,Xvld,yvld,nn);
    yhtrn=netFeed(C,V,W,Xtrn);
    
    % test step
    Yh1 = netFeed(C,V,W,Xtst);
    Yh1_ext = netFeed(C,V,W,X_ext);

    
    %%%%%%%%TEST Measurments  MSE, SMSE, MAE, r-squared, adjusted r-squared
    mse_train = mse(Ytrn,yhtrn);
    rmse_train = sqrt(mse(Ytrn,yhtrn));
    mae_train = mae(Ytrn,yhtrn);
    mse_test = mse(Ytst,Yh1);
    rmse_test = sqrt(mse(Ytst,Yh1));
    mae_test = mae(Ytst,Yh1);
    
    mse_ext_test = mse(Y_ext,Yh1_ext);
    rmse_ext_test = sqrt(mse(Y_ext,Yh1_ext));
    mae_ext_test = mae(Y_ext,Yh1_ext);
    %%%%%%%%TRAIN Measurments
    MSEtrain(K,1) = mse_train;
    RMSEtrain(K,1) = rmse_train;
    MAEtrain(K,1) = mae_train;
    %%%%%%%%TEST Measurments
    MSEtest(K,1) = mse_test;
    RMSEtest(K,1) = rmse_test;
    MAEtest(K,1) = mae_test;
    
    
    %%%%%%%%External TEST Measurments
    MSEexttest(K,1) = mse_ext_test;
    RMSEexttest(K,1) = rmse_ext_test;
    MAEexttest(K,1) = mae_ext_test;
end%%%For for Kfold
%
TrainResult.MSE= MSEtrain;
TrainResult.RMSE= RMSEtrain;
TrainResult.MAE= MAEtrain;
%
TestResult.MSE= MSEtest;
TestResult.RMSE= RMSEtest;
TestResult.MAE= MAEtest;

ExtTestResult.MSE= MSEexttest;
ExtTestResult.RMSE= RMSEexttest;
ExtTestResult.MAE= MAEexttest;

end %function

