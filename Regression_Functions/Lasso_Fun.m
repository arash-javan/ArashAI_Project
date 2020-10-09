function [TestResult,TrainResult]= Lasso_Fun(x,y,No_of_folds)

data_arash=[x y];

[test_data,train_data] = KFoldCrossValidation(data_arash,No_of_folds);
for K =1 : No_of_folds
    Train_Validedata=train_data(K);
    Train_Validedatase=cell2mat(Train_Validedata);
    Xtrn=Train_Validedatase(:,1:end-1);
    Ytrn=Train_Validedatase(:,end);
    test_datatest=cell2mat(test_data(K));
    Xtst= test_datatest(:,1:end-1);
    Ytst=test_datatest(:,end);
    ctr=size(Xtrn,2);
    %adds 2 if standard deviation is zero?
    for l=1:ctr
        if std( Xtrn(:,l))==0
            Xtrn(1,l)=Xtrn(1,l)+( 2);
        end
    end
    cte=size(Xtst,2);
    for l=1:cte
        if std( Xtst(:,l))==0
            Xtst(1,l)=Xtst(1,l)+(2);
        end
    end
    
    
    
    [B,FitInfo] = lasso(Xtrn,Ytrn,'CV',10);
    %lassoPlot(B,FitInfo,'PlotType','CV'); % Look at MSE vs Lambda
   % coeff = B(:,FitInfo.IndexMinMSE); % Regression coefficients for min MSE, use these coefficients.
   % yhtrn = Xtrn * coeff; % Predicted response
 
    yhtrn = predict(mdl,Xtrn);
    
    mse_train = mse(Ytrn,yhtrn);
    rmse_train = sqrt(mse(Ytrn,yhtrn));
    mae_train = mae(Ytrn,yhtrn);
   
    % test step
    Yh1 = Xtst * coeff;
    
    mse_test = mse(Ytst,Yh1);
    rmse_test = sqrt(mse(Ytst,Yh1));
    mae_test = mae(Ytst,Yh1);
    %%%%%%%%TRAIN Measurments
    MSEtrain(K,1) = mse_train;
    RMSEtrain(K,1) = rmse_train;
    MAEtrain(K,1) = mae_train;
    %%%%%%%%TEST Measurments
    MSEtest(K,1) = mse_test;
    RMSEtest(K,1) = rmse_test;
    MAEtest(K,1) = mae_test;
%     
%     r2 = mdl.Rsquared.Ordinary;
%     r2a = mdl.Rsquared.Adjusted;
end%%%For for Kfold
% 
 TrainResult.MSE= MSEtrain;
 TrainResult.RMSE= RMSEtrain;
 TrainResult.MAE= MAEtrain;
% 
 TestResult.MSE= MSEtest;
 TestResult.RMSE= RMSEtest;
 TestResult.MAE= MAEtest;

end %function 