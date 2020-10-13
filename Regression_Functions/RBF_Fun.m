function [TestResult,TrainResult,ExtTestResult]=RBF_Fun(X,Y,X_ext,Y_ext,No_of_folds,Max_output)
No_of_class=max(Y);
InputNum=size(X,2);

data=[X Y];
MAXY=max(Y);
[test_data,train_data] = KFoldCrossValidation(data,No_of_folds);
for K =1 : No_of_folds
    
    Train_Validedata=train_data(K);
    Train_Validedatase=cell2mat(Train_Validedata);
    Xtrn=Train_Validedatase(:,1:end-1)';
    Ytrn=Train_Validedatase(:,end)';
    
    test_datatest=cell2mat(test_data(K));
    Xtst= test_datatest(:,1:end-1)';
    Ytst=test_datatest(:,end)';
    ctr=size(Xtrn,2);
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
    
    
    % Ytrn=zeros(MAXY,size(Ytrnini,2));
    
    net3 = newrb(Xtrn,Ytrn);
    %view(net3)
    
    yhtrn1 = sim(net3,Xtrn);
    yhtrn = vec2ind(yhtrn1);
    Yh = sim(net3,Xtst);
    Yh1 = vec2ind(Yh);
    Yh_ext = sim(net3,X_ext');
    Yh1_ext = vec2ind(Yh_ext);
    
    Ytrn_real = Ytrn*Max_output;
    yhtrn_real = yhtrn*Max_output;
    Ytst_real = Ytst*Max_output;
    Yh1_real = Yh1*Max_output;
    Yext_real = Y_ext*Max_output;
    Yh1_ext_real = Yh1_ext*Max_output;
    
    %%%%%%%%TEST Measurments  MSE, SMSE, MAE, r-squared, adjusted r-squared
    mse_train = mse(Ytrn_real,yhtrn_real);
    rmse_train = sqrt(Ytrn_real,yhtrn_real);
    mae_train = mae(Ytrn_real,yhtrn_real);
    
    mse_test = mse(Ytst_real,Yh1_real);
    rmse_test = sqrt(mse(Ytst_real,Yh1_real));
    mae_test = mae(Ytst_real,Yh1_real);
    
    mse_ext_test = mse(Yext_real,Yh1_ext_real);
    rmse_ext_test = sqrt(mse(Yext_real,Yh1_ext_real));
    mae_ext_test = mae(Yext_real,Yh1_ext_real);
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