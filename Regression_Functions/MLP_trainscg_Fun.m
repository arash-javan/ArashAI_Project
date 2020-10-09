function [TestResult,TrainResult]=MLP_trainscg_Fun(X,Y,No_of_folds)
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
    
   
    %  trainlm for regression
    %  trainscg for classification and unit column vector targets
    %  trainrp for huge datasets
    
    performFcn = 'crossentropy';  % Cross-Entropy
    %cost function can change to ex. MSE etc (usually use matlab func)
    trainFcn = 'trainscg';% Scaled conjugate gradient backpropagation.
    % 'trainscg' can choose the type of classification such as gradient
    % descent, recurrent neural network...
    
    nna=ceil(InputNum+1);
    nnb=ceil(((3/2)*nna));
    
    nnc=ceil(((1/3)*nnb));
    nnd=ceil(((1/3)*nnc));
    nne=ceil(((1/3)*nnd));
    
    
    net3 = patternnet([nna nnb 3],trainFcn,performFcn);
    net3.divideParam.valRatio = 10/100;
    net3.trainParam.epochs=5000;
    net3.trainParam.max_fail=40;
    net3.trainParam.sigma=5.0e-5;
    net3.trainParam.lambda=5.0e-7;
    net3.trainParam.showWindow = false;
    net3.trainParam.showCommandLine = false;
%     
    net3 = train(net3,Xtrn,Ytrn);
    
    %     view(net3)

    yhtrn = net3(Xtrn);
    
    % test step
    Yh1 = net3(Xtst);

    
    %%%%%%%%TEST Measurments  MSE, SMSE, MAE, r-squared, adjusted r-squared
    mse_train = mse(Ytrn,yhtrn);
    rmse_train = sqrt(mse(Ytrn,yhtrn));
    mae_train = mae(Ytrn,yhtrn);
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
