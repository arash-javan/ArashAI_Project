function [TestResult,TrainResult,ExtTestResult]= regression_algos(X,Y,X_ext,Y_ext,No_of_folds,i,nn,Max_output)
addpath('C:\Users\Arash\Downloads\ArashAI_Project_oct8\Regression_functions')

switch i
    
    case 1
        [TestResult,TrainResult,ExtTestResult]=LinearRegression_Fun(X,Y,X_ext,Y_ext,No_of_folds,Max_output);
    case 2
        [TestResult,TrainResult,ExtTestResult]= GaussianProcess_Fun(X,Y,X_ext,Y_ext,No_of_folds,Max_output);
    case 3
        [TestResult,TrainResult,ExtTestResult]= LogisticRegression_Fun(X,Y,X_ext,Y_ext,No_of_folds,Max_output);
        %     case 4
        %         [TestResult,TrainResult]= MLP_trainlm_mse_Fun(X,Y,No_of_folds);
    case 4
        [TestResult,TrainResult,ExtTestResult]= MLP_trainscg_mse_Fun(X,Y,X_ext,Y_ext,No_of_folds,Max_output);
        %     case 6
        %         [TestResult,TrainResult]= MLP_trainscg_Fun(X,Y,No_of_folds);
    case 5
        [TestResult,TrainResult,ExtTestResult]= RNN_trainlm_Fun(X,Y,X_ext,Y_ext,No_of_folds,Max_output);
        %     case 8
        %         [TestResult,TrainResult]= RNN_trainscg_Fun(X,Y,No_of_folds);
        %     case 9
        %         [TestResult,TrainResult]= ANFIS_Fun(X,Y,No_of_folds,nn);
    case 6
        [TestResult,TrainResult,ExtTestResult]= RBF_Fun(X,Y,X_ext,Y_ext,No_of_folds,Max_output);
    case 7
        [TestResult,TrainResult,ExtTestResult]= Lolimot_Fun(X,Y,X_ext,Y_ext,No_of_folds,nn,Max_output);
end


