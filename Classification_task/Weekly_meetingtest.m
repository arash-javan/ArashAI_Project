clc;
clear;
close all;
No_of_folds=5;
i=1;
p=1;
o=1;
  dd=1;         load  MRI_INPUT_OUTPUT;
           
            
            X=data(:,2:end-1);
            Y=data(:,end);
%             
                           [TestResult,TrainResult]=KNN(X,Y,No_of_folds,1,1,1,3,0);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);

% 
%                               [TestResult,TrainResult]=Naive_Bayes(X(:,1:10),Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
%                             
                            
%                             [TestResult,TrainResult]=MultiLabel_SVM(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            

% 
%                             [TestResult,TrainResult]=SetRF(X,Y,No_of_folds,1,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);










%                             [TestResult,TrainResult]=Decision_TreeClassifiyer(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
%                             [TestResult,TrainResult]=Ensemble(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
%                             
%                             [TestResult,TrainResult]=ClassificationDiscriminant_class(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
%                             
%                             [TestResult,TrainResult]=NEWPPN(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
%                             
%                             [TestResult,TrainResult]=ClassificationECOC_class(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
%                             
%                             [TestResult,TrainResult]=MLP(X,Y,No_of_folds,i,p,o)
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
% 
%                             
%                             [TestResult,TrainResult]=RNN(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
%                             
%                             [TestResult,TrainResult]= SetRBF(X,Y,No_of_folds,q,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
%                             
%                             [TestResult,TrainResult]=SetLolimat(X,Y,No_of_folds,q,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
%                             
%                             [TestResult,TrainResult]=GaussianMLClassifier3(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
% 
%                             [TestResult,TrainResult]=GaussianMLClassifier2(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
%                             
% 
%                             [TestResult,TrainResult]=GaussianMLClassifier1(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
%                             
%                        % case 17
%                             % [TestResult,TrainResult]=SetANFISS(X(:,2:5),Y,i,No_of_folds,q,p,o);
%                             % Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             %Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
%                             %
% 
%                             [TestResult,TrainResult]=Fit_linear_regression_model(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
% 
%                             [TestResult,TrainResult]=Gaussian_Reg(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
%                             
% 
%                             [TestResult,TrainResult]=SET_Fit_linear_regression_model(X,Y,No_of_folds,i,p,o);
%                             Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
%                             Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
   


