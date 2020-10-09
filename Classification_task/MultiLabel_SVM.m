function [TestResult,TrainResult]=LIb_SVM(X,Y,No_of_folds,i,p,o)
%%%%%%%%%%%https://github.com/iamaureen/Multiclass-Classification-using-SVM
% %%%%% mean
% metr=mean(Xtrn);
% meva=mean(Xvld);
% mete=mean(Xtst);
% load 'Input';
% X=input(:,2:end-1);
% Y=input(:,end);
No_of_class = max(Y);
% No_of_folds = 5;
InputNum=size(X,2);

data=[X Y];


Accurtrain= zeros(No_of_folds,1);
Sensittrain = zeros(No_of_class,No_of_folds);
Specitrain = zeros(No_of_class,No_of_folds);
Fscoretrain = zeros(No_of_class,No_of_folds);
Pesrcitrain =zeros(No_of_class,No_of_folds);
Recalltrain =zeros(No_of_class,No_of_folds);

Accurtest = zeros(No_of_folds,1);
Sensittest =zeros(No_of_class,No_of_folds);
Specitest = zeros(No_of_class,No_of_folds);
Fscoretest = zeros(No_of_class,No_of_folds);
Pesrcitest =zeros(No_of_class,No_of_folds);
Recalltest = zeros(No_of_class,No_of_folds);

[test_data,train_data] = KFoldCrossValidation(data,No_of_folds);
for K =1 : No_of_folds
    
 
    Train_Validedata=train_data(K);
    Train_Validedatase=cell2mat(Train_Validedata);
    Xtrn=Train_Validedatase(:,1:end-1);
    Ytrn=Train_Validedatase(:,end);
    
    test_datatest=cell2mat(test_data(K));
    Xtst= test_datatest(:,1:end-1);
    Ytest=test_datatest(:,end);
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

 %% Train Decision Tree Classifier
%% Make LibSVM

%transposing the class label vectors
y_train_transpose =Ytrn';
y_test_transpose = Ytest';
% Xtst=Xtst';
% Xtrn=Xtrn';

Numtest=size(Ytest,1);
Numtrn=size(Ytrn,1);

%initialization
%number of class-6
%number of test samples-2947
SVMModel = cell(No_of_class,1);
labeltest = zeros(No_of_class,Numtest);
labeltrn = zeros(No_of_class,Numtrn);

%1 in the place of index, other class 0
trainingClassLabelsMatrix = full(ind2vec(y_train_transpose,No_of_class));

%train the model one-vs-all
for index=1:No_of_class
    SVMModel{index} = fitcsvm(Xtrn,trainingClassLabelsMatrix(index,:),'KernelFunction','polynomial','PolynomialOrder',2);
end


%predict values train
for index=1:No_of_class
    labeltrn(index,:) = predict(SVMModel{index},Xtrn);
end

%predict values test
for index=1:No_of_class
    labeltest(index,:) = predict(SVMModel{index},Xtst);
end

%transform into index
predictedLabeltest=vec2ind(labeltest);
predictedLabeltrn=vec2ind(labeltrn);

    %%%%%%%%TRAIN Measurments  Accuracy, Specifisity, Sensivitivity, f_SCORE,.....
    statsTrain = confusionmatStats(Ytrn',predictedLabeltrn);
    
    Accurtrain(K,1) = statsTrain.accuracy;
    % Sensittrain(:,K) = statsTrain.sensitivity;
    % Specitrain(:,K) = statsTrain.specificity;
    % Fscoretrain(:,K) = statsTrain.Fscore;
    % Pesrcitrain(:,K) = statsTrain.precision;
    % Recalltrain(:,K) = statsTrain.recall;
    
    
   
    %%%%%%%%TEST Measurments  Accuracy, Specifisity, Sensivitivity, f_SCORE,.....
    statsTest = confusionmatStats(Ytest',predictedLabeltest);
    
    %%%%%%%%%%%%Filling measurements%%%%%%%%%%%%%%%%%
    Accurtest(K,1) = statsTest.accuracy;
    % Sensittest(:,K) = statsTest.sensitivity;
    % Specitest(:,K) = statsTest.specificity;
    % Fscoretest(:,K) = statsTest.Fscore;
    % Pesrcitest(:,K) = statsTest.precision;
    % Recalltest(:,K) = statsTest.recall;
    
end%%%For for Kfold

TestResult.Accurtest=(Accurtest);
% TrainResult.Sensittest=(Sensittest);
% TrainResult.specificity=(Specitrain);
% TrainResult.Fscore=(Fscoretest);
% TrainResult.precision=(Pesrcitest);
% TrainResult.recall=(Recalltest);


TrainResult.Accurtrain=(Accurtrain);
% TestResult.Sensittest=(Sensittrain);
% TestResult.specificity=(specificitytarin);
% TestResult.Fscore=(Fscoretrain);
% TestResult.precision=(Pesrcitrain);
% TestResult.recall=(Recalltrain);

end %function

