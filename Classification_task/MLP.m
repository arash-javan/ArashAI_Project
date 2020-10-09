function [TestResult,TrainResult]=MLP(X,Y,No_of_folds,i,p,o)

% %%%%% mean
% metr=mean(Xtrn);
% meva=mean(Xvld);
% mete=mean(Xtst);

% No_of_folds = 5;
No_of_class=max(Y);
InputNum=size(X,2);

data=[X Y];
MAXY=max(Y);

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
    Xtrn=Train_Validedatase(:,1:end-1)';
    Ytrnini=Train_Validedatase(:,end)';
    
    test_datatest=cell2mat(test_data(K));
    Xtst= test_datatest(:,1:end-1)';
    Ytestini=test_datatest(:,end)';
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
    
    
    
    for pp=1:MAXY
        bb=size(Ytrnini,2);
        for kk=1:bb
            if Ytrnini(1,kk)==pp
                Ytrn(pp,kk)=1;
            else
                Ytrn(pp,kk)=0;
            end
        end
    end
    
    
    for pp=1:MAXY
        bb=size(Ytestini,2);
        for kk=1:bb
            if Ytestini(1,kk)==pp
                Ytest(pp,kk)=1;
            else
                Ytest(pp,kk)=0;
            end
        end
    end
    
    %  trainlm for regression
    %  trainscg for classification and unit column vector targets
    %  trainrp for huge datasets
    
    performFcn = 'crossentropy';  % Cross-Entropy
    
    trainFcn = 'trainscg';% Scaled conjugate gradient backpropagation.
    
    nna=ceil(InputNum+1);
    nnb=ceil(((3/2)*nna));
    
    nnc=ceil(((1/3)*nnb));
    nnd=ceil(((1/3)*nnc));
    nne=ceil(((1/3)*nnd));
    
    
    net3 = patternnet([nna nnb nnc nnd nne],trainFcn,performFcn);
    net3.divideParam.valRatio = 5/100;
    net3.trainParam.epochs=5000;
    net3.trainParam.max_fail=40;
    net3.trainParam.sigma=5.0e-5;
    net3.trainParam.lambda=5.0e-7;
    net3.trainParam.showWindow = false;
    net3.trainParam.showCommandLine = false
    
    net3 = train(net3,Xtrn,Ytrn);
    
    %     view(net3)
    
    yhtrn1 = net3(Xtrn);
    
    yhtrn = vec2ind(yhtrn1);
    
    %%%%%%%%TRAIN Measurments  Accuracy, Specifisity, Sensivitivity, f_SCORE,.....
    statsTrain = confusionmatStats(Ytrnini,yhtrn);
    
    Accurtrain(K,1) = statsTrain.accuracy;
    % Sensittrain(:,K) = statsTrain.sensitivity;
    % Specitrain(:,K) = statsTrain.specificity;
    % Fscoretrain(:,K) = statsTrain.Fscore;
    % Pesrcitrain(:,K) = statsTrain.precision;
    % Recalltrain(:,K) = statsTrain.recall;
    
    
    % test step
    Yh1 = net3(Xtst);
    Yh = vec2ind(Yh1);
    
    %%%%%%%%TEST Measurments  Accuracy, Specifisity, Sensivitivity, f_SCORE,.....
    statsTest = confusionmatStats(Ytestini,Yh);
    
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

