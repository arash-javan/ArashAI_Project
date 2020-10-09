function [TestResult,TrainResult]=RBF(X,Y,No_of_folds,i,p,o,nn,SaveResults)


if nn==0 || nn<0
    nn=2;
end
nn=round(nn);
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
    
    %  train step
    if nn==0
        nn = round((InputNum/2.2)+2);
    end
    
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
    
    net = newrb(Xtrn,Ytrn,0,1,nn,1);
    %output = sim(net,TrainData);
    yhtrn1 = sim(net,Xtrn);
     yhtrn = vec2ind(yhtrn1);
    %%%%%%%%TRAIN Measurments  Accuracy, Specifisity, Sensivitivity, f_SCORE,.....
    statsTrain = confusionmatStats(Ytrnini,round(yhtrn));
    
    Accurtrain(K,1) = statsTrain.accuracy;
    % Sensittrain(:,K) = statsTrain.sensitivity;
    % Specitrain(:,K) = statsTrain.specificity;
    % Fscoretrain(:,K) = statsTrain.Fscore;
    % Pesrcitrain(:,K) = statsTrain.precision;
    % Recalltrain(:,K) = statsTrain.recall;
    
    
    % test step
    Yh1 = sim(net,Xtst);
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

