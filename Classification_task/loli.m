function [TestResult,TrainResult]=loli(X,Y,No_of_folds,i,p,o,nn,SaveResults)

nn=round(nn);

if nn==0 || nn<0
    nn=2;
end
No_of_class=max(Y);
InputNum=size(X,2);

data=[X Y];

MAXY=max(Y);

Accurtrainnew= zeros(No_of_folds,1);
Accurtrain= zeros(No_of_folds,1);
Sensittrain = zeros(No_of_class,No_of_folds);
Specitrain = zeros(No_of_class,No_of_folds);
Fscoretrain = zeros(No_of_class,No_of_folds);
Pesrcitrain =zeros(No_of_class,No_of_folds);
Recalltrain =zeros(No_of_class,No_of_folds);

Accurtestnew = zeros(No_of_folds,1);
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
    
    % [x,y,z]=size(train_data);
    
    
    Train_Validedata=train_data(K);
    Train_Validedatase=cell2mat(Train_Validedata);
    [r c]=size(Train_Validedatase);
    numTrain=ceil(95*r/100);
    Xtrn=Train_Validedatase(1:numTrain,1:end-1);
    Ytrn=Train_Validedatase(1:numTrain,end);
    Xvld= Train_Validedatase(numTrain:end,1:end-1);
    yvld=Train_Validedatase(numTrain:end,end);
    
    test_datatest=cell2mat(test_data(K));
    Xtst= test_datatest(:,1:end-1);
    Ytest=test_datatest(:,end);
    
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
    
%     for pp=1:MAXY
%         bb=size(Ytrnini,1);
%         for kk=1:bb
%             if Ytrnini(kk,1)==pp
%                 Ytrn(kk,pp)=1;
%             else
%                 Ytrn(kk,pp)=0;
%             end
%         end
%     end
    
%     for pp=1:MAXY
%         bb=size(yvldini,1);
%         for kk=1:bb
%             if yvldini(kk,1)==pp
%                 yvld(kk,pp)=1;
%             else
%                 yvld(kk,pp)=0;
%             end
%         end
%     end
    
    
%     for pp=1:MAXY
%         bb=size(Ytestini,1);
%         for kk=1:bb
%             if Ytestini(kk,1)==pp
%                 Ytest(kk,pp)=1;
%             else
%                 Ytest(kk,pp)=0;
%             end
%         end
%     end
    
    
    [C I LB M UB V W] = FLoLiMoT(Xtrn, Ytrn, Xvld, yvld,nn);
    yhtrn=netFeed(C ,V, W ,Xtrn);
    
% yhtrn = vec2ind(yhtrn1);
    
    
    %%%%%%%%TRAIN Measurments  Accuracy, Specifisity, Sensivitivity, f_SCORE,.....
    statsTrain = confusionmatStats(Ytrn,round(yhtrn));
%     resultstestnew=evaluation(Ytrn,round(yhtrn));
    
    Accurtrain(K,1) = statsTrain.accuracy;
    
%     % Sensittrain(:,K) = statsTrain.sensitivity;
    % Specitrain(:,K) = statsTrain.specificity;
    % Fscoretrain(:,K) = statsTrain.Fscore;
    % Pesrcitrain(:,K) = statsTrain.precision;
    % Recalltrain(:,K) = statsTrain.recall;
    
    % test step
    Yh = netFeed(C ,V, W ,Xtst);
    
%   Yh1 = vec2ind(Yh);
    
    %%%%%%%%TEST Measurments  Accuracy, Specifisity, Sensivitivity, f_SCORE,.....
    statsTest = confusionmatStats(Ytest,round(Yh));
    
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

