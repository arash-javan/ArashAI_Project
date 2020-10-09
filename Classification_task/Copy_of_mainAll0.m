clc;
clear;
close all;
NUM_File=5;
for i=1:1 %%%%NUM_File
    q=i-1;
    distype=1; %%%Euclidean distance=1,Pearson correlation=2};
    Num_Evalua=1;
    Num_Clus_Met_limit=13;
    Num_Clus_Met=8;
    Num_of_Fe_Redu=16;
    neighbour=15;
    NumberOfClassifications=19;
    No_of_folds=5;
    %     Num_features=ceil(0.15*size(Xini,1));
    Table_reliability_Test=zeros((NumberOfClassifications+1)*Num_of_Fe_Redu,Num_Clus_Met);
    Table_reliability_Train=zeros((NumberOfClassifications+1)*Num_of_Fe_Redu,Num_Clus_Met);
    e=0;
    for p=8:8%:Num_of_Fe_Redu
        for oo=12:Num_Clus_Met_limit
            o=oo-5;
            
            dirname=['Clusteringwithimage_year' num2str(q)];
            ffname=strcat(dirname,'/table_ResultAll1.mat');
            load(ffname);
            Final_Cluster=table(p,end);
            
            dirname=['Inputdata_Y' num2str(q),'/SiseReduction'...
                num2str(p) ,'/Cluster' num2str(Final_Cluster),'/ClusteringA' num2str(oo)];
            ffname=strcat(dirname,'/Input.mat');
            
            load(ffname);
            DataNum = size(input,1);
            selectData=randperm(DataNum);
            
            data=input(selectData,:);
            
            data=data(:,2:end);
            
            X=data(:,1:end-1);
            Y=data(:,end);
            
            for d=1:NumberOfClassifications
                dd=e+d;
                try
                    
                    switch d
                        case 1
                            [TestResult,TrainResult]=Decision_TreeClassifiyer(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 2
                            [TestResult,TrainResult]=MultiLabel_SVM(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 3
                            [TestResult,TrainResult]=Naive_Bayes(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 4
                            [TestResult,TrainResult]=SetKNN(X,Y,No_of_folds,q,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 5
                            [TestResult,TrainResult]=Ensemble(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 6
                            [TestResult,TrainResult]=ClassificationDiscriminant_class(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 7
                            [TestResult,TrainResult]=NEWPPN(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 8
                            [TestResult,TrainResult]=ClassificationECOC_class(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 9
                            [TestResult,TrainResult]=MLP(X,Y,No_of_folds,i,p,o)
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 10
                            [TestResult,TrainResult]=SetRF(X,Y,No_of_folds,q,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 11
                            [TestResult,TrainResult]=RNN(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 12
                            [TestResult,TrainResult]= SetRBF(X,Y,No_of_folds,q,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 13
                            [TestResult,TrainResult]=SetLolimat(X,Y,No_of_folds,q,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 14
                            [TestResult,TrainResult]=GaussianMLClassifier3(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 15
                            [TestResult,TrainResult]=GaussianMLClassifier2(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 16
                            [TestResult,TrainResult]=GaussianMLClassifier1(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                       % case 17
                            % [TestResult,TrainResult]=SetANFISS(X(:,2:5),Y,i,No_of_folds,q,p,o);
                            % Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            %Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            %
                        case 17
                            [TestResult,TrainResult]=Fit_linear_regression_model(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 18
                            [TestResult,TrainResult]=Gaussian_Reg(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                        case 19
                            [TestResult,TrainResult]=SET_Fit_linear_regression_model(X,Y,No_of_folds,i,p,o);
                            Table_reliability_Test(dd,o)=mean(TestResult.Accurtest);
                            Table_reliability_Train(dd,o)=mean(TrainResult.Accurtrain);
                            
                    end  %%%for switch
                catch MEError
                    mkdir('Errors');
                    errfilename=['Errors/' 'Error_'  num2str(q) '_' num2str(p) '_' num2str(o) '_' num2str(d) '_' num2str(rand) '.mat'];
                    save(errfilename);
                end
                if d==NumberOfClassifications
                    dd=dd+1;
                    
                    
                    Table_reliability_Test(dd,o)=table(p,end);
                    Table_reliability_Train(dd,o)=table(p,end);
                    
                    dirname=['Reliability_results_SIZE8_CLUSTER7u8' num2str(i)];
                    mkdir(dirname);
                    ffname=strcat(dirname,'/Table_reliability_Train.mat');
                    save(ffname,'Table_reliability_Train');
                    ffname1=strcat(dirname,'/Table_reliability_Test.mat');
                    save(ffname1,'Table_reliability_Test');
                end
            end%%%for FOR classification
        end %%%for clusteringAL
        
        e=dd;
        
    end %%%for sizeReduction
    
    
    clc;
    clear;
    close all;
    
end

