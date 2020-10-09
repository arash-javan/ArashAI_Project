

load 'MRI_INPUT_OUTPUT.mat';

            
            X=data(:,1:end-1);
            Y=data(:,end);


                            [TestResult,TrainResult]=Decision_TreeClassifiyer(X,Y,5,1,1,1);
                            Table_reliability_Test=mean(TestResult.Accurtest);
                            Table_reliability_Train=mean(TrainResult.Accurtrain);