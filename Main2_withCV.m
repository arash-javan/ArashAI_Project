clc;
clear;
close all;
No_of_folds=5;
No_Neighbor=5;
No_of_trees=5;
n=7;
nn=10;
%num_feats = round(numel(X(:,1))*0.15);

%%%%Only Prediction algorithms in each trajectory

data_loc = 'C:\Users\ajavanmardi\Desktop\ArashAI_Project_new\Arash_data_onlyPD.xlsx';
[~,sheets] = xlsfinfo(data_loc);


for i = 1:length(sheets)
    data{i} = xlsread(data_loc,sheets{i});
end

for i = 1:length(data)
    ndata = data{i};
    Max_output = max(ndata(:,end));
    for ar = 1:size(ndata,2)
        norm_data = ndata./max(ndata);
    end
    [test_data,train_data] = KFoldCrossValidation(norm_data,No_of_folds);
    for k =1 : No_of_folds
        Train_Validedata=train_data(k);
        Train_Validedatase=cell2mat(Train_Validedata);
        X=Train_Validedatase(:,3:end-1);
        Y=Train_Validedatase(:,end);
        test_datatest=cell2mat(test_data(k));
        X_ext= test_datatest(:,3:end-1);
        Y_ext=test_datatest(:,end);
        for l = 1:4
            [model, final_x] = feature_selection(X,Y,l);
            models(l) = {model};
            final_X_ext=X_ext(:,model(:,1));
            for j = 1:n
                % try
                [i k l j]
                [TestResult,TrainResult,ExtTestResult]= regression_algos(final_x,Y,final_X_ext,Y_ext,No_of_folds,j,nn,Max_output);
                TOTAL_Test_MAE(k,j,:) = TestResult.MAE;
                TOTAL_Test_MSE(k,j,:) = TestResult.MSE;
                TOTAL_Test_RMSE(k,j,:) = TestResult.RMSE;
                
                TOTAL_Train_MAE(k,j,:) = TrainResult.MAE;
                TOTAL_Train_MSE(k,j,:) = TrainResult.MSE;
                TOTAL_Train_RMSE(k,j,:) = TrainResult.RMSE;
                
                TOTAL_Ext_Test_MAE(k,j,:) = ExtTestResult.MAE;
                TOTAL_Ext_Test_MSE(k,j,:) = ExtTestResult.MSE;
                TOTAL_Ext_Test_RMSE(k,j,:) = ExtTestResult.RMSE;
                
                %catch
                %end
            end
            TOTAL_TestResult.('MAE') = TOTAL_Test_MAE;
            TOTAL_TestResult.('MSE') = TOTAL_Test_MSE;
            TOTAL_TestResult.('RMSE') = TOTAL_Test_RMSE;
            
            TOTAL_TrainResult.('MAE') = TOTAL_Train_MAE;
            TOTAL_TrainResult.('MSE') = TOTAL_Train_MSE;
            TOTAL_TrainResult.('RMSE') = TOTAL_Train_RMSE;
            
            TOTAL_Ext_TestResult.('MAE') = TOTAL_Ext_Test_MAE;
            TOTAL_Ext_TestResult.('MSE') = TOTAL_Ext_Test_MSE;
            TOTAL_Ext_TestResult.('RMSE') = TOTAL_Ext_Test_RMSE;
        end
        dirname = [strcat('Oct12_Result_with_Feat_Sel_',string(l))];
        mkdir(dirname);
        ffname1 = strcat(dirname,'/TOTAL_TrainResult_Dataset', string(i), '.mat');
        ffname2 = strcat(dirname,'/TOTAL_TestResult_Dataset', string(i),'.mat');
        ffname3 = strcat(dirname,'/TOTAL_Ext_TestResult_Dataset', string(i), '.mat');
        save(ffname1,'TOTAL_TrainResult');
        save(ffname2,'TOTAL_TestResult');
        save(ffname3,'TOTAL_Ext_TestResult');
    end
end

