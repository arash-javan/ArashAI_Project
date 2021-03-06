clc;
clear all;
close all;

figure;
for i = 1:7
    pathname = strcat('C:\Users\Arash\Downloads\ArashAI_Project_oct8\Oct_8_Result_withCtrl_withoutanyDRA');
    addpath(pathname)
    %fname = strcat(pathname,'\TOTAL_Ext_TestResult_Dataset',num2str(i),'.mat');
    fname = strcat(pathname,'\TOTAL_TestResult_Dataset',num2str(i),'.mat');
    load (fname)
    
    %mae = TOTAL_Ext_TestResult.MAE;
    mae = TOTAL_TestResult.MAE;
    %     mae(9,:) = [];
    %     mae(6,:) = [];
        for j = 1:5
        [i j]
        final_mae = mae(j,:,:)
        mean_mae = mean(final_mae,3)
        std_mae = std(final_mae,[],3);
        
        subplot(7,1,i);
        bar(mean_mae);
        hold on
        errorbar(mean_mae,std_mae ,'.')
        %axis([0 12 0 1.1])
        grid minor
        
        hold off
             names = {'DTI', 'DAT', 'NI', 'DTI&DAT', 'DTI&NI', 'DAT&NI', 'DTI&DAT&NI'};
        %names = {'PCA', 'Kernel PCA', 'TSNE','FA', 'Sammon', 'Isomap', 'Landmark Isomap', 'Laplacian Eigen', 'ILE'};
        %     %names = {'DTI', 'DAT', 'NI', 'DTI&DAT', 'DTI&NI', 'DAT&NI'};
        ylabel(names(i),'FontSize',5);
    end
end
%title('Results of Sammon Feature Extraction');
xlabel('Different regression algorithms');
k=1;
