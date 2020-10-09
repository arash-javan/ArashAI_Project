clc;
clear all;
close all;


figure;
for i = 1:15
    pathname = strcat('C:\Users\Arash\Downloads\ArashAI_Project_oct8\Result_with_Feat_Extr_',num2str(i));
    addpath(pathname)
    fname = strcat(pathname,'\TOTAL_TestResult_Dataset',num2str(1),'.mat');
    load (fname)
    
    mae = TOTAL_TestResult.MAE;
    %     mae(9,:) = [];
    %     mae(6,:) = [];
    
    mean_mae = mean(mae,2)
    std_mae = std(mae,[],2);
    
    subplot(15,1,i);
    bar(mean_mae);
    hold on
    errorbar(mean_mae,std_mae ,'.')
    %axis([0 12 0 1.1])
    grid minor
    
    hold off
%     names = {'DTI', 'DAT', 'NI', 'DTI&DAT', 'DTI&NI', 'DAT&NI', 'DTI&DAT&NI'};
     names = {'PCA', 'Kernel PCA', 'TSNE','FA', 'Sammon', 'Isomap', 'Landmark Isomap', 'Laplacian Eigen', 'ILE'};
%     %names = {'DTI', 'DAT', 'NI', 'DTI&DAT', 'DTI&NI', 'DAT&NI'};
     ylabel(names(i),'FontSize',5);
    
end
%title('Results of Sammon Feature Extraction');
xlabel('Different regression algorithms');
k=1;
