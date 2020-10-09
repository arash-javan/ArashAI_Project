function [model, final_x]= NCA_Fun(X,Y)
N = 100;

mdl = fsrnca(X,Y,'Verbose',1,'Lambda',0.5/N);
% 
% figure()
% plot(mdl.FeatureWeights,'ro')
% grid on
% xlabel('Feature index')
% ylabel('Feature weight')
% Poisson Regression
% [B,S] = lassoglm(x,y,'poisson','DFmax',100,'CV',10,'Alpha',0.5,'Options',opts);
model = mdl.FeatureWeights > 0.2 ;
final_x=X(:,model);