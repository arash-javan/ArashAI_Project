function [model, final_x]= Lassoglm_Fun(X,Y)


opts = statset('UseParallel',true);
[B,S] = lassoglm(X,Y,'poisson','DFmax',100,'CV',10,'Alpha',0.5,'Options',opts);

% [B,FitInfo] = lassoglm(X,Y,'poisson','CV',10);
% Poisson Regression
% [B,S] = lassoglm(x,y,'poisson','DFmax',100,'CV',10,'Alpha',0.5,'Options',opts);
model = B(:,S.IndexMinDeviance)~=0;
final_x=X(:,model);