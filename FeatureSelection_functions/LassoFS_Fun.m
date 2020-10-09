function [model, final_x]= LassoFS_Fun(X,Y)
opts = statset('UseParallel',true);
featureSum = zeros(12,1);
featureWeight = zeros(50,1);

% [B,S] = lassoglm(x,y);
% featureSum = sum(B(:,S.IndexMinDeviance)~=0);
% featureWeight = featureWeight+B(:,S.IndexMinDeviance);
[B,S] = lasso(X,Y,'DFmax',100,'CV',10,'Alpha',0.5,'Options',opts);
%[B,S] = lasso(X,Y,'DFmax',50);
% Poisson Regression
% [B,S] = lassoglm(x,y,'poisson','DFmax',100,'CV',10,'Alpha',0.5,'Options',opts);
model = B(:,S.IndexMinMSE)~=0;
final_x=X(:,model);
end