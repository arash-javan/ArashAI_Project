function [model, final_x]= GPR_Fun(X,Y)

sigmaL0 = sqrt(size(X,2))*ones(size(X,2),1); % Length scale for predictors
sigmaF0 = 1; % Signal standard deviation
sigmaN0 = 1;

opts = statset('fitrgp');

mdl = fitrgp(X,Y,'KernelFunction','ardsquaredexponential','Verbose',1, ...
    'Optimizer','lbfgs','OptimizerOptions',opts, ...
    'KernelParameters',[sigmaL0;sigmaF0],'Sigma',sigmaN0,'InitialStepSize',1);

sigmaL = mdl.KernelInformation.KernelParameters(1:end-1); % Learned length scales
weights = exp(-sigmaL); % Predictor weights
weights = weights/sum(weights); % Normalized predictor weights

% figure()
% semilogx(weights,'ro')
% grid on
% xlabel('Feature index')
% ylabel('Feature weight')

model = weights > 0 ;
final_x=X(:,model);