function [model, final_x]= Linear_Fun(X,Y)

Lambda = logspace(-5,-1,15);
X = X'; 
rng(10); % For reproducibility
CVMdl = fitrlinear(X,Y,'ObservationsIn','columns','KFold',5,'Lambda',Lambda,...
    'Learner','leastsquares','Solver','sparsa','Regularization','lasso');

mse = kfoldLoss(CVMdl);
numNZCoeff = sum(CVMdl.Trained{1}.Beta~=0);
figure
[h,hL1,hL2] = plotyy(log10(Lambda),log10(mse),...
    log10(Lambda),log10(numNZCoeff)); 
hL1.Marker = 'o';
hL2.Marker = 'o';
ylabel(h(1),'log_{10} MSE')
ylabel(h(2),'log_{10} nonzero-coefficient frequency')
xlabel('log_{10} Lambda')
hold off
MdlFinal = selectModels(CVMdl.Trained{1},11);
idx = find(MdlFinal.Beta~=0);

idxs = idx(1:num_feats);
model = false(size(X,2),1);
for i = 1:size(idxs,2)
    model(idxs(i)) = 1;
end
final_x=X(:,model);