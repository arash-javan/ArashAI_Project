function [model, final_x]= Ftest_Fun(X,Y)

[idx,scores] = fsrftest(X,Y);

% 
% figure()
% bar(scores(idx))
% grid on
% xlabel('Predictor rank')
% ylabel('Predictor importance score')
% num_feats = round(numel(X(:,1))*0.10);

idxs = idx(1:num_feats);
model = false(size(X,2),1);
for i = 1:size(idxs,2)
    model(idxs(i)) = 1;
end
final_x=X(:,model(:,1));
end