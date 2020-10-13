function [model, final_x]= feature_selection(X,Y,i)
addpath('C:\Users\Njava\Documents\Arash\ArashAI_Project-master\FeatureSelection_functions')

switch i
    
    case 1
        [model, final_x]=LassoFS_Fun(X,Y);
    case 2
        [model, final_x]=Lassoglm_Fun(X,Y);
    case 3
        [model, final_x] = NCA_Fun(X,Y);
%     case 4
%         [model, final_x] = Ftest_Fun(X,Y);
%     case 5
%         [model, final_x] = Relieff_Fun(X,Y);
%     case 6
%         [model, final_x] = Sequentialfs_Fun(X,Y);
    case 4
        [model, final_x] = GPR_Fun(X,Y);
%     case 7
%         [model, final_x] = Linear_Fun(X,Y);
        
%     case 'inffs'
%         % Infinite Feature Selection 2015 updated 2016
%         alpha = 0.5;    % default, it should be cross-validated.
%         sup = 1;        % Supervised or Not
%         [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );
%         
%     case 'ilfs'
%         % Infinite Latent Feature Selection - ICCV 2017
%         [ranking, weights] = ILFS(X_train, Y_train , 6, 0 );
%         
%     case 'fsasl'
%         options.lambda1 = 1;
%         options.LassoType = 'SLEP';
%         options.SLEPrFlag = 1;
%         options.SLEPreg = 0.01;
%         options.LARSk = 5;
%         options.LARSratio = 2;
%         nClass=2;
%         [W, S, A, objHistory] = FSASL(X_train', nClass, options);
%         [v,ranking]=sort(abs(W(:,1))+abs(W(:,2)),'descend');
%     case 'lasso'
%         lambda = 25;
%         B = lasso(X_train,Y_train);
%         [v,ranking]=sort(B(:,lambda),'descend');
%         
%     case 'ufsol'
%         para.p0 = 'sample';
%         para.p1 = 1e6;
%         para.p2 = 1e2;
%         nClass = 2;
%         [~,~,ranking,~] = UFSwithOL(X_train',nClass,para) ;
%         
%     case 'dgufs'
%         
%         S = dist(X_train');
%         S = -S./max(max(S)); % it's a similarity
%         nClass = 2;
%         alpha = 0.5;
%         beta = 0.9;
%         nSel = 2;
%         [Y,L,V,Label] = DGUFS(X_train',nClass,S,alpha,beta,nSel);
%         [v,ranking]=sort(Y(:,1)+Y(:,2),'descend');
%         
%         
%     case 'mrmr'
%         ranking = mRMR(X_train, Y_train, numF);
%         
%     case 'relieff'
%         [ranking, w] = reliefF( X_train, Y_train, 20);
%         
%     case 'mutinffs'
%         [ ranking , w] = mutInfFS( X_train, Y_train, numF );
%         
%     case 'fsv'
%         [ ranking , w] = fsvFS( X_train, Y_train, numF );
%         
%     case 'laplacian'
%         W = dist(X_train');
%         W = -W./max(max(W)); % it's a similarity
%         [lscores] = LaplacianScore(X_train, W);
%         [junk, ranking] = sort(-lscores);
%         
%     case 'mcfs'
%         % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
%         options = [];
%         options.k = 5; %For unsupervised feature selection, you should tune
%         %this parameter k, the default k is 5.
%         options.nUseEigenfunction = 4;  %You should tune this parameter.
%         [FeaIndex,~] = MCFS_p(X_train,numF,options);
%         ranking = FeaIndex{1};
%         
%     case 'rfe'
%         ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
%         
%     case 'l0'
%         ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
%         
%     case 'fisher'
%         ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
%         
%         
%     case 'ecfs'
%         % Features Selection via Eigenvector Centrality 2016
%         alpha = 0.5; % default, it should be cross-validated.
%         ranking = ECFS( X_train, Y_train, alpha )  ;
%         
%     case 'udfs'
%         % Regularized Discriminative Feature Selection for Unsupervised Learning
%         nClass = 2;
%         ranking = UDFS(X_train , nClass );
%         
%     case 'cfs'
%         % BASELINE - Sort features according to pairwise correlations
%         ranking = cfs(X_train);
%         
%     case 'llcfs'
%         % Feature Selection and Kernel Learning for Local Learning-Based Clustering
%         ranking = llcfs( X_train );
%         
        
end
