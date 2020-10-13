function [final_x] = feature_extraction(X,k,num_feats,i)
addpath('C:\Users\Njava\Documents\Arash\ArashAI_Project-master\FSLib_v7.0.1_2020_2\lib\drtoolbox\techniques')
addpath('C:\Users\Njava\Documents\Arash\ArashAI_Project-master\FSLib_v7.0.1_2020_2\lib\drtoolbox')
switch i
    case 1
        final_x = pca(X,num_feats);
    case 2
        final_x = kernel_pca(X,num_feats);
    case 3
        final_x = tsne(X,[],num_feats);
    case 4
        final_x = fa(X,num_feats);
    case 5
        final_x =sammon(X,num_feats);     
    case 6
        final_x =isomap(X,num_feats,k); %%%%[mappedX, mapping] = isomap(X, no_dims, k); that K is number of neighbor      
    case 7
        final_x = landmark_isomap(X, num_feats, k, 0.3); %%%%%%  [mappedX, mapping] = landmark_isomap(X, no_dims, k, percentage);    
    case 8
        final_x = laplacian_eigen(X, num_feats,k, 1, []);%   [mappedX, mapping] = laplacian_eigen(X, no_dims, k, sigma, eig_impl) 
    case 9
        final_x = lle(X, num_feats, k,[]); %%%%mappedX = lle(X, no_dims, k, eig_impl)
    case 10
        final_x = mds(X, num_feats);%%%%%mappedX = mds(X, no_dims);   
    case 11
        final_x = diffusion_maps(X, num_feats, 1, 1); %%%%mappedX = diffusion_maps(X, no_dims, alpha, sigma)&sigma is the variance of the Gaussian & The variable alpha
        % determines the operator that is applied on the graph (default = 1)       
    case 12
        final_x = spe(X, num_feats, 'Local', k);%   Y = spe(X, no_dims, 'Global')%   Y = spe(X, no_dims, 'Local', k)
    case 13
        final_x = gplvm(X, num_feats, 1);%%% Y = gplvm(X, no_dims, sigma)  
    case 14
        final_x = sne(X, num_feats, 30);%%%  mappedX = sne(X, no_dims, perplexity) 
    case 15
        final_x = sym_sne(X, num_feats, 30);%%%mappedX = sym_sne(X, no_dims, perplexity);
end


