

performFcn = 'crossentropy';  % Cross-Entropy
    
    trainFcn = 'trainscg';% Scaled conjugate gradient backpropagation.
    
    nna=ceil(35+1);
    nnb=ceil(((3/2)*nna));

    nnc=ceil(((1/3)*nnb));
    nnd=ceil(((1/3)*nnc));
    nne=ceil(((1/3)*nnd));
  
  
    net3 = patternnet([nna nnb nnc nnd nne],trainFcn,performFcn);
    net3.divideParam.valRatio = 5/100;
    net3.trainParam.epochs=5000;
    net3.trainParam.max_fail=40;
    net3.trainParam.sigma=5.0e-5;
    net3.trainParam.lambda=5.0e-7;
    
    net3 = train(net3,X',Y');
    YTestPred = classify(net3,X');
