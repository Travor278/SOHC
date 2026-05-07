function [Best,fBest,history] = STA(funfcn,SE,Dim,Range,Iterations)
% parameter setting
alpha_max = 0.01;
alpha_min = 1e-4;
alpha = alpha_max;
beta = 0.01;
gamma = 0.01;
delta = 1;
fc = 2;
% initialization
State = initialization(SE,Dim,Range);
[Best,fBest] = selection(funfcn,State);
counter = 0;
% iterative process
for iter = 1:Iterations
    if alpha < alpha_min
        alpha = alpha_max;
    end
    % flag
    oldfBest = fBest;
    
    [Best,fBest] = expand(funfcn,Best,SE,Range,beta,gamma);
    
    [Best,fBest] = rotate(funfcn,Best,SE,Range,alpha,beta);
    
    [Best,fBest] = axesion(funfcn,Best,SE,Range,beta,delta);

    fprintf('iter=%d      ObjVal=%g\n',iter,fBest);
    history(iter) = fBest;
    alpha = alpha/fc;
end



