function y=op_translate(oldBest,newBest,SE,beta)
%ÒÆÎ»Ëã×Ó
n = length(oldBest);
y = repmat(newBest',1,SE) + beta/(norm(newBest-oldBest)+eps)*reshape(kron(rand(SE,1),(newBest-oldBest)'),n,SE);
y = y';