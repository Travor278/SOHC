function y=op_rotate(Best,SE,alpha)
%Ðý×ªËã×Ó
n = length(Best);
y = repmat(Best',1,SE) + alpha*(1/n/(norm(Best)+eps))*reshape(unifrnd(-1,1,SE*n,n)*Best',n,SE);
y = y';