function y = op_expand(Best,SE,gamma)
%ÉìËõ±ä»»
n = length(Best);
y = repmat(Best',1,SE) + gamma*(normrnd(0,1,n,SE).*repmat(Best',1,SE));
y = y';