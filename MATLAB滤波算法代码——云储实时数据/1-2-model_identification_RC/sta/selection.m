function [Best,fBest] = selection(funfcn,State)
% calculate fitness
SE = size(State,1);
fState = zeros(SE,1);
for i = 1:SE
    fState(i) = feval(funfcn,State(i,:));
end
[fGBest, g] = min(fState);
fBest = fGBest;
Best = State(g,:);