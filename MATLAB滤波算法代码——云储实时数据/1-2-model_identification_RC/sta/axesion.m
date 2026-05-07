function [Best,fBest] = axesion(funfcn,oldBest,SE,Range,beta,delta)

Pop_Lb=repmat(Range(1,:),SE,1);
Pop_Ub=repmat(Range(2,:),SE,1);

Best = oldBest;
fBest  = feval(funfcn,Best);
flag = 0;

State = op_axes(Best,SE,delta); %axesion operator
%Apply  for State > Pop_Ub or State < Pop_Lb
changeRows = State > Pop_Ub;
State(changeRows) = Pop_Ub(changeRows);
changeRows = State < Pop_Lb;
State(changeRows) = Pop_Lb(changeRows);
%Apply  for State > Pop_Ub or State < Pop_Lb
[newBest,fGBest] = selection(funfcn,State);
if fGBest < fBest
    fBest = fGBest;
    Best = newBest;
    flag = 1;
else
    flag = 0;
end

if flag ==1
    State = op_translate(oldBest,Best,SE,beta);%translation operator
    %Apply  for State > Pop_Ub or State < Pop_Lb
    changeRows = State > Pop_Ub;
    State(changeRows) = Pop_Ub(changeRows);
    changeRows = State < Pop_Lb;
    State(changeRows) = Pop_Lb(changeRows);
    %Apply  for State > Pop_Ub or State < Pop_Lb
    [newBest,fGBest] = selection(funfcn,State);
    if fGBest < fBest
        fBest = fGBest;
        Best = newBest;
    end
end