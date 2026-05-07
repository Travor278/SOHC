function [Z_pred] = second_model_sta_process(x,u,I,cnt,Ts)
    % 输入 ：one_frame每一帧的电压、电流；cnt：索引，到第几帧了
    % 输出：x_ 状态变量， Z_pred:端电压预测值
    persistent is_initialized 
    persistent x_history cur_history        % 仅仅存储上一时刻的x_
    persistent x_array                      % 存储所有时刻的x_，是一个动态数组
    
    % 状态维度和测量维度
    n_x = 2;
    
    % =====================================================================
    %                         初始化模型参数
    % =====================================================================
    Cur = I(cnt);
    ModelParam.R0 = x(1);
    ModelParam.R1 = x(2);
    ModelParam.R2 = x(3);
    ModelParam.C1 = x(4);
    ModelParam.C2 = x(5); 
    if ( isempty(is_initialized) )
        is_initialized = false;
    end
    
    % 开始算法
    if (~is_initialized)
        % =================================================================
        %                      0. 初始化第一帧数据
        % =================================================================
        x_ = [0 0]';
        Z_pred = u(1);   % 第一帧返回的值是测量值本身
        
        is_initialized = true;
    else
        % =================================================================
        %                      1. 系统方程状态转移
        % =================================================================
        x_ = x_history;
        Cur_diff = cur_history;
        [x_] = State_Trans(x_, Cur_diff, Ts, ModelParam,n_x);
        x_pred = x_;
        
        % =================================================================
        %                      2. 测量方程计算端电压
        % =================================================================
        [Z_pred] = Measure(x_pred, Cur, ModelParam);
        
    end
    
    % 记录历史数据
    x_history = x_;
    cur_history = Cur;
    
    
end

% =================================================================
%                      1. 系统方程状态转移
% =================================================================
function [x_pre] = State_Trans(x_,  Cur_diff, Ts, ModelParam, n_x)
    % 模型参数
    R1 = ModelParam.R1;
    C1 = ModelParam.C1;
    R2 = ModelParam.R2;
    C2 = ModelParam.C2;
    
    % 初始化状态的结果向量
    diff_X_esti = zeros(n_x, 1);
    x_pre = zeros(n_x,1);
    
    % state prediction   
    diff_X_esti(1,1) = ( 1-Ts/(R1*C1) ) * x_(1,1) + ( Ts/C1 ) * Cur_diff;
    diff_X_esti(2,1) = ( 1-Ts/(R2*C2) ) * x_(2,1) + ( Ts/C2 ) * Cur_diff;
    
    x_pre(:,1) = diff_X_esti(:,1);
    
end

% =================================================================
%                      2. 测量方程计算端电压
% =================================================================
function [Z_pred] = Measure(x_pred, Cur, ModelParam)
    R0 = ModelParam.R0;
    U2 = x_pred(1,1);
    U3 = x_pred(2,1);
    Z_pred = -R0*Cur - U2 - U3;
end
