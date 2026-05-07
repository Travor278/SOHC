function [x_, Z_pred , U2 , U3] = second_model_equal_verification_process(u_init,I, cnt, p, bino_fir, alpha, ini)
    %输入 ：one_frame每一帧的电压、电流；cnt：索引，到第几帧了；p：OCV_SOC多项式
    %输出：x_ 状态变量， Z_pred:端电压预测值
    global n_x n_z check_output L ModelParam          % L是滑动窗的大小，控制分数阶状态项的个数
    persistent is_initialized 
    persistent x_history                    % 仅仅存储上一时刻的x_
    persistent x_array                      % 存储所有时刻的x_，是一个动态数组
    check_output = true;
    
    % 状态维度和测量维度
    n_x = 3;
    n_z = 1;
    if(mod(cnt,1000) == 0)
        disp(cnt)
    end

    % =====================================================================
    %                         初始化模型参数
    % =====================================================================
    Ts = ini.main.Ts;
    Cur = -I(cnt); % 电流
    if(cnt > 1)
        Cur_diff = -I(cnt-1);
    end
    if ( isempty(ModelParam) )
        ModelParam.Qn = ini.param.Qn * 3600;     
        ModelParam.R0 = ini.param.R0;
        ModelParam.R1 = ini.param.R1;
        ModelParam.C1 = ini.param.C1;
        ModelParam.R2 = ini.param.R1;
        ModelParam.C2 = ini.param.C2; 
    end
    if ( isempty(is_initialized) )
        is_initialized = false;
    end
    
    % 开始算法
    if (~is_initialized)
        % =================================================================
        %                      0. 初始化第一帧数据
        % =================================================================
        x_ = [0 0 1]';
        Z_pred = u_init;   % 第一帧返回的值是测量值本身
        Cur_diff = 0;
        U2 = 0;
        U3 = 0;
        is_initialized = true;
    else
        % =================================================================
        %                      1. 系统方程状态转移
        % =================================================================
        x_ = x_history;
        [x_] = State_Trans(x_, x_array, Cur, Cur_diff, Ts, alpha, cnt, bino_fir, ModelParam);
        x_pred = x_;
        
        % =================================================================
        %                      2. 测量方程计算端电压
        % =================================================================
        [Z_pred , U2 , U3] = Measure(x_pred, Cur, ModelParam, p);
        
    end
    
    % 记录历史数据
    x_history = x_;
    x_array = [x_array x_];      % 插入当前的状态估计x_
    if(length(x_array) > L+1)
        x_array = x_array(:,end-L:end);
    end
    % pause();
    
    
end

% =================================================================
%                      1. 系统方程状态转移
% =================================================================
function [x_pre] = State_Trans(x_, x_array, Cur, Cur_diff, Ts,alpha, cnt, bino_fir, ModelParam)
    global n_x L
    % 模型参数
    Qn = ModelParam.Qn;
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
    diff_X_esti(3,1) = x_(3,1) + (-1*Ts/Qn) * Cur_diff;
    
    x_pre(:,1) = diff_X_esti(:,1);
    
end

% =================================================================
%                      2. 测量方程计算端电压
% =================================================================
function [Z_pred, U2 , U3 , U4] = Measure(x_pred, Cur, ModelParam, p)
    % 模型参数
    R0 = ModelParam.R0;
    U2 = x_pred(1,1);
    U3 = x_pred(2,1);
    SOC_pred = x_pred(3,1);
    Uoc = polyval(p,SOC_pred);

    Z_pred = -R0*Cur - U2 - U3 + Uoc;

end

