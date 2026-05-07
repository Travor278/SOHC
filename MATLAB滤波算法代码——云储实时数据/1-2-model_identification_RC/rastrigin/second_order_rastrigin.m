function y = second_order_rastrigin(x)
    global u I L Ts
    num_measurements = size(I);
    
    % ===========================================
    % 1.计算二项式系数
    % ===========================================
    n_d = 2;
    N = L;
    
    % ===========================================
    % 2.遍历每一帧测量数据，开始处理
    % ===========================================
    Vot_pred_all = zeros(num_measurements);
    count = 0;
    for i = 1:num_measurements
        [u_hat] = second_model_sta_process(x,u,I,i,Ts);
        count = count +1;
        Vot_pred_all(count, :) = u_hat;       % 所有端电压预测   
    end
    clear second_model_sta_process
    
    % ===========================================
    % 3.计算损失
    % ===========================================
    y = sum(abs(u - Vot_pred_all));
    fprintf('y = %.4f\n', y)
    
    % plot(u,'b','LineWidth',2)             % 检查点
    % hold on;
    % plot(u_pred_all,'g','LineWidth',2)
    % pause()
end