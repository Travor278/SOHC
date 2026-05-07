clear all
close all

% 设置工程子文件夹目录
addpath('data')
addpath('settings')
addpath('model_verification')
% 全局变量
global L

% 0.加载配置文件
ini = ini2struct('Second_order_mol_eis_veri.ini');

% 1.加载电压电流数据
data = xlsread(ini.data_file.filename); %这里读出来只有4列，日期、电流、电压、SOC
disp('加载OCV-SOC曲线，并进行多项式拟合')
OCV_SOC = xlsread(ini.main.ocv_soc_file);
OCV_SOC = OCV_SOC';                     %获取SOC_OCV
x=OCV_SOC(2,:);                         %SOC
y=OCV_SOC(1,:);                         %OCV
p=polyfit(x,y,9);                       %多项式参数值
disp('多项式拟合完成')

% 2.加载生成的ground truth和测量数据
n_x = 3;
n_z = 1;
n_d = 3;
alpha = [ini.param.d1	ini.param.d2  1]; %Mol参数

L = ini.main.L;
N = L;

% 初始化二项式系数
Bino_Fir = zeros(n_d,n_d,N);
Bino_Fir(:,:,1) = [1 0 0
                   0 1 0
                   0 0 0];
for i = 2:1:N
    last_bino_mat = Bino_Fir(:,:,i-1);
    now_bino_mat = zeros(n_d, n_d);
    now_bino_mat(1,1) = (1-(alpha(1)+1)/(i-1))*last_bino_mat(1,1);  % (1-(alpha+1)/(i-1))是二项系数的一个等价的计算公式
    now_bino_mat(2,2) = (1-(alpha(2)+1)/(i-1))*last_bino_mat(2,2);
    now_bino_mat(3,3) = 0;
    Bino_Fir(:,:,i) = now_bino_mat;
end

% 3.初始化结果向量
count = 0;
I = data(:,2);
u_init = data(1,3);
num_measurements = length(I);
Vot_pred_all_resample = zeros(num_measurements,1);
SOC_pred_all_resample = zeros(num_measurements,1);
U234_pred_all_resample = zeros(num_measurements,2);

% 4.遍历每一帧测量数据，开始处理
for i = 1:num_measurements
   [X, Vot_pred,U2,U3] = second_model_equal_verification_process(u_init,I, i, p, Bino_Fir, alpha, ini);
   count = count +1;
   SOC_pred_all_resample(count) = X(3);
   U234_pred_all_resample(count,:) = [U2;U3];
   Vot_pred_all_resample(count, :) = Vot_pred;       % 所有端电压预测
end
U234_pred_all =  U234_pred_all_resample;
SOC_pred_all = SOC_pred_all_resample;
Vot_pred_all = Vot_pred_all_resample;
Vot_gt_all = data(:,3);
SOC_gt_all = data(:,4);
fprintf('Total_num_frames = %d\n',count);
disp('正在画图中，稍安勿躁~')
figure(1);
hold on;grid on; box on;
plot(Vot_gt_all(1:end),'color', [0 0 0]/255,'LineWidth',1.5);             % 绘制模型电压
plot(Vot_pred_all(1:end),'color', [255 0 255]/255,'LineWidth',1.5);             % 绘制ground truth
%plot(U234_pred_all(:,1))
%plot(U234_pred_all(:,2))
legend('Measurement','Estimation of IOM');
xlabel('t(s)')
ylabel('Voltage(V)')

figure(2)
hold on; box on; grid on;
Vot_error = Vot_pred_all-Vot_gt_all;
plot(Vot_error(1:end),'color', [255 0 255]/255,'LineWidth',1.5)
xlabel('t(s)')
ylabel('Voltage error(V)')

disp('RMSE为')
disp( sqrt(sum((Vot_pred_all - Vot_gt_all).^2)/num_measurements) )
disp('MAE为')
disp( sum(abs(Vot_pred_all - Vot_gt_all))/ num_measurements )
disp('maximum为')
disp( max(abs(Vot_pred_all - Vot_gt_all)) )

