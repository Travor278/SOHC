% ===================================
% 优化三步曲
% 1.更改数据段
% 2.在Initialization.m更改参数初值
% 3.在Rastrigin.m更改R0
% ===================================

clear all
close all
% 全局变量
global u I L Ts

% 设置工程子文件夹目录
addpath('data')
addpath('settings')
addpath('rastrigin')

% 0.加载配置文件
ini = ini2struct('Second_order_sta.ini');


% ================================================
% 1.读取HPPC测试数据文件
% ================================================
currentFolder = pwd;
addpath(genpath(currentFolder))
warning('off')
opts = spreadsheetImportOptions("NumVariables", 5);
opts.Sheet = "Sheet1";
opts.DataRange = ini.data_file.data_range;        
num_data = ini.data_file.use_data_num;            % 使用数据数目，数据数目越多，对低频部分的拟合效果就越好
% 指定列名称和类型
opts.VariableNames = ["I", "U","dontcare1","dontcare1","SOC"];
opts.VariableTypes = ["double","double","double","double","double"];

% 导入数据
tbl = readtable(ini.data_file.filename, opts, "UseExcel", false);


% ================================================
% 2.读取HPPC测试电压电流SOC
% ================================================
u = tbl.U;   u_origin = u;       % 保存原始电压，绘图用
I = tbl.I;
I = -I;                          % 统一在main函数给I反向
u = u(1:num_data);
I = I(1:num_data);

% 加载OCV-SOC曲线
disp('加载OCV-SOC曲线，并进行多项式拟合')
OCV_SOC = xlsread(ini.main.ocv_soc_file);
OCV_SOC = OCV_SOC';                     % 获取SOC_OCV
SOC_battery=OCV_SOC(2,:);               % SOC
OCV_battery=OCV_SOC(1,:);               % OCV
SOC = tbl.SOC(1:num_data);
Uoc = interp1(SOC_battery,OCV_battery,SOC,'pchip');

% 端电压减去Uoc
u = u-Uoc;                              % 检查点1，减去Uoc之后的形状要对
plot(u)

% 清除临时变量
clear opts tbl


% ================================================
% 3.开始STA优化算法
% ================================================
SE =  70; % degree of search enforcement
Dim = 7;% dimension
L = ini.sta.L;
Ts = ini.sta.Ts;
Range = [0.0001 0.1         % R0上下界
         0.0001 0.1         % R1上下界
         0.0001 0.1         % R2上下界
         1e1  1e5             % C1上下界
         1e1  1e4            % C2上下界
         0     1              % d1上下界
         0     1              % d2上下界
         ]';        
Iterations = ini.sta.iterations;% maximum number of iterations
disp('优化参数设置完成，开始优化')
tic
[Best,fBest,history] = STA(@second_order_rastrigin,SE,Dim,Range,Iterations);
toc
semilogy(history)
save(ini.main.savename);




