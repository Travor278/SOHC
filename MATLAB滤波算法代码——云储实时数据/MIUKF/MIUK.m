clear;clc;
%% 模型参数
% load('R0.mat');
% R0=R0*0.9;
% load('R1.mat');
% R1=R1*1.5;
% load('R2.mat');
% R1=R1*1.3;
% load('C1.mat');
% load('C2.mat');
R0 = 0.0001;
R1 = 0.000302189866398078;
C1 = 100000;
R2 = 0.00152546021732605;
C2 = 9985.09709326495;
load('charge.mat');%放电数据
load('ocv-soc.mat');%OCV-SOC关系
Ts=30;%采样间隔
% discharge_udds_udds=discharge_udds_udds(:,1:8000);
charge_udds=unnamed(:,1:553);
Qn=280*3600;%容量，单位：A*S库伦
%% 系统矩阵
A=[1-1*Ts/R1/C1 0 0;0 1-1*Ts/R2/C2 0;0 0 1];
B=[1*Ts/C1;1*Ts/C2;-1*Ts/Qn];
C=[-1 -1 0];
D=0;
P0=0.01*[0.01 0 0;0 0.01 0;0 0 1];%状态误差协方差初始值
%% 赋值
tm=charge_udds(1,:)';
Cur=-charge_udds(2,:)';
Vot=charge_udds(3,:)';   %端电压
RSOC=charge_udds(4,:)';
T=length(tm);
%% 初始值1
Q=0.00001*eye(3);  %系统误差协方差 eye返回3*3单位矩阵
R=1.1;  %测量误差协方差
a=3; %移动窗口大小（新息长度p），调整该值会影响估计结果
lanmuta(1)=1;  %系数，参考MI-EKF算法   λ，不同的权重
for i=1:a-1
    lanmuta(i+1)=0.5/(a-1);        %λ 1，0.026
end
%% OCV-SOC
x=unnamed1(2,:);
y=unnamed1(1,:);
p=polyfit(x,y,8);  % polyfit函数是matlab中用于进行曲线拟合的一个函数，8为拟合的阶数
%p=polyfit(x,y,5);  % polyfit函数是matlab中用于进行曲线拟合的一个函数，5为拟合的阶数
% figure(5);
% plot(x,y);
%% 初始值
Xekf=[0;0;0.4];  %[U1,U2,SOC]初始值    SOC初值反推OCV
L=length(Xekf);
Uoc(1)=p(1)*Xekf(3)^8+p(2)*Xekf(3)^7+p(3)*Xekf(3)^6+p(4)*Xekf(3)^5+p(5)*Xekf(3)^4+p(6)*Xekf(3)^3+p(7)*Xekf(3)^2+p(8)*Xekf(3)+p(9);%开路电压OCV 由初始SOC算初始OCV
%Uoc(1)=p(1)*Xekf(3)^5+p(2)*Xekf(3)^4+p(3)*Xekf(3)^3+p(4)*Xekf(3)^2+p(5)*Xekf(3)+p(6);%开路电压OCV 由初始SOC算初始OCV
Vekf(1)=Uoc(1)+C*Xekf-Cur(1)*R0;    %OCV估计得到的端电压值  c为系统矩阵 
Vekf_error(1)=Vot(1)-Vekf(1);     %端电压误差 
alpha=0.01;   %α
ki=0;
beta=2;   %β
lamba=alpha^2*(L+ki)-L;    %λ  0.0001*3-3=-2.9997  
c=L+lamba;
Wm=[lamba/c 0.5/c+zeros(1,2*L)];  %？
Wc=Wm;
Wc(1)=Wc(1)+(1-alpha^2+beta);
c=sqrt(c);
for i=1:a-1       % 1~19
    delta=c*chol(P0)';
    Xekf_i=Xekf(:,i);
    Y=Xekf_i(:,ones(1,numel(Xekf_i)));
    X=[Xekf_i Y+delta Y-delta];
    LL=length(X);
    xx=zeros(3,1);
    XX=zeros(3,LL);
    for k=1:LL
        XX(:,k)=A*X(:,k)+B*Cur(i);
        xx=xx+Wm(k)*XX(:,k);
    end
    Xekf(:,i+1)=xx;
    H(i,:)=[-1 -1 p(1)*8*Xekf(3,i+1)^7+p(2)*7*Xekf(3,i+1)^6+p(3)*6*Xekf(3,i+1)^5+p(4)*5*Xekf(3,i+1)^4+p(5)*4*Xekf(3,i+1)^3+p(6)*3*Xekf(3,i+1)^2+p(7)*2*Xekf(3,i+1)+p(8)];
    %H(i,:)=[-1 -1 p(1)*5*Xekf(3,i+1)^4+p(2)*4*Xekf(3,i+1)^3+p(3)*3*Xekf(3,i+1)^2+p(4)*2*Xekf(3,i+1)+p(5)];
    X1=XX-xx(:,ones(1,LL));
    P_xx=X1*diag(Wc)*X1'+Q;   %根据k-1时刻的协方差矩阵预估协方差矩阵
    LL_y=length(XX);   % LL_y=7
    yy=zeros(1,1);
    YY=zeros(1,LL_y);
    for k=1:LL_y
        Uoc(i+1)=p(1)*XX(3,k)^8+p(2)*XX(3,k)^7+p(3)*XX(3,k)^6+p(4)*XX(3,k)^5+p(5)*XX(3,k)^4+p(6)*XX(3,k)^3+p(7)*XX(3,k)^2+p(8)*XX(3,k)+p(9);
        %Uoc(i+1)=p(1)*XX(3,k)^5+p(2)*XX(3,k)^4+p(3)*XX(3,k)^3+p(4)*XX(3,k)^2+p(5)*XX(3,k)+p(6);
        YY(k)=Uoc(i+1)+C*XX(:,k)-Cur(i+1)*R0;
        yy=yy+Wm(k)*YY(k);
    end
    Vekf(i+1)=yy;
    Vekf_error(i+1)=Vot(i+1)-Vekf(i+1);
    Y1=YY-yy(:,ones(1,LL_y));
    P_yy=Y1*diag(Wc)*Y1'+R;
    P_xy=X1*diag(Wc)*Y1';
    K(:,i)=P_xy*inv(P_yy);   %inv求逆
    
    Xekf_zengyi=0;
    for m=1:i
        Xekf_zengyi=Xekf_zengyi+lanmuta(m)*K(:,i+1-m)*Vekf_error(i+2-m);    %状态修正过程   xk=xk+Kk（yk-^yk）
    end
    Xekf(:,i+1)=Xekf(:,i+1)+Xekf_zengyi;
    P0=P_xx-K(:,i)*P_xy';    %更新噪声协方差
end

%%
for i=a:T-1    %20—20000
    delta=c*chol(P0)';
    Xekf_i=Xekf(:,i);
    Y=Xekf_i(:,ones(1,numel(Xekf_i)));
    X=[Xekf_i Y+delta Y-delta];
    LL=length(X);
    xx=zeros(3,1);
    XX=zeros(3,LL);
    for k=1:LL
        XX(:,k)=A*X(:,k)+B*Cur(i);
        xx=xx+Wm(k)*XX(:,k);
    end
    Xekf(:,i+1)=xx;
    H(i,:)=[-1 -1 p(1)*8*Xekf(3,i+1)^7+p(2)*7*Xekf(3,i+1)^6+p(3)*6*Xekf(3,i+1)^5+p(4)*5*Xekf(3,i+1)^4+p(5)*4*Xekf(3,i+1)^3+p(6)*3*Xekf(3,i+1)^2+p(7)*2*Xekf(3,i+1)+p(8)];
    %H(i,:)=[-1 -1 p(1)*5*Xekf(3,i+1)^4+p(2)*4*Xekf(3,i+1)^3+p(3)*3*Xekf(3,i+1)^2+p(4)*2*Xekf(3,i+1)+p(5)];
    X1=XX-xx(:,ones(1,LL));
    P_xx=X1*diag(Wc)*X1'+Q;    %状态变量协方差更新  diag对角矩阵
    LL_y=length(XX);
    yy=zeros(1,1);
    YY=zeros(1,LL_y);
    for k=1:LL_y
        Uoc(i+1)=p(1)*XX(3,k)^8+p(2)*XX(3,k)^7+p(3)*XX(3,k)^6+p(4)*XX(3,k)^5+p(5)*XX(3,k)^4+p(6)*XX(3,k)^3+p(7)*XX(3,k)^2+p(8)*XX(3,k)+p(9);
        %Uoc(i+1)=p(1)*XX(3,k)^5+p(2)*XX(3,k)^4+p(3)*XX(3,k)^3+p(4)*XX(3,k)^2+p(5)*XX(3,k)+p(6);
        YY(k)=Uoc(i+1)+C*XX(:,k)-Cur(i+1)*R0;
        yy=yy+Wm(k)*YY(k);
    end
    Vekf(i+1)=yy;
    Vekf_error(i+1)=Vot(i+1)-Vekf(i+1);
    S=sum(Vekf_error);  
    Vpingjun=S/21;    %average error
    Y1=YY-yy(:,ones(1,LL_y));
    P_yy=Y1*diag(Wc)*Y1'+R;   %观测变量更新
    P_xy=X1*diag(Wc)*Y1';
    K(:,i)=P_xy*inv(P_yy);
    Xekf_zengyi=0;
    for m=1:a
        Xekf_zengyi=Xekf_zengyi+lanmuta(m)*K(:,i+1-m)*Vekf_error(i+2-m);
    end

    Xekf(:,i+1)=Xekf(:,i+1)+Xekf_zengyi;
    P0=P_xx-K(:,i)*P_xy';
end
%% 画图
t=0:1:length(tm)-1;
figure(1);
plot(t,Vot,'-k',t,Vekf,'-r','lineWidth',2); grid on
legend('真实值','估计值-MIUKF');
ylabel('端电压(V)','Fontsize', 10)
xlabel('时间(s)', 'Fontsize', 10)

figure(2);
plot(t,RSOC,'-k',t,Xekf(3,:),'-r','lineWidth',2); grid on
legend('真实值','估计值-MIUKF');
ylabel('SOC','Fontsize', 10)
xlabel('时间(s)', 'Fontsize', 10)
    V_error=Vot-Vekf';
    SOC_error=RSOC-Xekf(3,:)';
    SOC_error_mean=mean(abs(SOC_error(500:end)));  %最小误差0.67 
    SOC_error_max=max(abs(SOC_error(500:end)));  %最大误差1.14 
    SOC_error_rmse=sqrt(mean(SOC_error(500:end).^2));
figure(3);
plot(t,V_error,'-k','lineWidth',2); grid on
legend('端电压误差');
ylabel('端电压误差','Fontsize', 10)
xlabel('时间(s)', 'Fontsize', 10)

figure(4);
plot(t,SOC_error,'-k','lineWidth',2); grid on
legend('SOC误差');
ylabel('SOC误差','Fontsize', 10)
xlabel('时间(s)', 'Fontsize', 10)
SOC_MIUKF=Xekf(3,:);
SOC_error_MIUKF=SOC_error;
S=sum(abs(SOC_error)); 
SOC_error_average=S/20000;  %MIUKF平均误差 0.78 
save SOC_MIUKF.mat SOC_MIUKF
save SOC_error_MIUKF.mat SOC_error_MIUKF