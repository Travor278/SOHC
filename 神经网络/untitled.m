Tlen = 2001;
time = (0:Tlen-1)';   % 单位：步数（或秒）

%% ================== 3. 合并数据 ==================
% 每一行 = 一个时间点
% 列顺序：time | SOC1~SOC8 | V1~V8 | I1~I8 | T1~T8

data = [time,out.V1,out.V2,out.V3,out.V4,out.V5,out.V6,out.V7,out.V8,out.SOC1,out.SOC2,out.SOC3,out.SOC4,out.SOC5,out.SOC6,out.SOC7,out.SOC8,out.I1,out.I2,out.I3,out.I4,out.I5,out.I6,out.I7,out.I8,out.T1,out.T2,out.T3,out.T4,out.T5,out.T6,out.T7,out.T8]
varNames = strings(1, 1 + 8*4);
idx = 1;

varNames(idx) = "time";
idx = idx + 1;



for i = 1:8
    varNames(idx) = "V" + i;
    idx = idx + 1;
end




for i = 1:8
    varNames(idx) = "SOC" + i;
    idx = idx + 1;
end

for i = 1:8
    varNames(idx) = "I" + i;
    idx = idx + 1;
end

for i = 1:8
    varNames(idx) = "T" + i;
    idx = idx + 1;
end


%% ================== 5. 转为 table ==================

T_table = array2table(data, "VariableNames", varNames);
filename = "battery_data3.csv";
writetable(T_table, filename);

fprintf("CSV 文件已保存：%s\n", filename);
fprintf("数据尺寸：%d 行 × %d 列\n", size(T_table,1), size(T_table,2));