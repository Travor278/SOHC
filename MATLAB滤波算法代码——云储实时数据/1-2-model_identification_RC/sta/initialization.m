function  State=initialization(SE,Dim,Range)
    addpath('settings')

    % 0.¼ÓÔØÅäÖÃÎÄ¼ş
    ini = ini2struct('Second_order_sta.ini');

    if ini.main.order == 2
        R0 = ini.init.R0;
        R1 = ini.init.R1;
        C1 = ini.init.C1;
        R2 = ini.init.R2;
        C2 = ini.init.C2;

        d1 = ini.init.d1;
        d2 = ini.init.d2;

        State = [R0 R1 R2 C1 C2 d1 d2];      
    end

end
