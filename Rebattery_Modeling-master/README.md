## Matlab simulation of BMS

Tengfei modified 0719-1952

FEI-CSU modified 0719-2058

Tengfei modified 0719-2138

done!  zhaoxu@20230719-2258@macos

done!  zhaoxu@20230719-2327@win@github desktop

done!  zhaoxu@20230719-2338@win@matlab





#### Tengfei modified 0720-1130

1）Simple modification of the simulation interface,

2）Display of synchronization signal,

3）PID parameter is not adjusted.



#### Tengfei modified 0720-1655

1）The inner loop control mode is slightly adjusted to change the problem that the voltage cannot be dropped，

2）The PID parameters of the outer ring were slightly adjusted.



#### Tengfei modified 0722-1012

The resistance was removed from the subsystem.







#### Tengfei modified 0722-1620

1）The system was changed from 7 modules to 30 modules, each with a rated voltage of 25.6v,

2）Modify some MOSFET parameters by referring to the manual.


#### zhaoxu  0723-1200
1) add tic and toc on slx, so we can get the simulation time after the slx is finished.
2) the simulation time is 982.61s on i7-12700f/16G
3) during switch the battery modular on， it seems that the output voltage ripple is singnificent high especially.
 I suggest to retune the pid parametars.


### zhaoxu 0723-1243
1) add second switch to the battery modular.

#### Tengfei modified 0727-1928

1）The accumulator with zero clearing function is realized, and the open-loop voltage ripple is improved,

2）The PID parameters need to be further adjusted.

#### Tengfei modified 0729-2010

1）The inner loop parameters are adjusted to reduce the output voltage ripple,

2）The reason for the poor PID control effect before is that the proportion is too large, resulting in the system becoming 01 regulation,

3）Next, adjust the parameters of the outer ring PID.

#### Tengfei modified 0730-1700

1）Added comments and streamlined.

#### Tengfei modified 0818-1500

1）Labeled the final version of the simulation model.

#### cxx test 0819-1129

1）test
#### cxx submit 1009-2126

1）Submit the single_battery_charge_discharge model.
