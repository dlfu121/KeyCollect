%RUN_RM65_VALIDATION RM65 MATLAB 解析逆运动学可视化验证入口。
% 直接运行本脚本即可开始 RM65-B、12 秒、30 Hz 的实时动画。

clc;
close all;
model = "RM65-B";
durationSeconds = 12;
realtimePlayback = true;

validationResult = rm65_validate_animation(model, durationSeconds, realtimePlayback);
