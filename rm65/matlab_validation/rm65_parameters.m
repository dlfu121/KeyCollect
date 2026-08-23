function robot = rm65_parameters(model, d6Override)
%RM65_PARAMETERS 返回 RM65 系列机械臂的改进 D-H 参数和关节限位。
%   robot = RM65_PARAMETERS("RM65-B") 使用指定型号的法兰长度。
%   robot = RM65_PARAMETERS("CUSTOM", d6) 使用自定义 d6，单位为米。
%
%   厂家约定：模型角度 = 机械关节角度 + offset。
%   内部长度单位为米，角度单位为弧度。

if nargin < 1 || isempty(model)
    model = "RM65-B";
end
model = upper(string(model));

if nargin >= 2 && ~isempty(d6Override)
    d6 = double(d6Override);
else
    switch model
        case {"RM65-B", "RM65-B-V"}
            d6 = 0.144;
        case {"RM65-6FB", "RM65-6FB-V"}
            d6 = 0.161;
        case "RM65-6F"
            d6 = 0.1725;
        otherwise
            error('RM65:UnknownModel', '未知型号 %s，请显式提供 d6（米）。', model);
    end
end

robot.model = model;
robot.a = [0, 0, 0.256, 0, 0, 0];
robot.alpha = deg2rad([0, 90, 0, 90, -90, 90]);
robot.d = [0.2405, 0, 0, 0.210, 0, d6];
robot.offset = deg2rad([0, 90, 90, 0, 0, 0]);
robot.limitsDeg = [-178, 178; -130, 130; -135, 135; ...
                   -178, 178; -128, 128; -360, 360];
robot.limits = deg2rad(robot.limitsDeg);
end
