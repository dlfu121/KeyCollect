function [transform, frames] = rm65_fk(joints, robot, useDegrees)
%RM65_FK 计算 RM65 法兰正运动学以及各关节坐标系。
%   transform = RM65_FK(joints, robot) 默认输入为弧度。
%   [transform, frames] 同时返回 4×4×7 的基座及六关节坐标系。

if nargin < 3
    useDegrees = false;
end
joints = double(joints(:).');
if numel(joints) ~= 6
    error('RM65:JointCount', 'joints 必须包含 6 个关节角。');
end
if useDegrees
    joints = deg2rad(joints);
end

transform = eye(4);
frames = zeros(4, 4, 7);
frames(:, :, 1) = transform;
for index = 1:6
    theta = joints(index) + robot.offset(index);
    transform = transform * rm65_mdh_transform( ...
        robot.a(index), robot.alpha(index), robot.d(index), theta);
    frames(:, :, index + 1) = transform;
end
end
