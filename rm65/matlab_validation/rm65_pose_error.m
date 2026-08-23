function [positionError, orientationError] = rm65_pose_error(actual, target)
%RM65_POSE_ERROR 计算两个齐次位姿之间的位置和姿态误差。

positionError = norm(actual(1:3, 4) - target(1:3, 4));
relativeRotation = target(1:3, 1:3) * actual(1:3, 1:3).';
cosine = max(-1, min(1, (trace(relativeRotation) - 1) / 2));
orientationError = acos(cosine);
end
