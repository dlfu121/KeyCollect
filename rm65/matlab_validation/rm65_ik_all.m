function solutions = rm65_ik_all(target, robot, seed, tolerance)
%RM65_IK_ALL 枚举关节限位内的 RM65 封闭解析逆解。
%   solutions = RM65_IK_ALL(target, robot, seed) 返回结构体数组。
%   每项包含 q、qDeg、shoulder、elbow、wrist、positionError 和
%   orientationError。seed 仅用于奇异代表解及结果排序，不参与迭代。
%
%   常规位姿按照 2 个肩部分支、2 个肘部分支和 2 个腕部分支展开。
%   J6 的 ±360° 限位还可能产生相差整周的等价关节表示。

if nargin < 3 || isempty(seed)
    seed = zeros(1, 6);
end
if nargin < 4 || isempty(tolerance)
    tolerance = 1e-8;
end
seed = double(seed(:).');
if numel(seed) ~= 6
    error('RM65:SeedCount', 'seed 必须包含 6 个弧度制关节角。');
end
validate_target(target, max(tolerance, 1e-7));

emptyItem = struct('q', {}, 'qDeg', {}, 'shoulder', {}, 'elbow', {}, ...
    'wrist', {}, 'positionError', {}, 'orientationError', {});
solutions = emptyItem;

% 沿法兰 z6 轴回退 d6，得到 J4、J5、J6 三个轴的交点。
wristCenter = target(1:3, 4) - robot.d(6) * target(1:3, 3);
x = wristCenter(1);
y = wristCenter(2);
verticalDistance = wristCenter(3) - robot.d(1);
radialDistance = hypot(x, y);
upperArm = robot.a(3);
forearm = robot.d(4);

% 每行依次为 q1 原始值、带符号的平面径向坐标和肩部编号。
if radialDistance <= tolerance
    shoulderRows = [seed(1), 0, 0];
    shoulderNames = "肩部奇异";
else
    baseAngle = atan2(-y, -x);
    shoulderRows = [baseAngle, radialDistance, 1; ...
                    baseAngle + pi, -radialDistance, 2];
    shoulderNames = ["肩部正支", "肩部反支"];
end

for shoulderIndex = 1:size(shoulderRows, 1)
    q1Raw = shoulderRows(shoulderIndex, 1);
    planarRadius = shoulderRows(shoulderIndex, 2);
    shoulderName = shoulderNames(shoulderIndex);

    cosineQ3 = (planarRadius^2 + verticalDistance^2 - upperArm^2 - forearm^2) / ...
        (2 * upperArm * forearm);
    if cosineQ3 < -1 - tolerance || cosineQ3 > 1 + tolerance
        continue;
    end
    cosineQ3 = max(-1, min(1, cosineQ3));
    elbowAngle = acos(cosineQ3);

    for elbowIndex = 1:2
        if elbowIndex == 1
            q3Raw = elbowAngle;
            elbowName = "肘下";
        else
            q3Raw = -elbowAngle;
            elbowName = "肘上";
        end
        q2Raw = atan2(planarRadius, verticalDistance) - ...
            atan2(forearm * sin(q3Raw), upperArm + forearm * cos(q3Raw));

        q1Values = equivalent_angles(q1Raw, robot.limits(1, :), tolerance);
        q2Values = equivalent_angles(q2Raw, robot.limits(2, :), tolerance);
        q3Values = equivalent_angles(q3Raw, robot.limits(3, :), tolerance);

        for q1 = q1Values
            for q2 = q2Values
                for q3 = q3Values
                    [~, frames] = rm65_fk([q1, q2, q3, 0, 0, 0], robot);
                    rotation03 = frames(1:3, 1:3, 4);
                    rotation36 = rotation03.' * target(1:3, 1:3);

                    % 从 R36 的矩阵元素直接解析腕部三个角度。
                    cosineQ5 = max(-1, min(1, -rotation36(2, 3)));
                    sineQ5Abs = hypot(rotation36(2, 1), rotation36(2, 2));
                    if sineQ5Abs <= tolerance
                        % J5 限位排除了 ±pi，所以这里只需处理 q5=0。
                        q4Rows = seed(4);
                        q5Rows = 0;
                        combined = atan2(rotation36(3, 1), rotation36(1, 1));
                        q6Rows = combined - q4Rows;
                        wristNames = "腕部奇异代表解";
                    else
                        q5Positive = atan2(sineQ5Abs, cosineQ5);
                        q4Rows = zeros(2, 1);
                        q5Rows = [q5Positive; -q5Positive];
                        q6Rows = zeros(2, 1);
                        wristNames = ["腕部不翻转", "腕部翻转"];
                        signs = [1; -1];
                        for wristIndex = 1:2
                            signValue = signs(wristIndex);
                            q4Rows(wristIndex) = atan2(signValue * rotation36(3, 3), ...
                                signValue * rotation36(1, 3));
                            q6Rows(wristIndex) = atan2(-signValue * rotation36(2, 2), ...
                                signValue * rotation36(2, 1));
                        end
                    end

                    for wristIndex = 1:numel(q4Rows)
                        q4Values = equivalent_angles(q4Rows(wristIndex), robot.limits(4, :), tolerance);
                        q5Values = equivalent_angles(q5Rows(wristIndex), robot.limits(5, :), tolerance);
                        q6Values = equivalent_angles(q6Rows(wristIndex), robot.limits(6, :), tolerance);
                        for q4 = q4Values
                            for q5 = q5Values
                                for q6 = q6Values
                                    candidate = [q1, q2, q3, q4, q5, q6];
                                    solutions = append_if_valid(solutions, candidate, target, robot, ...
                                        shoulderName, elbowName, wristNames(wristIndex), ...
                                        max(tolerance, 1e-7));
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

% 将与上一帧关节角最接近的解析解放在第一项，以维持轨迹连续性。
if ~isempty(solutions)
    distances = zeros(1, numel(solutions));
    for index = 1:numel(solutions)
        distances(index) = norm(solutions(index).q - seed);
    end
    [~, order] = sort(distances);
    solutions = solutions(order);
end
end


function values = equivalent_angles(angle, limits, tolerance)
%EQUIVALENT_ANGLES 返回限位内相差整数个 2*pi 的全部等价角。
period = 2 * pi;
firstTurn = ceil((limits(1) - angle - tolerance) / period);
lastTurn = floor((limits(2) - angle + tolerance) / period);
if firstTurn > lastTurn
    values = zeros(1, 0);
else
    values = angle + period * (firstTurn:lastTurn);
    values = min(limits(2), max(limits(1), values));
end
end


function solutions = append_if_valid(solutions, candidate, target, robot, ...
    shoulderName, elbowName, wristName, tolerance)
%APPEND_IF_VALID 对候选解执行限位、去重和正运动学回代检查。
if any(candidate < robot.limits(:, 1).' - tolerance) || ...
        any(candidate > robot.limits(:, 2).' + tolerance)
    return;
end
for index = 1:numel(solutions)
    if max(abs(solutions(index).q - candidate)) < tolerance
        return;
    end
end

actual = rm65_fk(candidate, robot);
[positionError, orientationError] = rm65_pose_error(actual, target);
if positionError <= tolerance && orientationError <= tolerance
    item.q = candidate;
    item.qDeg = rad2deg(candidate);
    item.shoulder = shoulderName;
    item.elbow = elbowName;
    item.wrist = wristName;
    item.positionError = positionError;
    item.orientationError = orientationError;
    solutions(end + 1) = item; %#ok<AGROW>
end
end


function validate_target(target, tolerance)
%VALIDATE_TARGET 检查输入是否为合法刚体齐次变换矩阵。
if ~isequal(size(target), [4, 4])
    error('RM65:TargetSize', 'target 必须是 4×4 齐次变换矩阵。');
end
rotation = target(1:3, 1:3);
if norm(target(4, :) - [0, 0, 0, 1]) > tolerance
    error('RM65:TargetLastRow', 'target 最后一行必须为 [0, 0, 0, 1]。');
end
if norm(rotation.' * rotation - eye(3), 'fro') > tolerance || ...
        abs(det(rotation) - 1) > tolerance
    error('RM65:TargetRotation', 'target 的旋转部分不是合法旋转矩阵。');
end
end
