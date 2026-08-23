function result = rm65_validate_animation(model, durationSeconds, realtimePlayback)
%RM65_VALIDATE_ANIMATION 以 30 Hz 验证解析逆解并可视化整个运动过程。
%   result = RM65_VALIDATE_ANIMATION("RM65-B", 12, true)
%
%   验证流程：
%   1. 按网页公布的改进 D-H 参数建立 RM65；
%   2. 由平滑关节参考轨迹生成一系列笛卡尔法兰路径点；
%   3. 求解阶段只把目标位姿和上一帧解析解传给逆解器；
%   4. 用正运动学回代，并实时显示机械臂、关节角、角速度和误差。
%
%   realtimePlayback=true 时按墙钟时间保持 30 Hz；设为 false 可快速验证。

if nargin < 1 || isempty(model)
    model = "RM65-B";
end
if nargin < 2 || isempty(durationSeconds)
    durationSeconds = 12;
end
if nargin < 3 || isempty(realtimePlayback)
    realtimePlayback = true;
end

sampleRate = 30;
sampleCount = max(2, round(durationSeconds * sampleRate) + 1);
time = (0:sampleCount-1).' / sampleRate;
phase = time / time(end);
robot = rm65_parameters(model);

% 这些参考关节角仅用于离线生成平滑、闭合且保证可达的笛卡尔路径点。
% 逆解循环不会读取对应时刻的参考关节角，只读取目标齐次位姿。
referenceDeg = zeros(sampleCount, 6);
referenceDeg(:, 1) = 20 + 35 * sin(2*pi*phase);
referenceDeg(:, 2) = -25 + 20 * sin(2*pi*phase + 0.4);
referenceDeg(:, 3) = 45 + 30 * sin(4*pi*phase + 0.2);
referenceDeg(:, 4) = 15 + 60 * sin(2*pi*phase - 0.5);
referenceDeg(:, 5) = -45 + 22 * sin(2*pi*phase + 0.7);
referenceDeg(:, 6) = 40 + 150 * sin(4*pi*phase);

targetPoses = zeros(4, 4, sampleCount);
targetPositions = zeros(sampleCount, 3);
for index = 1:sampleCount
    targetPoses(:, :, index) = rm65_fk(referenceDeg(index, :), robot, true);
    targetPositions(index, :) = targetPoses(1:3, 4, index).';
end

solvedRad = nan(sampleCount, 6);
solvedDeg = nan(sampleCount, 6);
jointVelocityDeg = zeros(sampleCount, 6);
actualPositions = nan(sampleCount, 3);
positionErrors = nan(sampleCount, 1);
orientationErrors = nan(sampleCount, 1);
solutionCounts = zeros(sampleCount, 1);
branchNames = strings(sampleCount, 1);
selectionCosts = zeros(sampleCount, 1);

% 第一帧参考角只用于选定期望的初始几何支路，之后始终使用上一帧解。
seed = deg2rad(referenceDeg(1, :));

figureHandle = figure('Name', 'RM65 解析逆运动学 30 Hz 验证', ...
    'Color', 'w', 'NumberTitle', 'off');
set(figureHandle, 'Position', [80, 80, 1500, 850]);

axisRobot = subplot(3, 2, [1, 3, 5], 'Parent', figureHandle);
hold(axisRobot, 'on'); grid(axisRobot, 'on'); axis(axisRobot, 'equal');
xlabel(axisRobot, 'X / m'); ylabel(axisRobot, 'Y / m'); zlabel(axisRobot, 'Z / m');
title(axisRobot, 'RM65 机械臂、目标路径与实际路径');
view(axisRobot, 42, 25);
xlim(axisRobot, [-0.7, 0.7]); ylim(axisRobot, [-0.7, 0.7]); zlim(axisRobot, [-0.15, 0.85]);

% 绘制底座参考圆和整条目标轨迹。
baseAngle = linspace(0, 2*pi, 100);
plot3(axisRobot, 0.065*cos(baseAngle), 0.065*sin(baseAngle), zeros(size(baseAngle)), ...
    'Color', [0.25, 0.25, 0.25], 'LineWidth', 2);
targetPathLine = plot3(axisRobot, targetPositions(:, 1), targetPositions(:, 2), ...
    targetPositions(:, 3), '--', 'Color', [0.15, 0.65, 0.25], 'LineWidth', 1.2);
actualPathLine = plot3(axisRobot, nan, nan, nan, '-', ...
    'Color', [0.85, 0.20, 0.15], 'LineWidth', 1.8);
robotLine = plot3(axisRobot, nan, nan, nan, '-o', 'Color', [0.05, 0.30, 0.80], ...
    'MarkerFaceColor', [0.90, 0.93, 1.00], 'MarkerSize', 7, 'LineWidth', 4);
targetPoint = plot3(axisRobot, nan, nan, nan, 'p', 'Color', [0.10, 0.60, 0.15], ...
    'MarkerFaceColor', [0.20, 0.85, 0.30], 'MarkerSize', 12);
legend(axisRobot, [targetPathLine, actualPathLine, robotLine, targetPoint], ...
    {'目标法兰路径', '解析解实际路径', '机械臂连杆', '当前目标点'}, 'Location', 'southoutside');

actualTriad = create_triad(axisRobot, 2.5);
targetTriad = create_triad(axisRobot, 1.2);

axisAngle = subplot(3, 2, 2, 'Parent', figureHandle);
hold(axisAngle, 'on'); grid(axisAngle, 'on');
title(axisAngle, '六关节角实时变化'); xlabel(axisAngle, '时间 / s'); ylabel(axisAngle, '角度 / °');
xlim(axisAngle, [0, time(end)]); ylim(axisAngle, [-370, 370]);
colors = lines(6);
angleLines = gobjects(6, 1);
for joint = 1:6
    angleLines(joint) = plot(axisAngle, nan, nan, 'Color', colors(joint, :), 'LineWidth', 1.4);
end
legend(axisAngle, {'J1', 'J2', 'J3', 'J4', 'J5', 'J6'}, ...
    'Location', 'eastoutside', 'NumColumns', 2);

axisVelocity = subplot(3, 2, 4, 'Parent', figureHandle);
hold(axisVelocity, 'on'); grid(axisVelocity, 'on');
title(axisVelocity, '六关节角速度'); xlabel(axisVelocity, '时间 / s'); ylabel(axisVelocity, '角速度 / (°/s)');
xlim(axisVelocity, [0, time(end)]); ylim(axisVelocity, [-250, 250]);
velocityLines = gobjects(6, 1);
for joint = 1:6
    velocityLines(joint) = plot(axisVelocity, nan, nan, 'Color', colors(joint, :), 'LineWidth', 1.2);
end

axisError = subplot(3, 2, 6, 'Parent', figureHandle);
hold(axisError, 'on'); grid(axisError, 'on');
title(axisError, '解析解正运动学回代误差'); xlabel(axisError, '时间 / s');
ylabel(axisError, '对数尺度'); set(axisError, 'YScale', 'log');
xlim(axisError, [0, time(end)]); ylim(axisError, [1e-16, 1e-5]);
positionErrorLine = plot(axisError, nan, nan, 'LineWidth', 1.4, 'Color', [0.85, 0.20, 0.15]);
orientationErrorLine = plot(axisError, nan, nan, 'LineWidth', 1.4, 'Color', [0.45, 0.15, 0.75]);
legend(axisError, {'位置误差 / m', '姿态误差 / rad'}, 'Location', 'eastoutside');

for index = 1:sampleCount
    frameTimer = tic;
    target = targetPoses(:, :, index);
    allSolutions = rm65_ik_all(target, robot, seed);
    if isempty(allSolutions)
        error('RM65:PathPointUnreachable', '第 %d 个路径点没有关节限位内的解析解。', index);
    end
    if index == 1
        solution = allSolutions(1);
        currentVelocityRad = zeros(1, 6);
    else
        [solution, currentVelocityRad, selectionCosts(index)] = ...
            rm65_select_continuous(allSolutions, previousSolution, ...
            previousVelocityRad, robot, 1/sampleRate);
    end
    solutionCounts(index) = numel(allSolutions);
    solvedRad(index, :) = solution.q;
    solvedDeg(index, :) = solution.qDeg;
    branchNames(index) = solution.shoulder + " / " + solution.elbow + " / " + solution.wrist;
    seed = solution.q;
    previousSolution = solution;
    previousVelocityRad = currentVelocityRad;

    [actualPose, frames] = rm65_fk(solution.q, robot);
    actualPositions(index, :) = actualPose(1:3, 4).';
    [positionErrors(index), orientationErrors(index)] = rm65_pose_error(actualPose, target);
    if index > 1
        jointVelocityDeg(index, :) = (solvedDeg(index, :) - solvedDeg(index-1, :)) * sampleRate;
    end

    framePositions = squeeze(frames(1:3, 4, :));
    set(robotLine, 'XData', framePositions(1, :), 'YData', framePositions(2, :), ...
        'ZData', framePositions(3, :));
    set(targetPoint, 'XData', target(1, 4), 'YData', target(2, 4), 'ZData', target(3, 4));
    set(actualPathLine, 'XData', actualPositions(1:index, 1), ...
        'YData', actualPositions(1:index, 2), 'ZData', actualPositions(1:index, 3));
    update_triad(actualTriad, actualPose, 0.055);
    update_triad(targetTriad, target, 0.075);

    for joint = 1:6
        set(angleLines(joint), 'XData', time(1:index), 'YData', solvedDeg(1:index, joint));
        set(velocityLines(joint), 'XData', time(1:index), 'YData', jointVelocityDeg(1:index, joint));
    end
    set(positionErrorLine, 'XData', time(1:index), ...
        'YData', max(positionErrors(1:index), eps));
    set(orientationErrorLine, 'XData', time(1:index), ...
        'YData', max(orientationErrors(1:index), eps));
    title(axisRobot, sprintf('RM65 解析运动｜t=%.2f s｜%s｜当前有效解 %d 组', ...
        time(index), branchNames(index), solutionCounts(index)));
    drawnow;

    % 扣除解析计算和绘图耗时，使帧起始频率尽量保持在 30 Hz。
    if realtimePlayback && index < sampleCount
        pause(max(0, 1/sampleRate - toc(frameTimer)));
    end
    if ~isvalid(figureHandle)
        break;
    end
end

result.model = robot.model;
result.sampleRate = sampleRate;
result.time = time;
result.targetPoses = targetPoses;
result.referenceJointDeg = referenceDeg;
result.solvedJointDeg = solvedDeg;
result.jointVelocityDegPerSecond = jointVelocityDeg;
result.actualPositions = actualPositions;
result.positionError = positionErrors;
result.orientationError = orientationErrors;
result.solutionCount = solutionCounts;
result.branch = branchNames;
result.selectionCost = selectionCosts;

fprintf('验证完成：%d 个路径点，显示频率 %d Hz。\n', sampleCount, sampleRate);
fprintf('最大位置回代误差：%.3e m\n', max(positionErrors));
fprintf('最大姿态回代误差：%.3e rad\n', max(orientationErrors));
fprintf('每个路径点的有效解析解数量范围：%d～%d\n', min(solutionCounts), max(solutionCounts));
end


function handles = create_triad(parentAxis, lineWidth)
%CREATE_TRIAD 创建末端坐标系的三个彩色轴句柄。
axisColors = [0.90, 0.10, 0.10; 0.10, 0.65, 0.15; 0.10, 0.30, 0.90];
handles = gobjects(3, 1);
for axisIndex = 1:3
    handles(axisIndex) = plot3(parentAxis, nan, nan, nan, '-', ...
        'Color', axisColors(axisIndex, :), 'LineWidth', lineWidth);
end
end


function update_triad(handles, transform, scale)
%UPDATE_TRIAD 更新末端坐标系三个轴的显示位置。
origin = transform(1:3, 4);
for axisIndex = 1:3
    endpoint = origin + scale * transform(1:3, axisIndex);
    set(handles(axisIndex), 'XData', [origin(1), endpoint(1)], ...
        'YData', [origin(2), endpoint(2)], 'ZData', [origin(3), endpoint(3)]);
end
end
