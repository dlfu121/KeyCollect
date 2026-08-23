function report = rm65_batch_test(model, randomCount)
%RM65_BATCH_TEST 无动画批量测试 MATLAB 封闭解析逆解器。
%   report = RM65_BATCH_TEST("RM65-B", 500) 随机生成限位内关节角，
%   通过正运动学得到目标位姿，再检查解析全解和最近解是否正确回代。

if nargin < 1 || isempty(model)
    model = "RM65-B";
end
if nargin < 2 || isempty(randomCount)
    randomCount = 500;
end
robot = rm65_parameters(model);
rng(65, 'twister');

maximumPositionError = 0;
maximumOrientationError = 0;
minimumSolutionCount = inf;
maximumSolutionCount = 0;

for sample = 1:randomCount
    randomRatio = rand(6, 1);
    joints = robot.limits(:, 1) + randomRatio .* ...
        (robot.limits(:, 2) - robot.limits(:, 1));
    joints = joints.';
    target = rm65_fk(joints, robot);
    solutions = rm65_ik_all(target, robot, joints);
    assert(~isempty(solutions), '第 %d 个随机位姿没有返回解析解。', sample);

    nearest = solutions(1);
    actual = rm65_fk(nearest.q, robot);
    [positionError, orientationError] = rm65_pose_error(actual, target);
    assert(positionError < 1e-7, '第 %d 个位姿的位置回代误差过大。', sample);
    assert(orientationError < 1e-7, '第 %d 个位姿的姿态回代误差过大。', sample);
    assert(max(abs(nearest.q - joints)) < 1e-6, ...
        '第 %d 个位姿没有把 seed 对应的解析支路排在第一位。', sample);

    maximumPositionError = max(maximumPositionError, positionError);
    maximumOrientationError = max(maximumOrientationError, orientationError);
    minimumSolutionCount = min(minimumSolutionCount, numel(solutions));
    maximumSolutionCount = max(maximumSolutionCount, numel(solutions));
end

report.model = robot.model;
report.randomCount = randomCount;
report.maximumPositionError = maximumPositionError;
report.maximumOrientationError = maximumOrientationError;
report.minimumSolutionCount = minimumSolutionCount;
report.maximumSolutionCount = maximumSolutionCount;

fprintf('RM65 MATLAB 解析逆解批量验证通过。\n');
fprintf('型号：%s；随机位姿：%d 组。\n', robot.model, randomCount);
fprintf('最大位置误差：%.3e m；最大姿态误差：%.3e rad。\n', ...
    maximumPositionError, maximumOrientationError);
fprintf('有效解析解数量范围：%d～%d。\n', minimumSolutionCount, maximumSolutionCount);
end
