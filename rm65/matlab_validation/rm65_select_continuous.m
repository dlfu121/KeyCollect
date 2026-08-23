function [selected, velocity, totalCost, terms] = rm65_select_continuous( ...
    solutions, previous, previousVelocity, robot, samplePeriod)
%RM65_SELECT_CONTINUOUS 从解析全解中选择跳变风险最低的一支。
%   代价包含关节位移、速度变化、速度超限、分支切换、关节限位软区，
%   以及腕部奇异附近 J4/J6 大幅互相补偿运动的惩罚。

if isempty(solutions)
    error('RM65:NoCandidate', '没有可供连续选择的有效解析解。');
end
maximumSpeed = deg2rad([180, 180, 225, 225, 225, 225]);
singularityZone = deg2rad(5);
softLimitZone = 0.10;

costs = inf(1, numel(solutions));
termList = repmat(struct('jointMotion', 0, 'velocityChange', 0, ...
    'speedExcess', 0, 'branchSwitch', 0, 'jointLimit', 0, ...
    'wristSingularity', 0), 1, numel(solutions));

for index = 1:numel(solutions)
    candidate = solutions(index);
    delta = candidate.q - previous.q;
    candidateVelocity = delta / samplePeriod;
    normalizedStep = delta ./ (maximumSpeed * samplePeriod);

    item.jointMotion = sum(normalizedStep.^2);
    normalizedVelocityChange = (candidateVelocity - previousVelocity) ./ maximumSpeed;
    item.velocityChange = 0.35 * sum(normalizedVelocityChange.^2);
    speedExcess = max(abs(candidateVelocity) ./ maximumSpeed - 1, 0);
    item.speedExcess = 1000 * sum(speedExcess.^2);

    item.branchSwitch = 0;
    if branch_changed(previous.shoulder, candidate.shoulder)
        item.branchSwitch = item.branchSwitch + 60;
    end
    elbowNearSingular = min(abs(previous.q(3)), abs(candidate.q(3))) < singularityZone;
    if ~elbowNearSingular && branch_changed(previous.elbow, candidate.elbow)
        item.branchSwitch = item.branchSwitch + 45;
    end
    wristNearSingular = min(abs(previous.q(5)), abs(candidate.q(5))) < singularityZone;
    if ~wristNearSingular && branch_changed(previous.wrist, candidate.wrist)
        item.branchSwitch = item.branchSwitch + 30;
    end

    jointRanges = robot.limits(:, 2).' - robot.limits(:, 1).';
    lowerFraction = (candidate.q - robot.limits(:, 1).') ./ jointRanges;
    upperFraction = (robot.limits(:, 2).' - candidate.q) ./ jointRanges;
    marginFraction = min(lowerFraction, upperFraction);
    limitIntrusion = max((softLimitZone - marginFraction) / softLimitZone, 0);
    item.jointLimit = 4 * sum(limitIntrusion.^2);

    sineZone = max(sin(singularityZone), 1e-9);
    sineQ5 = max(abs(sin(candidate.q(5))), 1e-9);
    singularityRatio = min(max(sineZone / sineQ5 - 1, 0), 100);
    item.wristSingularity = 2 * singularityRatio^2 * ...
        sum(normalizedStep([4, 6]).^2);

    termList(index) = item;
    costs(index) = item.jointMotion + item.velocityChange + item.speedExcess + ...
        item.branchSwitch + item.jointLimit + item.wristSingularity;
end

[totalCost, selectedIndex] = min(costs);
selected = solutions(selectedIndex);
velocity = (selected.q - previous.q) / samplePeriod;
terms = termList(selectedIndex);
end


function changed = branch_changed(previousName, currentName)
%BRANCH_CHANGED 奇异代表标签不作为普通分支切换处理。
changed = previousName ~= currentName && ...
    ~contains(previousName, "奇异") && ~contains(currentName, "奇异");
end
