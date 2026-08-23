function solution = rm65_ik_nearest(target, robot, seed, tolerance)
%RM65_IK_NEAREST 返回最接近 seed 的一组封闭解析逆解。
%   若目标不可达或全部解析分支超出关节限位，则抛出异常。

if nargin < 3 || isempty(seed)
    seed = zeros(1, 6);
end
if nargin < 4
    tolerance = 1e-8;
end
solutions = rm65_ik_all(target, robot, seed, tolerance);
if isempty(solutions)
    error('RM65:Unreachable', '目标位姿不可达，或全部解析分支均超出关节限位。');
end
solution = solutions(1);
end
