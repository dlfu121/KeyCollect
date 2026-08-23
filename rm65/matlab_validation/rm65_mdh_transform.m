function transform = rm65_mdh_transform(a, alpha, d, theta)
%RM65_MDH_TRANSFORM 计算一行改进 D-H 参数对应的齐次变换。
%   T = Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)

ca = cos(alpha);
sa = sin(alpha);
ct = cos(theta);
st = sin(theta);

transform = [ct,      -st,       0,      a; ...
             ca*st,   ca*ct,   -sa,  -d*sa; ...
             sa*st,   sa*ct,    ca,   d*ca; ...
             0,           0,     0,      1];
end
