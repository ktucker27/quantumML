function [evecs, theta, phi] = random_spin(pdim)

[~, ~, sz, sx, sy] = local_ops(pdim);

theta = rand()*pi;
phi = rand()*2*pi;

nvec = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)];

if abs(norm(nvec) - 1) > 1e-15
    error('Direction vector does not have unit norm');
end

s = nvec(1)*sx + nvec(2)*sy + nvec(3)*sz;
[evecs, ~] = eig(s);
% TODO - Sort by eigenvalue?