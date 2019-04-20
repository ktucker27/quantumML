function rho = nqs_density(a, theta_l, theta_r, theta_m, szl, szr)

x = prod(cosh(theta_l))*prod(cosh(theta_r));
y = prod(cosh(theta_m));
rho = exp(a.'*szl + a'*szr)*x*y;

if isinf(rho)
    disp('ERROR: Found infinite wave function value');
end