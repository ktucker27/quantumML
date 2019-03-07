function psi = nqs_wave(a, theta, sz)

% Divided by 2^m to avoid overflow
psi = exp(a.'*sz)*prod(cosh(theta));

if isinf(psi)
    disp('ERROR: Found infinite wave function value');
end