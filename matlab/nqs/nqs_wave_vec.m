function [dvec,psi] = nqs_wave_vec(a, b, w)

n = size(a,1);

if n > 15
    disp("ERROR - Vector is getting pretty big there chief... Need N <= 15");
    psi = 0;
    return;
end

dvec = zeros(2^n,1);
psi = zeros(2^n,1);
for d=1:2^n
    dvec(d,1) = d-1;
    sz = -2*get_bi(d-1, n) + 1;
    theta = b + w*sz;
    psi(d,1) = nqs_wave(a, theta, sz);
end

psi = psi/norm(psi);