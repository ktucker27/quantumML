function eloc = tfi_eloc(wave, a, b, w, sz, h)
% tfi_eloc Local Hamiltonian for the transverse-field Ising model

if nargin < 6
    h = 1;
end

n = size(sz,1);

theta = b + w*sz;

s1 = 0;
s2 = 0;
szflip = sz;
psi_sz = feval(wave, a, theta, sz);

if abs(psi_sz) < 1e-10
    eloc = 0;
    return;
end

for i=1:n
    if i > 1
        s2 = s2 + sz(i)*sz(i-1)*psi_sz;
        szflip(i-1) = -1*szflip(i-1);
    else
        % Periodic boundary conditions
        s2 = s2 + sz(i)*sz(n)*psi_sz;
    end
    
    szflip(i) = -1*szflip(i);
    s1 = s1 + feval(wave, a, theta - 2*w(:,i)*sz(i), szflip);
end

eloc = (-h*s1 - s2)/psi_sz;