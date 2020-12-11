function eloc = thermal_eloc(wave, a, b, w, sz, V, bq)
% thermal_eloc Local Hamiltonian for the thermal Hamiltonian for a spin-1/2
%              system

if nargin < 7
    bq = 0;
end

n = size(sz,1);

theta = b + w*sz;

xsum = 0;
ysum = 0;
zsum = 0;
zzsum = 0;

psi_sz = feval(wave, a, theta, sz);

if abs(psi_sz) < 1e-10
    eloc = 0;
    return;
end

for i=1:n
    for j=i+1:n
        szflip = sz;
        szflip(i) = -1*szflip(i);
        szflip(j) = -1*szflip(j);
        
        % The 0.25 factor is due to the 1/2 in each spin operator
        xsum = xsum + 0.25*V(i,j)*feval(wave, a, theta - 2*w(:,i)*sz(i) - 2*w(:,j)*sz(j), szflip);
        ysum = ysum - 0.25*V(i,j)*sz(i)*sz(j)*feval(wave, a, theta - 2*w(:,i)*sz(i) - 2*w(:,j)*sz(j), szflip);
        zsum = zsum + 0.25*V(i,j)*sz(i)*sz(j)*psi_sz;
    end
    
    % Add bq*szi*szi term
    zzsum = zzsum + 0.25*bq*psi_sz;
end

eloc = (zsum - 0.5*(xsum + ysum) + zzsum)/psi_sz;