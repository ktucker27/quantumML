function eloc = exc_eloc(wave, a, b, w, sz, d)
% exc_eloc Local Hamiltonian for the anisotropic exchange model

n = size(a,1);

theta = b + w*sz;
psi_sz = feval(wave, a, theta, sz);

if abs(psi_sz) < 1e-10
    eloc = 0;
    return;
end

eloc = 0;
for i=1:n
    for j=1:n
        if j == i
            continue;
        end
        
        eloc = eloc + site_contrib(i,j,d,wave,a,w,theta,sz);
    end
end

eloc = eloc/psi_sz;
end

function contrib = site_contrib(i,j,d,wave,a,w,theta,sz)
szflip = sz;
szflip(i) = -szflip(i);
szflip(j) = -szflip(j);
psi_szflip = feval(wave, a, theta - 2*w(:,i)*sz(i) - 2*w(:,j)*sz(j), szflip);

% sigma_x
contrib = psi_szflip;
% sigma_y = i*sigma_x*sigma_z
contrib = contrib - sz(i)*sz(j)*psi_szflip;
% sigma_z
%contrib = contrib + sz(i)*sz(j)*psi_sz;
% normalize
contrib = abs(i - j)^(-d)*contrib;
end
