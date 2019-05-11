function oloc = sx_oloc(wave, a, b, w, sz)
% sx_eloc Local operator for collective spin in the x-direction

n = size(a,1);

theta = b + w*sz;
psi_sz = feval(wave, a, theta, sz);

if abs(psi_sz) < 1e-10
    oloc = 0;
    return;
end

oloc = 0;
for i=1:n
    oloc = oloc + site_contrib(i,wave,a,w,theta,sz);
end

oloc = oloc/psi_sz;
end

function contrib = site_contrib(i,wave,a,w,theta,sz)
szflip = sz;
szflip(i) = -szflip(i);
psi_szflip = feval(wave, a, theta - 2*w(:,i)*sz(i), szflip);

% sigma_x
contrib = psi_szflip;
end
