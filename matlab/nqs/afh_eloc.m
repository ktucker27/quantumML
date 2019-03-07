function eloc = afh_eloc(wave, a, b, w, sz, nx, ny)
% afh_eloc Local Hamiltonian for the antiferromagnetic Heisenberg model

n = size(sz,1);

if nargin < 7
    nx = n;
    ny = 1;
end

if nx*ny ~= n
    disp('ERROR: Number of lattice sites not equal to the number of particles');
    eloc = 0;
    return;
end

theta = b + w*sz;
psi_sz = feval(wave, a, theta, sz);

if abs(psi_sz) < 1e-10
    eloc = 0;
    return;
end

eloc = 0;
for i=1:nx
    for j=1:ny
        % Get the left site contribution
        left_i = i-1;
        if left_i == 0
            % Periodic boundary conditions
            left_i = nx;
        end
        
        if left_i ~= i
            eloc = eloc + site_contrib(i,j,left_i,j,nx,wave,a,w,theta,sz,psi_sz);
        end
        
        % Get the below site contribution
        below_j = j-1;
        if below_j == 0
            % Periodic boundary conditions
            below_j = ny;
        end
        
        if below_j ~= j
            eloc = eloc + site_contrib(i,j,i,below_j,nx,wave,a,w,theta,sz,psi_sz);
        end
    end
end

eloc = eloc/psi_sz;
end

function contrib = site_contrib(i,j,i2,j2,nx,wave,a,w,theta,sz,psi_sz)
idx1 = site_idx(i,j,nx);
idx2 = site_idx(i2,j2,nx);
szflip = sz;
szflip(idx1) = -szflip(idx1);
szflip(idx2) = -szflip(idx2);
psi_szflip = feval(wave, a, theta - 2*w(:,idx1)*sz(idx1) - 2*w(:,idx2)*sz(idx2), szflip);

% sigma_x
contrib = psi_szflip;
% sigma_y = i*sigma_x*sigma_z
contrib = contrib - sz(idx1)*sz(idx2)*psi_szflip;
% sigma_z
contrib = contrib + sz(idx1)*sz(idx2)*psi_sz;
end

function idx = site_idx(i,j,nx)
% Maps a 2D site index (i,j) to a 1D particle index. Assumes particles are
% in row major order
idx = (j-1)*nx + i;
end