function V = thermal_coeff(n, d, rpow)

V = zeros(n^d,n^d);

idx_i = IndexIter(n*ones(1,d));
ii = 1;

while ~idx_i.end()
    idx_j = IndexIter(n*ones(1,d));
    jj = 1;
    
    while ~idx_j.end()
        if idx_i.equals(idx_j)
            idx_j.next();
            jj = jj + 1;
            continue;
        end
        
        diff = idx_i.curridx - idx_j.curridx;
        dist = norm(diff,2);
        diff3d = [diff, zeros(1, 3 - d)];
        cosphi = diff3d(3)/dist; % Assumes magnetic field in the z-direction
        
        V(ii,jj) = (1 - 3*cosphi^2)/dist^rpow;
        
        idx_j.next();
        jj = jj + 1;
    end
    
    idx_i.next();
    ii = ii + 1;
end