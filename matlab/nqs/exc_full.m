function H = exc_full(n, d)
% exc_full Returns the full anisotropic exchange Hamiltonian with power law
%          decaying interaction strength |i-j|^-d

if n > 16
    disp("ERROR - Matrix is getting pretty big there chief... Need N <= 16");
    H = 0;
    return;
end

sx = [0 1;1 0];
sy = [0 -1i;1i 0];

H = zeros(2^n, 2^n);
for i=1:n
    sxi = get_sa(sx, n, i);
    syi = get_sa(sy, n, i);
    
    for j=1:n
        if j == i
            continue;
        end
        
        sxj = get_sa(sx, n, j);
        syj = get_sa(sy, n, j);
        
        H = H + abs(i-j)^(-d)*(sxi*sxj + syi*syj);
    end
end
end
