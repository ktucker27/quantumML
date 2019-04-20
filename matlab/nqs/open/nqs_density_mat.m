function [dvecl,dvecr,rho,rho_vec] = nqs_density_mat(a, b, c, w, u)

n = size(a,1);

if n > 15
    disp("ERROR - Matrix is getting pretty big there chief... Need N <= 15");
    rho = 0;
    return;
end

dvecl = zeros(2^n,1);
dvecr = zeros(2^n,1);
rho = zeros(2^n,2^n);
for dl=1:2^n
    dvecl(dl,1) = dl-1;
    szl = -2*get_bi(dl-1, n) + 1;
    for dr=1:2^n
        dvecr(dr,1) = dr-1;
        szr = -2*get_bi(dr-1, n) + 1;
        rho(dl,dr) = nqs_density_z(a, b, c, w, u, szl, szr);
    end
end

rho = rho/trace(rho);
rho_vec = reshape(rho,[4^n,1]);