function L = get_lloc_mat(lloc, n)
% Requires multiplication by rho1 in the usual lloc operator

a = zeros(n,1);
b = 0;
c = 0;
w = 0;
u = 0;

L = zeros(4^n,4^n);
col_idx = 1;
for dr=1:2^n
    %szr = -2*get_bi(dr-1, n) + 1;
    for dl=1:2^n
        %szl = -2*get_bi(dl-1, n) + 1;
        rho = zeros(2^n,2^n);
        rho(dl,dr) = 1;
        density_op = @(a, b, c, w, u, szl, szr)(kron_density(rho, a, b, c, w, u, szl, szr));
        row_idx = 1;
        for dr2=1:2^n
            szr2 = -2*get_bi(dr2-1, n) + 1;
            for dl2=1:2^n
                szl2 = -2*get_bi(dl2-1, n) + 1;
                L(row_idx, col_idx) = lloc(density_op,a,b,c,w,u,szl2,szr2);
                row_idx = row_idx + 1;
            end
        end
        col_idx = col_idx + 1;
    end
end