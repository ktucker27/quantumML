function H = get_eloc_mat(eloc, n)
% Requires multiplication by psi_sz in the usual eloc operator

a = zeros(n,1);
b = 0;
w = zeros(1,n);

H = zeros(2^n,2^n);
col_idx = 1;
for d=1:2^n
    %sz = -2*get_bi(d-1, n) + 1;
    psi = zeros(2^n,1);
    psi(d,1) = 1;
    wave_vec = @(a, theta, sz)(kron_wave(psi, a, theta, sz));
    row_idx = 1;
    for d2=1:2^n
        sz2 = -2*get_bi(d2-1, n) + 1;
        H(row_idx, col_idx) = eloc(wave_vec,a,b,w,sz2);
        row_idx = row_idx + 1;
    end
    col_idx = col_idx + 1;
end