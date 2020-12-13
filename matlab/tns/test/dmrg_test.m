function pass = dmrg_test()

pass = 1;

disp('  transverse_ising_test');
if transverse_ising_test() ~= 1
    disp('FAIL: dmrg_test.transverse_ising_test');
    pass = 0;
end

end

function pass = transverse_ising_test()

pass = 1;

tol = 1e-12;
maxit = 20;

n = 6;
pdim = 2;
J = 1;
h = 1;

[~, ~, sz, sx, ~] = local_ops(pdim);
one_site = {-h*sz};
two_site = {{-J*sx,sx}};

[mpo, ~] = build_mpo(one_site,two_site,pdim,n);
H = mpo.matrix();
H2 = build_ham(one_site,two_site,pdim,n);
if max(max(abs(H - H2))) > tol
    disp('FAIL: MPO matrix differs from expected matrix');
    pass = 0;
end

[evecs, evals] = eig(H);
[e0, min_idx] = min(diag(evals));
psi0 = evecs(:,min_idx);
psi = rand(pdim^n,1) + 1i*rand(pdim^n,1);
psi = psi/norm(psi);

mps = state_to_mps(psi, n, pdim);
mps_out = dmrg(mpo, mps, tol, maxit);
psi_out = mps_out.state_vector();

phaser = psi_out(1,1)/psi0(1,1);
if abs(abs(phaser) - 1) > tol
    disp(['FAIL: Converged state vector not equal to ground state, error: ', num2str(abs(abs(phaser) - 1))]);
    pass = 0;
end

if max(abs(psi0 - psi_out/phaser)) > tol
    disp(['FAIL: Phased converged state vector not equal to ground state, error: ', num2str(max(abs(psi0 - psi_out/phaser)))]);
    pass = 0;
end

if abs(mps_out.inner(mps_out) - 1) > tol
    disp('FAIL: Output eigenvector is not norm 1');
    pass = 0;
end

e = mps_out.inner(apply_mpo(mpo, mps_out));
if abs(e - e0) > tol
    disp(['FAIL: Output eigenvalue does not match ground state, error: ', num2str(abs(e - e0))]);
    pass = 0;
end

end