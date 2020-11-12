function pass = tdvp_gse_test()

pass = 1;

disp('  transverse_ising_test');
if transverse_ising_test() ~= 1
    disp('FAIL: tdvp_gse_test.transverse_ising_test');
    pass = 0;
end

disp('  oat_test');
if oat_test() ~= 1
    disp('FAIL: tdvp_gse_test.oat_test');
    pass = 0;
end

disp('  thermal_test');
if thermal_test() ~= 1
    disp('FAIL: tdvp_gse_test.thermal_test');
    pass = 0;
end

end

function pass = transverse_ising_test()

pass = 1;

debug = true;

tol = 1e-6;

n = 4;
pdim = 2;
J = 1;
h = 1;
dt = 0.01;
tfinal = 1;
kdim = 3;
eps_vec = [1e-6,1e-8,1e-8,1e-10];

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

one_site_exp = {1i*dt*h*sz,(1/n)*eye(pdim)};
two_site_exp = {{1i*dt*J*sx,sx}};
[mpo_exp, ~] = build_mpo(one_site_exp,two_site_exp,pdim,n);

% Set up rank one initial condition representing psi0
psi0 = [1;zeros(pdim^n-1,1)];
A = zeros(1,1,2);
A(1,1,1) = 1;
ms = cell(1,n);
for ii=1:n
    ms{ii} = Tensor(A);
end
mps = MPS(ms);

tic()
[tvec, mps_out] = tdvp_gse(mpo, mpo_exp, kdim, mps, dt, tfinal, eps_vec, debug);
disp(['Run time (s): ', num2str(toc())]);

evec = zeros(1,size(tvec,2));
dtmat = expm(-1i*H*dt);
psi = psi0;
for ii=1:size(tvec,2)
    psi2 = mps_out{ii}.state_vector();
    phaser = psi(1,1)/psi2(1,1);
    if abs(abs(phaser) - 1) > tol
        disp(['FAIL: Found phaser without unit modulus, error: ', num2str(abs(abs(phaser) - 1))]);
        pass = 0;
        return
    end
    evec(ii) = max(abs(psi - psi2*phaser));
    psi = dtmat*psi;
end

if max(evec) > tol
    disp('FAIL: TDVP state differs from analytical solution');
    pass = 0;
end

end

function pass = oat_test()

pass = 1;

debug = true;

tol = 4e-4;

n = 10;
pdim = 2;
chi = 1;
rmult = 2;
rpow = 0;
N = 3;
dt = 0.01;
tfinal = 1;
kdim = 3;
eps_vec = [1e-6,1e-6,1e-6,1e-6];

% Build the MPO
[~, ~, sz, sx, ~] = local_ops(pdim);
ops = {chi*sz,sz};
lops = {(1/4)*eye(pdim)};
[mpo,~] = build_long_range_mpo(ops,pdim,n,rmult,rpow,N,lops);
mpo_x = build_mpo({sx},{},pdim,n);

% Build the expansion MPO
ops_exp = {-1i*dt*chi*sz,sz};
lops_exp = {(1/n-1i*dt*1/4)*eye(pdim)};
[mpo_exp,~] = build_long_range_mpo(ops_exp,pdim,n,rmult,rpow,N,lops_exp);

% Build the full product space Hamiltonian
csx = zeros(pdim^n, pdim^n);
csz = zeros(pdim^n, pdim^n);
for i=1:n
    [~, ~, szi, sxi] = prod_ops(i, pdim, n);
    csx = csx + sxi;
    csz = csz + szi;
end
H2 = csz*csz;

% H = mpo.matrix();
% if max(max(abs(H - H2))) > tol
%     disp('FAIL: MPO matrix differs from expected matrix');
%     pass = 0;
% end

% Get the initial condition +x
[evecs, evals] = eig(csx);
[~,idx] = max(diag(evals));
psi0 = evecs(:,idx);
%mps = state_to_mps(psi0, n, pdim);
A = ones(1,1,2);
ms = cell(1,n);
for ii=1:n
ms{ii} = Tensor(A);
end
mps = MPS(ms);
mps.right_normalize(eps_vec(2));

% Do the time evolution
tic()
[tvec, mps_out, ~, exp_out] = tdvp_gse(mpo, mpo_exp, kdim, mps, dt, tfinal, eps_vec, debug, [], {mpo_x});
disp(['Run time (s): ', num2str(toc())]);

% Compare evolved state with the exact state
evec = zeros(1,size(tvec,2));
dtmat = expm(-1i*H2*dt);
psi = psi0;
for ii=1:size(tvec,2)
    psi2 = mps_out{ii}.state_vector();
    phaser = psi(1,1)/psi2(1,1);
    if abs(abs(phaser) - 1) > 10*tol
        disp(['FAIL: Found phaser without unit modulus, error: ', num2str(abs(abs(phaser) - 1))]);
        pass = 0;
        return
    end
    evec(ii) = max(abs(psi - psi2*phaser));
    psi = dtmat*psi;
end

if max(evec) > tol
    disp('FAIL: TDVP state differs from analytical solution');
    pass = 0;
end

% Compare S_x value to expected
ex = (n/2)*cos(tvec).^(n-1);
xerr = max(abs(ex - exp_out));
if xerr > 1e-2
    disp(['FAIL: Expected S_x value differs from exptected, error: ', num2str(xerr)]);
end

end

function pass = thermal_test()

pass = 1;

debug = true;

tol = 2e-4;

n = 4;
rpow = 3;
rmult = 1;
N = 2;
pdim = 2;
kdim = 3;
eps_vec = [1e-6,1e-6,1e-6,1e-6];
[~, ~, sz, sx, sy] = local_ops(pdim);

tfinal = -1i;
dt = -0.01*1i;

% Build the Hamiltonian matrix on the physical/auxiliary product space
V = zeros(n,n);
for ii=1:n
    for jj=1:n
        if ii ~= jj
            V(ii,jj) = 1/abs(ii-jj)^3;
        end
    end
end
H = full(thermal_ham(n, pdim, 0, V));

% Build the MPO
ops = {-0.5*sx,sx;-0.5*sy,sy;sz,sz};
mpo = build_purification_mpo(ops,pdim,n,rmult,rpow,N);

% Build the expansion MPO
ops_exp = {1i*abs(dt)*0.5*sx,sx;1i*abs(dt)*0.5*sy,sy;-1i*abs(dt)*sz,sz};
lops_exp = {(1/n)*eye(pdim)};
[mpo_exp] = build_purification_mpo(ops_exp,pdim,n,rmult,rpow,N,lops_exp);

% Perform the imaginary time evolution
mps0 = build_init_purification(n,pdim,1);
tic()
[tvec, ~, eout] = tdvp_gse(mpo, mpo_exp, kdim, mps0, dt, tfinal, eps_vec, debug);
disp(['Run time (s): ', num2str(toc())]);

% Get the exact energy expectations
eout0 = zeros(size(tvec));
for ii=1:size(tvec,2)
    beta = 2*abs(tvec(ii));
    rho = expm(-beta*H);
    rho = rho/trace(rho);
    eout0(ii) = trace(rho*H);
end

% Do the comparison
if max(abs(eout - eout0)) > tol
    disp(['FAIL: Computed energy did not match expected, error: ', num2str(max(abs(eout - eout0)))]);
    pass = 0;
end
end