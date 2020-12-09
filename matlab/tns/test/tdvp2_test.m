function pass = tdvp2_test()

pass = 1;

disp('  transverse_ising_test');
if transverse_ising_test() ~= 1
    disp('FAIL: tdvp2_test.transverse_ising_test');
    pass = 0;
end

disp('  oat_test');
if oat_test() ~= 1
    disp('FAIL: tdvp2_test.oat_test');
    pass = 0;
end

disp('  thermal_rt_test');
if thermal_rt_test() ~= 1
    disp('FAIL: tdvp2_test.thermal_rt_test');
    pass = 0;
end

disp('  thermal_rt_test_spin3');
if thermal_rt_test_spin3() ~= 1
    disp('FAIL: tdvp2_test.thermal_rt_test_spin3');
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
svdtol = 1e-15;

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
[tvec, mps_out] = tdvp2(mpo, mps, dt, tfinal, svdtol, debug);
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
svdtol = 1e-6;

% Build the MPO
[~, ~, sz, sx, ~] = local_ops(pdim);
ops = {chi*sz,sz};
lops = {(1/4)*eye(pdim)};
[mpo,~] = build_long_range_mpo(ops,pdim,n,rmult,rpow,N,lops);
mpo_x = build_mpo({sx},{},pdim,n);

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

% Do the time evolution
tic()
[tvec, mps_out, ~, exp_out] = tdvp2(mpo, mps, dt, tfinal, svdtol, debug, [], {mpo_x});
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
    pass = 0;
end

end

function pass = thermal_rt_test()

pass = 1;

debug = true;

tol = 5e-5;

n = 4;
rpow = 3;
rmult = 1;
N = 2;
pdim = 2;
dt = 0.01;
tfinal = 1;
svdtol = 1e-15;

[~, ~, sz, sx, sy] = local_ops(pdim);

% Build the Hamiltonian matrix on the physical/auxiliary product space
V = zeros(n,n);
csx = zeros(pdim^n, pdim^n);
csy = zeros(pdim^n, pdim^n);
csz = zeros(pdim^n, pdim^n);
for ii=1:n
    for jj=1:n
        if ii ~= jj
            V(ii,jj) = 1/abs(ii-jj)^3;
        end
    end
    
    [~, ~, szi, sxi, syi] = prod_ops(ii, pdim, n);
    csx = csx + sxi;
    csy = csy + syi;
    csz = csz + szi;
end
H = full(thermal_ham(n, pdim, 0, V));
s2 = csx*csx + csy*csy + csz*csz;

% Build the MPO
ops = {-0.5*sx,sx;-0.5*sy,sy;sz,sz};
mpo = build_long_range_mpo(ops,pdim,n,rmult,rpow,N);

% Build observables to check expectations
mpo_x = build_mpo({sx},{},pdim,n);

ops_s = {sx,sx;sy,sy;sz,sz};
lops_s = {(3/4)*eye(pdim)};
mpo_s2 = build_long_range_mpo(ops_s,pdim,n,2,0,1,lops_s);

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

tic();
[tvec, mps_out, ~, exp_out] = tdvp2(mpo, mps, dt, tfinal, svdtol, debug, [], {mpo_x, mpo_s2});
disp(['Run time (s): ', num2str(toc())]);

% Compare evolved state with the exact state
evec = zeros(1,size(tvec,2));
ex = zeros(2,size(tvec,2));
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
    ex(1,ii) = psi'*csx*psi;
    ex(2,ii) = psi'*s2*psi;
    psi = dtmat*psi;
end

if max(evec) > tol
    disp(['FAIL: TDVP state differs from analytical solution, error: ', num2str(max(evec))]);
    pass = 0;
end

% Compare S_x value to expected
xerr = max(abs(ex(1,:) - exp_out(1,:)));
if xerr > tol
    disp(['FAIL: Expected S_x value differs from exptected, error: ', num2str(xerr)]);
    pass = 0;
end

% Compare S^2 value to expected
s2err = max(abs(ex(2,:) - exp_out(2,:)));
if s2err > tol
    disp(['FAIL: Expected S^2 value differs from exptected, error: ', num2str(s2err)]);
    pass = 0;
end

end

function pass = thermal_rt_test_spin3()

pass = 1;

debug = true;

tol = 1e-3;

n = 3;
rpow = 3;
rmult = 1;
N = 2;
pdim = 7;
dt = 0.01;
tfinal = 1;
svdtol = 1e-6;

[~, ~, sz, sx, sy] = local_ops(pdim);

% Build the Hamiltonian matrix on the physical/auxiliary product space
V = zeros(n,n);
csx = zeros(pdim^n, pdim^n);
csy = zeros(pdim^n, pdim^n);
csz = zeros(pdim^n, pdim^n);
for ii=1:n
    for jj=1:n
        if ii ~= jj
            V(ii,jj) = 1/abs(ii-jj)^3;
        end
    end
    
    [~, ~, szi, sxi, syi] = prod_ops(ii, pdim, n);
    csx = csx + sxi;
    csy = csy + syi;
    csz = csz + szi;
end
H = full(thermal_ham(n, pdim, 0, V));
s2 = csx*csx + csy*csy + csz*csz;

% Build the MPO
ops = {-0.5*sx,sx;-0.5*sy,sy;sz,sz};
mpo = build_long_range_mpo(ops,pdim,n,rmult,rpow,N);

% Build observables to check expectations
mpo_x = build_mpo({sx},{},pdim,n);

ops_s = {sx,sx;sy,sy;sz,sz};
lops_s = {sx*sx;sy*sy;sz*sz};
mpo_s2 = build_long_range_mpo(ops_s,pdim,n,2,0,1,lops_s);

% Get the initial condition +x
[evecs, evals] = eig(csx);
[~,idx] = max(diag(evals));
psi0 = evecs(:,idx);
%mps = state_to_mps(psi0, n, pdim);

[evecs, evals] = eig(sx);
[~,idx] = max(diag(evals));
max_evec = evecs(:,idx);
A = ones(1,1,pdim);
for ii=1:pdim
    A(:,:,ii) = max_evec(ii);
end
ms = cell(1,n);
for ii=1:n
    ms{ii} = Tensor(A);
end
mps = MPS(ms);

tic();
[tvec, mps_out, ~, exp_out] = tdvp2(mpo, mps, dt, tfinal, svdtol, debug, [], {mpo_x, mpo_s2});
disp(['Run time (s): ', num2str(toc())]);

% Compare evolved state with the exact state
evec = zeros(1,size(tvec,2));
ex = zeros(2,size(tvec,2));
dtmat = expm(-1i*H*dt);
psi = psi0;
for ii=1:size(tvec,2)
    psi2 = mps_out{ii}.state_vector();
    phaser = psi(1,1)/psi2(1,1);
    if abs(abs(phaser) - 1) > tol
        disp(['FAIL: Found phaser without unit modulus, error: ', num2str(abs(abs(phaser) - 1))]);
        pass = 0;
        %return
    end
    evec(ii) = max(abs(psi - psi2*phaser));
    ex(1,ii) = psi'*csx*psi;
    ex(2,ii) = psi'*s2*psi;
    psi = dtmat*psi;
end

if max(evec) > tol
    disp(['FAIL: TDVP state differs from analytical solution, error: ', num2str(max(evec))]);
    pass = 0;
end

% Compare S_x value to expected
xerr = max(abs(ex(1,:) - exp_out(1,:)));
if xerr > tol
    disp(['FAIL: Expected S_x value differs from exptected, error: ', num2str(xerr)]);
    pass = 0;
end

% Compare S^2 value to expected
s2err = max(abs(ex(2,:) - exp_out(2,:)));
if s2err > tol
    disp(['FAIL: Expected S^2 value differs from exptected, error: ', num2str(s2err)]);
    pass = 0;
end

end
