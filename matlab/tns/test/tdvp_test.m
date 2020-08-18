function pass = tdvp_test()

pass = 1;

disp('  transverse_ising_test');
if transverse_ising_test() ~= 1
    disp('FAIL: tdvp_test.transverse_ising_test');
    pass = 0;
end

disp('  thermal_test');
if thermal_test() ~= 1
    disp('FAIL: tdvp_test.thermal_test');
    pass = 0;
end

end

function pass = transverse_ising_test()

pass = 1;

debug = true;

tol = 1e-12;

n = 4;
pdim = 2;
J = 1;
h = 1;
dt = 0.01;
tfinal = 1;

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

psi0 = [1;zeros(pdim^n-1,1)];
mps = state_to_mps(psi0, n, pdim);

[tvec, mps_out] = tdvp(mpo, mps, dt, tfinal, debug);

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

function pass = thermal_test()

pass = 1;

debug = true;

tol = 1e-8;

n = 3;
rpow = 3;
rmult = 1;
N = 2;
pdim = 2;
[~, ~, sz, sx, sy] = local_ops(pdim);

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

% Perform the imaginary time evolution
tfinal = -1i;
dt = -0.01*1i;
mps0 = build_init_purification(n,pdim,pdim^n);
[tvec, ~, eout] = tdvp(mpo, mps0, dt, tfinal, debug);

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
    disp('FAIL: Computed energy did not match expected');
    pass = 0;
end
end