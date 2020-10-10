function pass = tdvp2_test()

pass = 1;

disp('  transverse_ising_test');
if transverse_ising_test() ~= 1
    disp('FAIL: tdvp2_test.transverse_ising_test');
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
TA = Tensor(A);
mps = MPS({TA,TA,TA,TA});

[tvec, mps_out] = tdvp2(mpo, mps, dt, tfinal, svdtol, debug);

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