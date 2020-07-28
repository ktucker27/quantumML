function pass = mps_test()
pass = 1;

disp('  eval_test');
if eval_test(6, 3) ~= 1
    disp('FAIL: mps_test.eval_test');
    pass = 0;
end

disp('  conj_test');
if conj_test(4, 3) ~= 1
    disp('FAIL: mps_test.conj_test');
    pass = 0;
end

disp('  inner_test');
if inner_test(4, 2) ~= 1
    disp('FAIL: mps_test.inner_test');
    pass = 0;
end

end

function pass = eval_test(n, pdim)

pass = 1;

tol = 1e-12;

psi = rand(pdim^n,1);
psi = psi/norm(psi);

mps = state_to_mps(psi, n, pdim);

psi2 = mps.state_vector();

if norm(psi - psi2) > tol
    disp('FAIL: MPS state does not match original state');
    pass = 0;
end

end

function pass = conj_test(n, pdim)

pass = 1;

tol = 1e-12;

psi = rand(pdim^n,1) + 1i*rand(pdim^n,1);
psi = psi/norm(psi);

mps = state_to_mps(psi, n, pdim);
mps = mps.dagger();

psi2 = mps.state_vector();

if norm(psi - conj(psi2)) > tol
    disp('FAIL: Conjugate MPS state does not match original state');
    pass = 0;
end

end

function pass = inner_test(n, pdim)

pass = 1;

tol = 1e-12;

psi = rand(pdim^n,1) + 1i*rand(pdim^n,1);
psi2 = rand(pdim^n,1) + 1i*rand(pdim^n,1);

mps = state_to_mps(psi, n, pdim);
mps2 = state_to_mps(psi2, n, pdim);

val = mps.inner(mps2);

if abs(psi2'*psi - val) > tol
    disp(['FAIL: Inner product returned ', num2str(val), ' expected ', num2str(psi2'*psi)]);
    pass = 0;
end

end