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