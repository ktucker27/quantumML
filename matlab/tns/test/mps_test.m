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

disp('  equals_test');
if equals_test() ~= 1
    disp('FAIL: mps_test.equals_test');
    pass = 0;
end

disp('  normal_test');
if normal_test() ~= 1
    disp('FAIL: mps_test.normal_test');
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

% Test periodic boundary conditions
if n >= 4
    mps2 = mps.substate([2,3]);
    psi = mps2.state_vector();
    val = mps2.inner(mps2);
    if abs(psi'*psi - val) > tol
        disp(['FAIL: Inner product for PBC returned ', num2str(val), ' expected ', num2str(psi'*psi)]);
        pass = 0;
    end
end

end

function pass = equals_test()

pass = 1;

tol = 1e-12;

n = 4;
pdim = 3;

psi = rand(pdim^n,1);
psi = psi/norm(psi);
mps = state_to_mps(psi, n, pdim);
mps2 = mps.substate(1:n);

if ~mps.equals(mps2, tol)
    disp('FAIL - Equality test with identical MPS failed');
    pass = 0;
end

mps2.tensors{2}.set([1,2,1], mps2.tensors{2}.get([1,2,1]) + 0.5*tol);
if ~mps.equals(mps2, tol)
    disp('FAIL - Equality test with MPS within tolerance failed');
    pass = 0;
end

if mps.equals(mps2, 0.25*tol)
    disp('FAIL - Equality test with tensors outside tolerance failed');
    pass = 0;
end

mps2 = mps.substate(2:n-1);
if mps.equals(mps2, tol)
    disp('FAIL - Equality test with MPS of different sizes failed');
    pass = 0;
end
end

function pass = normal_test()

pass = 1;

n = 6;
pdim = 2;
tol = 1e-12;

psi = rand(pdim^n,1);
psi = psi/norm(psi);
mps = state_to_mps(psi, n, pdim);

% Initial MPS should be in left normal form
if ~mps.is_left_normal(tol)
    disp('FAIL: Expected left normal MPS after initialization from vector');
    pass = 0;
end

if mps.is_right_normal(tol)
    disp('FAIL: Found unexpected right normal MPS');
    pass = 0;
end

% Right normalize and check
mps.right_normalize();
if ~mps.is_right_normal(tol)
    disp('FAIL: Expected right normal MPS after normalization');
    pass = 0;
end

if mps.is_left_normal(tol)
    disp('FAIL: Found unexpected left normal MPS');
    pass = 0;
end

psi2 = mps.state_vector();
phaser = psi(1,1)/psi2(1,1);
if abs(abs(phaser) - 1) > tol
    disp(['FAIL: Found phaser without unit modulus after right normalize, error: ', num2str(abs(abs(phaser) - 1))]);
    pass = 0;
    return
end

if max(abs(psi - phaser*psi2)) > tol
    disp('FAIL: Right normalized MPS did not preserve state vector');
    pass = 0;
end

% Return to left normal
mps.left_normalize();

if ~mps.is_left_normal(tol)
    disp('FAIL: Expected left normal MPS after renormalization');
    pass = 0;
end

if mps.is_right_normal(tol)
    disp('FAIL: Found unexpected right normal MPS after renormalization');
    pass = 0;
end

phaser = psi(1,1)/psi2(1,1);
if abs(abs(phaser) - 1) > tol
    disp(['FAIL: Found phaser without unit modulus after left normalize, error: ', num2str(abs(abs(phaser) - 1))]);
    pass = 0;
    return
end

psi2 = mps.state_vector();
if max(abs(psi - phaser*psi2)) > tol
    disp('FAIL: Left normalized MPS did not preserve state vector');
    pass = 0;
end

end