function pass = mps_test()
pass = 1;

disp('  eval_test');
if eval_test(6, 3) ~= 1
    disp('FAIL: mps_test.eval_test');
    pass = 0;
end

end

function pass = eval_test(n, pdim)

pass = 1;

tol = 1e-12;

psi = rand(pdim^n,1);
psi = psi/norm(psi);

mps = state_to_mps(psi, n, pdim);

iter = IndexIter(pdim*ones(1,n));
psi2 = zeros(pdim^n,1);
ii = 1;
while ~iter.end()
psi2(ii,1) = mps.eval(flip(iter.curridx));
ii = ii + 1;
iter.next();
end

if norm(psi - psi2) > tol
    disp('FAIL: MPS state does not match original state');
    pass = 0;
end

end