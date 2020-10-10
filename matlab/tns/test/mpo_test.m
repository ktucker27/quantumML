function pass = mpo_test()

pass = 1;

disp('  matrix_test');
if matrix_test() ~= 1
    disp('FAIL: mpo_test.matrix_test');
    pass = 0;
end

disp('  long_range_test');
if long_range_test() ~= 1
    disp('FAIL: mpo_test.long_range_test');
    pass = 0;
end

disp('  purification_test');
if purification_test() ~= 1
    disp('FAIL: mpo_test.purification_test');
    pass = 0;
end

disp('  oat_test');
if oat_test() ~= 1
    disp('FAIL: mpo_test.oat_test');
    pass = 0;
end

end

function pass = matrix_test()

pass = 1;

[~, ~, sz, sx, ~] = local_ops(2);
one_site = {-sz};
two_site = {{-sx,sx}};
[mpo, ~] = build_mpo(one_site,two_site,2,4);
H = mpo.matrix();
H2 = build_ham(one_site,two_site,2,4);
if max(max(abs(H - H2))) ~= 0
    disp('FAIL: MPO matrix not equal to expected matrix');
    pass = 0;
end

end

function pass = long_range_test()

pass = 1;

tol = 1e-5;

n = 3;
pdim = 7;
rmult = 1;
rpow = 3;
N = 3;

V = zeros(n,n);
for ii=1:n
for jj=1:n
if ii ~= jj
V(ii,jj) = 1/abs(ii-jj)^3;
end
end
end

H = full(thermal_ham(n, pdim, 0, V));

[~, ~, sz, sx, sy] = local_ops(pdim);
ops = {-0.5*sx,sx;-0.5*sy,sy;sz,sz};
[mpo,~] = build_long_range_mpo(ops,pdim,n,rmult,rpow,N);
H2 = mpo.matrix();

if max(max(abs(H - H2))) > tol
    disp('FAIL: Long range MPO does not match expected matrix');
    pass = 0;
end

end

function pass = oat_test()

pass = 1;

tol = 1e-15;

n = 4;
pdim = 2;
rmult = 2;
rpow = 0;
N = 3;

% Build the full product space Hamiltonian
csz = zeros(pdim^n, pdim^n);
for i=1:n
    [~, ~, szi] = prod_ops(i, pdim, n);
    csz = csz + szi;
end
H = csz*csz;

[~, ~, sz, ~, ~] = local_ops(pdim);
ops = {sz,sz};
lops = {(1/n)*eye(pdim)};
[mpo,~] = build_long_range_mpo(ops,pdim,n,rmult,rpow,N,lops);
H2 = mpo.matrix();

if max(max(abs(H - H2))) > tol
    disp('FAIL: Long range MPO does not match expected matrix');
    pass = 0;
end

end

function pass = purification_test()

pass = 1;

tol = 1e-6;

n = 3;
rpow = 3;
rmult = 1;
N = 3;
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
I = eye(pdim^n);
Hp = kron(H,I);

% Build the MPO
ops = {-0.5*sx,sx;-0.5*sy,sy;sz,sz};
mpo = build_purification_mpo(ops,pdim,n,rmult,rpow,N);

% Contract the MPO and group to reproduce the matrix
for ii=1:2*n
    if ii == 1
        T = mpo.tensors{ii};
    else
        T = T.contract(mpo.tensors{ii}, [T.rank()-2,1]);
    end
end
T = T.squeeze();

T2 = T.group({[2*2*n:-4:4,2*2*n-2:-4:2],[2*2*n-1:-4:3,2*2*n-3:-4:1]});

% Compare the two matrices
if max(max(abs(Hp - T2.A))) > tol
    disp('FAIL: MPO on purification Hilbert space does not match expected matrix');
    pass = 0;
end

end
