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

H = full(thermal_ham(n, 0, V));

[~, ~, sz, sx, sy] = local_ops(pdim);
ops = {-0.5*sx,sx;-0.5*sy,sy;sz,sz};
[mpo,~] = build_long_range_mpo(ops,pdim,n,rmult,rpow,N);
H2 = mpo.matrix();

if max(max(abs(H - H2))) > tol
    disp('FAIL: Long range MPO does not match expected matrix');
    pass = 0;
end

end