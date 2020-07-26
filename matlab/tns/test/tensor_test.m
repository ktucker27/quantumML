function pass = tensor_test()
pass = 1;

disp('  scalar_contraction');
if scalar_contraction() ~= 1
    disp('FAIL: tensor_test.scalar_contraction');
    pass = 0;
end

disp('  rank_test');
if rank_test() ~= 1
    disp('FAIL: tensor_test.rank_test');
    pass = 0;
end

disp('  group_test');
if group_test() ~= 1
    disp('FAIL: tensor_test.group_test');
    pass = 0;
end

end

function pass = scalar_contraction()

pass = 1;

ta = Tensor(zeros(3,3));
tb = Tensor(zeros(3,3,3));
tc = Tensor(zeros(3,3));
td = Tensor(zeros(3,3,3));
for i=1:3
for j=1:3
ta.A(i,j) = (i-1)^2 - 2*(j-1);
tc.A(i,j) = (j-1);
for k=1:3
tb.A(i,j,k) = -3^(i-1)*(j-1) + (k-1);
td.A(i,j,k) = (i-1)*(j-1)*(k-1);
end
end
end
tab = ta.contract(tb, [2,1]);
tabd = tab.contract(td, [1,1;2,2]);
tabdc = tabd.contract(tc, [1,1;2,2]);

if tabdc.rank() ~= 0
    disp(['FAIL: Expected rank 0 tensor, got ', num2str(tabdc.rank())]);
    pass = 0;
    return
end

if tabdc.A ~= 1080
    disp(['FAIL: Expected value of 1080, got ', num2str(tabdc.A)]);
    pass = 0;
    return
end

end

function pass = rank_test()

pass = 1;

v = 2;
T = Tensor(v);
if T.rank() ~= 0
    disp(['FAIL: Expected rank of scalar to be 0, got ', num2str(T.rank())]);
    pass = 0;
end

v = rand(3,1);
T = Tensor(v);
if T.rank() ~= 1
    disp(['FAIL: Expected rank of column vector to be 1, got ', num2str(T.rank())]);
    pass = 0;
end

v = rand(1,3);
T = Tensor(v);
if T.rank() ~= 2
    disp(['FAIL: Expected rank of row vector to be 2, got ', num2str(T.rank())]);
    pass = 0;
end

v = rand(3,3);
T = Tensor(v);
if T.rank() ~= 2
    disp(['FAIL: Expected rank of matrix to be 2, got ', num2str(T.rank())]);
    pass = 0;
end

v = rand(3,3,3);
T = Tensor(v);
if T.rank() ~= 3
    disp(['FAIL: Expected rank of 3D matrix to be 3, got ', num2str(T.rank())]);
    pass = 0;
end
end

function pass = group_test()

pass = 1;

tol = 1e-12;

A = rand(3,3);
T1 = Tensor(A);
T2 = T1.group({[1,2]});
v = reshape(A, 9, []);

if ndims(v) ~= ndims(T2.A) || norm(size(v) - size(T2.A)) ~= 0
    disp(['FAIL: Expected size [', num2str(size(v)), '], got [', num2str(size(T2.A)), ']']);
    pass = 0;
    return
end

if norm(T2.A - v) > tol
    disp('FAIL: Grouped tensor differs from reshaped matrix');
    disp('Reshaped matrix:');
    disp(v);
    disp('Tensor:');
    disp(T2.A);
    pass = 0;
end

A = rand(3,3,3);
B = zeros(9,3);
for jj=1:3
    B(1:3,jj) = A(:,jj,1);
    B(4:6,jj) = A(:,jj,2);
    B(7:9,jj) = A(:,jj,3);
end

T1 = Tensor(A);
T2 = T1.group({[1,3], 2});

if ndims(B) ~= ndims(T2.A) || norm(size(B) - size(T2.A)) ~= 0
    disp(['FAIL: Expected size [', num2str(size(B)), '], got [', num2str(size(T2.A)), ']']);
    pass = 0;
    return
end

if norm(T2.A - B) > tol
    disp('FAIL: Grouped tensor differs from reshaped matrix');
    disp('Reshaped matrix:');
    disp(B);
    disp('Tensor:');
    disp(T2.A);
    pass = 0;
end

end