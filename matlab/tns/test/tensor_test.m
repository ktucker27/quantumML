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

disp('  equals_test');
if equals_test() ~= 1
    disp('FAIL: tensor_test.equals_test');
    pass = 0;
end

disp('  group_test');
if group_test() ~= 1
    disp('FAIL: tensor_test.group_test');
    pass = 0;
end

disp('  split_test');
if split_test() ~= 1
    disp('FAIL: tensor_test.split_test');
    pass = 0;
end

disp('  svd_test');
if svd_test() ~= 1
    disp('FAIL: tensor_test.svd_test');
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

function pass = equals_test()

pass = 1;

tol = 1e-12;

v = rand(9,1);
T1 = Tensor(v);
T2 = Tensor(v);

if ~T1.equals(T2, tol)
    disp('FAIL - Equality test with identical tensors failed');
    pass = 0;
end

T2.A(1,1) = T2.A(1,1) + 0.5*tol;
if ~T1.equals(T2, tol)
    disp('FAIL - Equality test with tensors within tolerance failed');
    pass = 0;
end

if T1.equals(T2, 0.25*tol)
    disp('FAIL - Equality test with tensors outside tolerance failed');
    pass = 0;
end

T3 = Tensor([v v]);
if T1.equals(T3, tol)
    disp('FAIL - Equality test with different rank tensors failed');
    pass = 0;
end

end

function pass = group_test()

pass = 1;

tol = 1e-12;

% Test a grouping using reshape
A = rand(3,3);
T1 = Tensor(A);
T2 = T1.group({[1,2]});
v = reshape(A, 9, []);

if ~compare_tensor_matrix(T2, v, tol)
    disp('FAIL: Group reshape test failed');
    pass = 0;
end

% Test a grouping that reshape cannot do
A = rand(3,3,3);
B = zeros(9,3);
for jj=1:3
    B(1:3,jj) = A(:,jj,1);
    B(4:6,jj) = A(:,jj,2);
    B(7:9,jj) = A(:,jj,3);
end

T1 = Tensor(A);
T2 = T1.group({[1,3], 2});

if ~compare_tensor_matrix(T2, B, tol)
    disp('FAIL: Group general test failed');
    pass = 0;
end

end

function pass = split_test()

pass = 1;

tol = 1e-12;

% Test a simple permutation
A = rand(3,4);
T1 = Tensor(A);
T2 = T1.split({2,1});

if ~compare_tensor_matrix(T2, A.', tol)
    disp('FAIL: Split permutation test failed');
    pass = 0;
end

% Test a splitting using reshape
v = rand(9,1);
T1 = Tensor(v);
T2 = T1.split({[1,2;3,3]});
T3 = T1.split({[2,1;3,3]});
A = reshape(v, 3, 3);

if ~compare_tensor_matrix(T2, A, tol)
    disp('FAIL: Split reshape test failed');
    pass = 0;
end

if ~compare_tensor_matrix(T3, A.', tol)
    disp('FAIL: Split reshape and permute test failed');
    pass = 0;
end

% Test a splitting that reshape cannot do
A = rand(6,3);
T1 = Tensor(A);
T2 = T1.split({[1,3;2,3], 2});

B = zeros(2,3,3);
B(:,:,1) = A(1:2,:);
B(:,:,2) = A(3:4,:);
B(:,:,3) = A(5:6,:);

if ~compare_tensor_matrix(T2, B, tol)
    disp('FAIL: Split general test failed');
    pass = 0;
end

end

function pass = svd_test()

pass = 1;

A = rand(10,5);
T1 = Tensor(A);
[TU,TS,TV] = T1.svd();

T2 = TU.contract(TS, [2,1]).contract(TV.conjugate(), [2,2]);

if ~T1.equals(T2, 1e-12)
    disp('FAIL: SVD contraction does not equal original tensor');
    pass = 0;
end

end

function pass = compare_tensor_matrix(T, A, tol)

pass = 1;

% Test rank and dimension
if ~isequal(size(A), size(T.A))
    disp(['FAIL: Expected size [', num2str(size(A)), '], got [', num2str(size(T.A)), ']']);
    pass = 0;
    return
end

% Test values
diffmat = abs(T.A - A);
if max(diffmat(:)) > tol
    disp('FAIL: Tensor values differ from matrix');
    disp('Matrix:');
    disp(A);
    disp('Tensor:');
    disp(T.A);
    pass = 0;
end

% Test Tensor equals
T2 = Tensor(A);
if pass ~= T.equals(T2, tol)
    disp('FAIL - Tensor equality does not match matrix comparison');
    pass = 0;
end
end