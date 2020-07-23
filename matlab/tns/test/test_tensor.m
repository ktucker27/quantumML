function pass = test_tensor()
pass = 1;

disp('  scalar_contraction');
if scalar_contraction() ~= 1
    disp('FAIL: scalar_contraction');
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