function pass = mpo_test()

pass = 1;

disp('  matrix_test');
if matrix_test() ~= 1
    disp('FAIL: mpo_test.matrix_test');
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