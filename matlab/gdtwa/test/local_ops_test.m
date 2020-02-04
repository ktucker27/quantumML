function pass = local_ops_test(n, tol)

[sp, sm, sz] = local_ops(n);

pass = 1;

if max(max(abs(sp*sm - sm*sp - 2*sz))) > tol
    pass = 0;
    disp('local_ops_test FAIL: [sp,sm] ~= 2*sz');
end

if max(max(abs(sz*sp - sp*sz - sp))) > tol
    pass = 0;
    disp('local_ops_test FAIL: [sz,sp] ~= sp');
end

if max(max(abs(sz*sm - sm*sz + sm))) > tol
    pass = 0;
    disp('local_ops_test FAIL: [sz,sm] ~= -sm');
end

if pass == 1
    disp(['local_ops_test(n = ', num2str(n), ') PASSED']);
else
    disp(['local_ops_test(n = ', num2str(n), ') FAILED']);
end