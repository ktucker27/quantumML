function pass = local_ops_test(n, tol)

[sp, sm, sz, sx, sy] = local_ops(n);

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

if max(max(abs(sx*sy - sy*sx - 1i*sz))) > tol
    pass = 0;
    disp('local_ops_test FAIL: [sx,sy] ~= 1i*sz');
end

if max(max(abs(sy*sz - sz*sy - 1i*sx))) > tol
    pass = 0;
    disp('local_ops_test FAIL: [sy,sz] ~= 1i*sx');
end

if max(max(abs(sz*sx - sx*sz - 1i*sy))) > tol
    pass = 0;
    disp('local_ops_test FAIL: [sz,sx] ~= 1i*sy');
end

if pass == 1
    disp(['local_ops_test(n = ', num2str(n), ') PASSED']);
else
    disp(['local_ops_test(n = ', num2str(n), ') FAILED']);
end