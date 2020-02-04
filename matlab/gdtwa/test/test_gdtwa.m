function pass = test_gdtwa()
%TEST_GDTWA Top level unit test for gdtwa package
%
% OUTPUT: 
%     pass   - 1 if all tests pass, 0 otherwise

TOL = 1e-14;

pass = 1;

for n = 2:10
    disp(['n = ', num2str(n), ':']);
    
    if gm_comm_test(n) ~= 1
        disp(['FAIL: gm_comm_test n = ', num2str(n)]);
        pass = 0;
    end
    
    if gm_mult_test(n) ~= 1
        disp(['FAIL: gm_mult_test n = ', num2str(n)]);
        pass = 0;
    end
    
    if local_ops_test(n, TOL) ~= 1
        disp(['FAIL: local_ops_test n = ', num2str(n)]);
        pass = 0;
    end
end

for n=2:4
    for m=2:4
        if gm_expand_test(n,m) ~= 1
            pass = 0;
        end
    end
end

disp(' ')

if pass == 1
    disp('test_gdtwa PASSED');
else
    disp('test_gdtwa FAILED');
end