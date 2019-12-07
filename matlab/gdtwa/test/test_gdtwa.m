function pass = test_gdtwa()
%TEST_GDTWA Top level unit test for gdtwa package
%
% OUTPUT: 
%     pass   - 1 if all tests pass, 0 otherwise

pass = 1;

for n = 1:10
    if gm_comm_test(n) ~= 1
        disp(['FAIL: gm_comm_test n = ', num2str(n)]);
        pass = 0;
    end
end

if pass == 1
    disp('test_gdtwa PASSED');
end