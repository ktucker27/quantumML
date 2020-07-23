function pass = test_tns()
%TEST_TNS Top level unit test for tns package
%
% OUTPUT: 
%     pass   - 1 if all tests pass, 0 otherwise

pass = 1;

disp('TESTS:');

disp('test_tensor:');
if test_tensor() ~= 1
    disp('FAIL: test_tensor');
    pass = 0;
end

disp(' ')

if pass == 1
    disp('test_tns PASSED');
else
    disp('test_tns FAILED');
end