function pass = test_tns()
%TEST_TNS Top level unit test for tns package
%
% OUTPUT: 
%     pass   - 1 if all tests pass, 0 otherwise

pass = 1;

disp('TESTS:');
disp(' ');

disp('tensor_test:');
if tensor_test() ~= 1
    disp('FAIL: tensor_test');
    pass = 0;
end

disp(' ')

disp('mps_test:');
if mps_test() ~= 1
    disp('FAIL: mps_test');
    pass = 0;
end

disp(' ')

disp('mpo_test:');
if mpo_test() ~= 1
    disp('FAIL: mpo_test');
    pass = 0;
end

disp(' ')

if pass == 1
    disp('test_tns PASSED');
else
    disp('test_tns FAILED');
end