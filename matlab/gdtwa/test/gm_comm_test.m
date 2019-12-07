function [pass, all_gm, norms] = gm_comm_test(n)
% GM_COMM_TEST A unit test for gm_comm to ensure that the returned
% operators and scalars are correct. This is done by explicitly
% constructing the matrices in question, performing the commutation
% operation, and comparing to the explicit construction of the matrix
% returned by gm_comm.
%
% INPUT:
%     n - Number of levels (2S + 1)
%
% OUTPUT: 
%     pass   - 1 if all tests pass, 0 otherwise
%     all_gm - Full normalized generalized Gell-Mann matrices.
%              all_gm(:,:,i) is the ith n x n GM matrix, i = 1:n^2
%     norms  - norms(i,j) is the norm of the difference between the actual
%              commutator of GM matrices i and j and the one constructed
%              from gm_comm output. Should all be practically zero if all
%              tests pass

TOL = 1e-12;

pass = 1;

d = n^2;

% Build the full Gell-Mann matrices
all_gm = zeros(n,n,d);
for mu = 1:d
    [type, alpha, beta] = gm_idx(mu, n);
    all_gm(:,:,mu) = gm_matrix(type, alpha, beta, n);
end

% Test every pair by taking the commutator of the full matrices and
% comparing it to the full matrix of the commutator returned by gm_comm
norms = zeros(d,d);
for mu = 1:d
    for nu = 1:d
        actual = all_gm(:,:,mu)*all_gm(:,:,nu) - all_gm(:,:,nu)*all_gm(:,:,mu);
        [scalar, idx] = gm_comm(mu, nu, n);
        generated = zeros(n,n);
        for i = 1:size(idx)
            [type, alpha, beta] = gm_idx(idx(i), n);
            generated = generated + scalar(i)*gm_matrix(type, alpha, beta, n);
        end
        norms(mu,nu) = norm(generated - actual);
        
        if norms(mu,nu) > TOL
            disp(['(', num2str(mu), ' ', num2str(nu), ') FAILED']);
        end
    end
end

if norm(norms) > TOL
    pass = 0;
else
    disp(['gm_comm_test(', num2str(n), ') PASSED']);
end