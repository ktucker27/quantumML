function [T, Q, alphas, betas] = lanczos_mps(L, R, ops, b, numsteps)
% lanczos_mps: An efficient Lanczos routine for common matrix vector
% operations in MPS algorithms
%
% b is the tensor being operated on with the following indices:
%
% Zero site:
%     ___
% 1__|   |__2
%    |___|
%
% One site:
%     ___
% 1__|   |__2
%    |___|
%      |
%      3
%
% Two site:
%     ______
% 1__|      |__2
%    |______|
%      |  |
%      3  4
%

n = prod(b.dims());
num_sites = size(ops,2);
T = zeros(numsteps+1, numsteps+1);
Q = zeros(n,numsteps+1);
bvec = b.group({1:b.rank()}).A;
Q(:,1) = bvec/norm(bvec);
alphas = zeros(numsteps+1,1);
betas = zeros(numsteps,1);
beta = 0;
for i=1:numsteps+1
    % In the following, order of operations are given in terms of the
    % dimensions:
    %
    % m1 = left bond dimension
    % m2 = right bond dimension
    %  k = MPO bond dimension
    %  d = physical dimension
    %  N = number of sites in b (1 or 2)
    
    % Perform the matrix vector multiplication with an efficient bubbling
    % z = A*Q(:,i); (Regular Lanczos analog)
    b = L.contract(b, [1,1]); % O(m1^2 m2 k d^N)
    
    if num_sites == 0
        b = b.contract(R, [3,1;1,2]); % O(m1 m2^2 k)
    elseif num_sites == 1
        b = b.contract(ops{1}, [1,1;4,3]); % O(m1 m2 k^2 d^2)
        b = b.contract(R, [2,1;3,2]); % O(m1 m2^2 k d)
    else
        b = b.contract(ops{1}, [1,1;4,3]); % O(m1 m2 k^2 d^3)
        b = b.contract(ops{2}, [4,1;3,3]); % O(m1 m2 k^2 d^3)
        b = b.contract(R, [2,1;4,2]); % O(m1 m2^2 k d^2)
    end
    
    if num_sites == 1
        b = b.split({1,3,2});
    elseif num_sites == 2
        b = b.split({1,3,4,2});
    end
    
    bdims = b.dims();
    z = b.group({1:b.rank()}).A;
    alpha = Q(:,i)'*z;
    
    if i == numsteps+1
        break;
    end
    
    z = z - alpha*Q(:,i);
    if i > 1
        z = z - beta*Q(:,i-1);
    end
    beta = norm(z);
    if abs(beta) < 1e-10
        %disp(['WARNING - Found linear dependence on step ', num2str(i)]);
        numsteps = i-1;
        Q = Q(:,1:numsteps+1);
        T = T(1:numsteps+1, 1:numsteps+1);
        break;
    end
    Q(:,i+1) = z/beta;
    
    T(i,i) = alpha;
    T(i,i+1) = beta;
    T(i+1,i) = beta;
    
    alphas(i,1) = alpha;
    betas(i,1) = beta;
    
    b = Tensor(Q(:,i+1));
    b = b.split({[1:(2+num_sites);bdims]});
end

%alpha = Q(:,numsteps+1)'*A*Q(:,numsteps+1);
T(numsteps+1, numsteps+1) = alpha;
alphas(numsteps+1,1) = alpha;