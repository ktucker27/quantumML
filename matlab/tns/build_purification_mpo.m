function [mpo, M] = build_purification_mpo(ops, pdim, n, rmult, rpow, N, lops)

if nargin < 7
    lops = {};
end

% Determine the N term expansion for 1/r^rpow
[alpha, beta, ~] = pow_2_exp(rpow, 3, N);
ab0 = [alpha,beta];
fun = @(alpha_beta)(exp_loss(alpha_beta, 1.0, rpow, n, N));
options = optimset('MaxFunEvals',10000000);
ab = fminsearch(fun,ab0,options);
alpha = ab(1:N);
beta = ab(N+1:end);
alpha = alpha.*beta; % So that coef. are rmult*(sum_n alpha_n*beta_n^(r-1))

% Build the transfer matrix M based on the automaton
% The operators will be read from right to left, and M(i,j,:,:) is the
% operator that is applied when transitioning from state j to state i
d = 2*size(ops,1)*N + 3;
M = zeros(d, d, pdim, pdim);
M(2,1,:,:) = eye(pdim);
M(1,2,:,:) = eye(pdim);
M(d,d,:,:) = eye(pdim);
for ii=1:size(ops,1)
    for jj=1:N
        stateidx = 2 + (ii-1)*2*N + (jj-1)*2 + 1;
        M(stateidx,2,:,:) = ops{ii,2};
        M(stateidx+1,stateidx,:,:) = eye(pdim);
        M(stateidx,stateidx+1,:,:) = beta(jj)*eye(pdim);
        M(d,stateidx+1,:,:) = rmult*alpha(jj)*ops{ii,1};
    end
end

% Add local operators as a direct transition from the second state to the
% last
for ii=1:size(lops,1)
    M(d,2,:,:) = reshape(M(d,2,:,:),[pdim,pdim]) + lops{ii};
end

% Assemble the MPO
ms = cell(1,2*n);
for ii=1:2*n
    if ii == 1
        ms{ii} = Tensor(M(end,:,:,:),4);
    elseif ii == 2*n
        A = zeros(1,1,pdim,pdim);
        A(1,1,:,:) = eye(pdim);
        ms{ii} = Tensor(A,4);
    elseif ii == 2*n - 1
        ms{ii} = Tensor(M(:,2,:,:),4);
    else
        ms{ii} = Tensor(M,4);
    end
end
mpo = MPO(ms);