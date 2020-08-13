function [mpo, M] = build_long_range_mpo(ops, pdim, n, rmult, rpow, N)

% Determine the N term expansion for 1/r^rpow
[alpha, beta, ~] = pow_2_exp(rpow, 3, N);
alpha = alpha.*beta; % So that coef. are rmult*(sum_n alpha_n*beta_n^(r-1))

% Build the transfer matrix M based on the automaton
% The operators will be read from right to left, and M(i,j,:,:) is the
% operator that is applied when transitioning from state j to state i
d = 3*N + 2;
M = zeros(d, d, pdim, pdim);
for ii=size(ops,1)
    M(1,1,:,:) = eye(pdim);
    M(d,d,:,:) = eye(pdim);
    for jj=1:N
        stateidx = 1 + (ii-1)*N + jj;
        M(stateidx,1,:,:) = ops{ii,2};
        M(stateidx,stateidx,:,:) = beta(jj)*eye(pdim);
        M(d,stateidx,:,:) = rmult(ii)*alpha(jj)*ops{ii,1};
    end
end

% Assemble the MPO
ms = cell(1,n);
for ii=1:n
    if ii == 1
        ms{ii} = Tensor(M(end,:,:,:));
    elseif ii == n
        ms{ii} = Tensor(M(:,1,:,:));
    else
        ms{ii} = Tensor(M);
    end
end
mpo = MPO(ms);