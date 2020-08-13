function [alpha, beta, c] = pow_2_exp(a, b, N)

c = ones(1,N);
for kk=1:N-1
    sumval = 0;
    for ii=0:kk-1
        sumval = sumval + c(N-ii)*b^(-a*(kk-ii))*exp(a*(1 - b^(ii-kk)));
    end
    c(N - kk) = 1 - sumval;
end

alpha = zeros(size(c));
beta = zeros(size(c));
for ii=1:size(alpha,2)
    alpha(ii) = c(ii)*(exp(1)/b^(ii-1))^a;
    beta(ii) = exp(-(a*b^(-(ii-1))));
end

% Given a max r as rcutoff, the following approximates 1/r^a
% rvec = 1:rcutoff;
% abfun = zeros(size(rvec));
% for ii=1:size(rvec,2)
% abfun(ii) = sum(alpha.*(beta.^(rvec(ii))));
% end