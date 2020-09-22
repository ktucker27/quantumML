function l = exp_loss(alpha_beta, rmult, rpow, rcutoff, N)

% if size(alpha_beta,1) ~= 1
%     error('Expected row vector for alpha_beta');
% end

rvec = 1:rcutoff;
% betamat = repmat(alpha_beta(N+1:end).',1,rcutoff).^repmat(rvec,N,1);
% alpha_beta_mat = repmat(alpha_beta(1:N).',1,rcutoff).*betamat;
% rmat = repmat(rmult*rvec.^(-rpow),N,1);
% 
% l = sum(sum((alpha_beta_mat - rmat).^2));

l = 0;
for jj=1:rcutoff
    sumval = 0;
    for ii=1:N
        sumval = sumval + alpha_beta(ii)*alpha_beta(N+ii)^(rvec(jj));
    end
    l = l + (sumval - rmult/rvec(jj)^rpow)^2;
end
