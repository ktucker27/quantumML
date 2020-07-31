function [mpo, M] = build_mpo(one_site, two_site, pdim, n)

num_one_site = size(one_site,2);
num_two_site = size(two_site,2);

d = num_one_site + num_two_site + 1;

M = zeros(d, d, pdim, pdim);
M(end,end,:,:) = eye(pdim);

for ii=1:num_one_site
    M(ii,1,:,:) = eye(pdim);
    M(end,ii,:,:) = one_site{ii};
end

for ii=1:num_two_site
    M(num_one_site+ii,1,:,:) = two_site{ii}{2};
    M(end,num_one_site+ii,:,:) = two_site{ii}{1};
end

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