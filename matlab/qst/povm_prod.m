function [M, local_idx] = povm_prod(type, n)
% povm_prod: Generates the full set of product POVM operators for n sites
% with a last index incremented first ordering, i.e.
% M(:,:,1) = Ml(:,:,1) otimes Ml(:,:,1) otimes ... Ml(:,:,1)
% M(:,:,2) = Ml(:,:,1) otimes Ml(:,:,1) otimes ... Ml(:,:,2)
% etc.
% In this case, the local_idx matrix is populated with the corresponding
% indices
% local_idx(1,:) = [1, 1, ..., 1]
% local_idx(2,:) = [1, 1, ..., 2]
% etc.
% where each line contains n entries. A kronecker product is taken to
% represent the product operators going from left to right. This should be
% consistent with the construction found in local_op_to_prod in the gdtwa
% package

% Get the local POVM
Ml = povm(type);

% Assemble the product using Tensors
d = size(Ml,3);
m = size(Ml,1); % Assumes square POVM operators
M = zeros(m^n, m^n, d^n);
local_idx = zeros(d^n, n);

midx = 1;
iter = IndexIter(d*ones(1,n));
while ~iter.end()
    currm = 1;
    for ii=1:n
        currm = kron(currm, Ml(:,:,iter.curridx(ii)));
    end
    M(:,:,midx) = currm;
    local_idx(midx,:) = iter.curridx;
    midx = midx + 1;
    iter.reverse_next();
end