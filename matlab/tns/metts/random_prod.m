function mps = random_prod(n, pdim, site_evecs)
% random_prod: Creates a rank one MPS that is a tensor product of randomly
% selected basis vectors at each site given in site_evecs, thereby 
% selecting a random member of the product basis

ms = cell(1,n);
for ii=1:n
    % Select a basis vector for this site's Hilbert space at random
    eps = rand();
    idx = floor(eps*pdim) + 1;
    
    % Create the Tensor for this site
    ms{ii} = Tensor(reshape(site_evecs{ii}(:,idx), 1, 1, []));
end

mps = MPS(ms);