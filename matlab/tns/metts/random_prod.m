function mps = random_prod(n, pdim)
% random_prod: Creates a rank one MPS that is a tensor product of randomly
% selected canonical basis vectors at each site, thereby selecting a random
% member of the product basis

ms = cell(1,n);
for ii=1:n
    % Select a canonical basis vector for this site's Hilbert space at
    % random
    eps = rand();
    if eps == 1
        error('Found eps = 1 in random_prod');
    end
    idx = floor(eps*pdim) + 1;
    
    % Create the Tensor for this site
    A = zeros(1,1,pdim);
    A(1,1,idx) = 1;
    ms{ii} = Tensor(A);
end

mps = MPS(ms);