function mps = build_init_purification(n, pdim)

% Build a maximally mixed state between a physical site and an auxiliary
% site
psi = zeros(pdim^2,1);
for ii=1:pdim
    sigma = zeros(pdim,1);
    sigma(ii) = 1;
    
    psi = psi + kron(sigma,sigma);
end
psi = (1/sqrt(pdim))*psi;

% Convert to a MPS
site_mps = state_to_mps(psi, 2, pdim);
T1 = site_mps.tensors{1};
T2 = site_mps.tensors{2};

% String the maximally mixed MPS states together, one for each site
ms = cell(1,2*n);
for ii=1:n
    ms{2*(ii-1)+1} = Tensor(T1.A, T1.rank());
    ms{2*(ii-1)+2} = Tensor(T2.A, T2.rank());
end

mps = MPS(ms);