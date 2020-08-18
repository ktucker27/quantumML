function mps = build_init_purification(n, pdim, bond)

if nargin < 3
    bond = 1;
end

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
A = site_mps.tensors{1}.A;
B = site_mps.tensors{2}.A;
if bond > size(A,2)
    T1 = Tensor(cat(2,A,zeros(size(A,1),bond-size(A,2),size(A,3))));
    T2 = Tensor(cat(1,B,zeros(bond-size(B,1),size(B,2),size(B,3))));
else
    T1 = Tensor(A);
    T2 = Tensor(B);
end

% String the maximally mixed MPS states together, one for each site
ms = cell(1,2*n);
for ii=1:n
    if ii == 1 || bond == 1
        ms{2*(ii-1)+1} = Tensor(T1.A, T1.rank());
    else
        ms{2*(ii-1)+1} = Tensor(cat(1,T1.A,zeros(bond-1,size(T1.A,2),size(T1.A,3))));
    end
    
    if ii == n || bond == 1
        ms{2*(ii-1)+2} = Tensor(T2.A, T2.rank());
    else
        ms{2*(ii-1)+2} = Tensor(cat(2,T2.A,zeros(size(T2.A,1),bond-1,size(T2.A,3))));
    end
end

mps = MPS(ms);