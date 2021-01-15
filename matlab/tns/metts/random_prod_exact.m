function [evecs, site_evecs, thetavec, phivec] = random_prod_exact(n, pdim)
% random_prod_exact: Select a spin operator with random direction at each
% site and return the orthonormal set of eigenvectors for each of these
% operators to use as a basis for each site's Hilbert space. Also returns
% the full set of product space orthonormal basis vectors

site_evecs = cell(1,n);
thetavec = zeros(1,n);
phivec = zeros(1,n);
evecs = zeros(pdim^n,pdim^n);

for ii=1:n
    [site_evecs{ii}, thetavec(ii), phivec(ii)] = random_spin(pdim);
end

dim = pdim*ones(1,n);
iter = IndexIter(dim);

idx = 1;
while ~iter.end()
    outvec = 1;
    for ii=1:n
        outvec = kron(outvec, site_evecs{ii}(:,iter.curridx(ii)));
    end
    evecs(:,idx) = outvec;
    idx = idx + 1;
    
    iter.next();
end