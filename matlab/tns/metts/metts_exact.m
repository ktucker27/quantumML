function [eout, idxvec] = metts_exact(n, pdim, basis, H, beta, numsteps)
% metts_exact: Performs the pure state method approach to estimating the
% energy of a system in the state rho = exp(-beta*H)/Z

% Initialize
dim = pdim^n;

eout = zeros(1,numsteps);
idxvec = zeros(1,numsteps+1);

% Step 1: Select a random pure state
idx = floor(rand()*dim) + 1;
psi = basis(:,idx);
idxvec(1) = idx;

expmat = expm(-beta*H/2);

for ii=1:numsteps
    % Step 2: Apply exp(-beta*H/2) and record the energy
    phi = expmat*psi;
    phi = phi/norm(phi,2);
    
    eout(ii) = phi'*H*phi;
    
    % Step 3: Perform a measurement according to the distribution
    % |<i|phi>|^2
    eps = rand();
    cdf = 0;
    for jj=1:dim
        cdf = cdf + abs(basis(:,jj)'*phi)^2;
        
        if eps < cdf
            idx = jj;
            break;
        end
    end
    
    psi = basis(:,idx);
    
    idxvec(ii+1) = idx;
end

if max(abs(imag(eout))) < 1e-15
    eout = real(eout);
else
    disp(['WARNING: Found nonzero imaginary part of energy: ', num2str(max(abs(imag(eout))))]);
end