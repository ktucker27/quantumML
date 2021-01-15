function [eout, site_evecs] = metts(n, pdim, mpo, dbeta, beta, numsteps)
% metts: Performs the METTS fixed point iteration returning energy at each
% step using matrix product states

% Initialize parameters
debug = true;
svdtol = 1e-6;
dt = -1i*dbeta/2;
tfinal = -1i*beta/2;

eout = zeros(1,numsteps);

% Select a random product state to initialize
site_evecs = cell(1,n);
for ii=1:n
    site_evecs{ii} = random_spin(pdim);
end
mps = random_prod(n,pdim,site_evecs);

tic();
for ii=1:numsteps
    % Apply exp(-beta*H/2) and record the energy
    [~, mps_out, eout_it] = tdvp2(mpo, mps, dt, tfinal, svdtol, debug);
    eout(ii) = eout_it(end);
    mps = mps_out{end};

    if debug
        disp(['Energy at step ', num2str(ii), ': ', num2str(eout(ii)), '. Time: ', num2str(toc())]);
    end
    
    % Perform a measurement w.r.t. the product basis given by site_evecs
    mps = measure_prod(mps, site_evecs);
end

if debug
    disp(['Total time: ', num2str(toc())]);
end

if max(abs(imag(eout))) < 1e-15
    eout = real(eout);
else
    disp(['WARNING: Found nonzero imaginary part of energy: ', num2str(max(abs(imag(eout))))]);
end