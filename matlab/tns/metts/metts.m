function eout = metts(n, pdim, mpo, dbeta, beta, numsteps)
% metts: Performs the METTS fixed point iteration returning energy at each
% step

% Initialize parameters
debug = true;
svdtol = 1e-6;
dt = -1i*dbeta/2;
tfinal = -1i*beta/2;
sz = [1,0;0,-1];

eout = zeros(1,numsteps);

% Select a random product state to initialize
mps = random_prod(n,pdim);

for ii=1:numsteps
    % Apply exp(-beta*H/2) and record the energy
    [~, mps_out, eout_it] = tdvp2(mpo, mps, dt, tfinal, svdtol, debug);
    eout(ii) = eout_it(end);
    mps = mps_out{end};
    
    if debug
        disp(['Energy at step ', num2str(ii), ': ', num2str(eout(ii))]);
    end
    
    % Perform a measurement of sz at all sites
    mps = measure_prod(mps, sz);
end