% Run script for NQS ground state eigensolver
% To run from the command line, use:
% /path_to_matlab/matlab -nodisplay -nosplash -nodesktop -r "run('/path_to_script/nqs_run.m'), exit"

% Add path to NQS package
addpath('~/git/dynasim/matlab/nqs/');

disp('Starting nqs_run...');

% Output file
%save_filepath = '/Users/tuckerkj/output/20201205/matlab/nqs_run_tfi';
save_filepath = '';

% Parameters
l = 2;
d = 2;
n = l^d;
rpow = 3;
alpha = 8;
m = alpha*n;
num_samp = 1000;
num_mh_steps = 2*10*2*n;
eta = 0.2;

num_save_iter = 10;
num_saves = 20;

% Hamiltonian
% h = 1;
% eloc = @(wave, a, b, w, sz)(tfi_eloc(wave, a, b, w, sz, h));
% H = tfi_full(n,h);

% Use this phi_vec on a 2x2 lattice to converge to a spin-up eigenvector
%phi_vec = [1/2, sqrt(3/4), 0];

phi_vec = [0,0,0];
V = thermal_coeff(l,d,rpow,phi_vec);
H = full(thermal_ham(n, 2, 0, V));
eloc = @(wave, a, b, w, sz)(thermal_eloc(wave, a, b, w, sz, V));

% Initialize parameters to random values
rng('shuffle');
% a = rand(n,1)*1e-1;
% b = rand(m,1)*1e-2;
% w = rand(m,n)*1e-2;

a = (rand(n,1) + 1i*rand(n,1))*1e-1;
b = (rand(m,1) + 1i*rand(m,1))*1e-2;
w = (rand(m,n) + 1i*rand(m,n))*1e-2;

eloc_evs = [];
vars = [];

% Run
total_iter = 0;
rt = 0;
if ~isempty(save_filepath)
    save(save_filepath);
end
for i=1:num_saves
    tic()
    [a, b, w, eloc_evs_update] = nqs_learn(a, b, w, eloc, @nqs_wave, eta, num_save_iter, num_samp, num_mh_steps);
    rt = rt + toc();
    total_iter = total_iter + num_save_iter;
    disp([num2str(total_iter), '/', num2str(num_save_iter*num_saves), ' iterations in ', num2str(rt), ' seconds']);
    
    eloc_evs = cat(1, eloc_evs, eloc_evs_update);
    
    [~,psi] = nqs_wave_vec(a, b, w);
    vars = cat(2, vars, psi'*H*H*psi - (psi'*H*psi)^2);
    
    if ~isempty(save_filepath)
        save(save_filepath);
    end
end