% Run script for NQS ground state eigensolver
% To run from the command line, use:
% /path_to_matlab/matlab -nodisplay -nosplash -nodesktop -r "run('/path_to_script/nqs_open_run.m'), exit"

% Add path to NQS package
addpath('~/git/dynasim/matlab/nqs/');

disp('Starting nqs_open_run...');

% Output file
save_filepath = '/Users/tuckerkj/output/20190408/matlab/nqs_open_run';

% Parameters
n = 4;
alpha = 4;
m = alpha*n;
num_samp = 10000;
num_mh_steps = 2*n;
eta = 0.2;

num_save_iter = 10;
num_saves = 10;

% Hamiltonian
h = 1;
eloc = @(wave, a, b, w, sz)(tfi_eloc(wave, a, b, w, sz, h));

% Initialize parameters to random values
rng('shuffle');
a = rand(n,1)*1e-1;
b = rand(m,1)*1e-2;
w = rand(m,n)*1e-2;

% Run
eloc_evs = [];
total_iter = 0;
rt = 0;
save(save_filepath);
for i=1:num_saves
    tic()
    [a, b, w, eloc_evs_update] = nqs_learn(a, b, w, eloc, @nqs_wave, eta, num_save_iter, num_samp, num_mh_steps);
    rt = rt + toc();
    total_iter = total_iter + num_save_iter;
    disp([num2str(total_iter), '/', num2str(num_save_iter*num_saves), ' iterations in ', num2str(rt), ' seconds']);
    
    eloc_evs = [eloc_evs; eloc_evs_update];
    
    save(save_filepath);
end