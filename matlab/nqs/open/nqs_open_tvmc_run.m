% Run script for NQS ground state eigensolver
% To run from the command line, use:
% /path_to_matlab/matlab -nodisplay -nosplash -nodesktop -r "run('/path_to_script/nqs_run.m'), exit"

% Add path to NQS package
addpath('~/git/quantumML/matlab/nqs/');
addpath('~/git/quantumML/matlab/nqs/open');

disp('Starting nqs_open_tvmc_run...');

% Output file
save_filepath = '/data/rey/tuckerkj/20190422/matlab/nqs_open_tvmc_run';

dt = 0.001;

save_time = .002;
num_saves = 500;
tfinal = num_saves*save_time;
time_steps_per_save = save_time/dt;

% Load init file

load /data/rey/tuckerkj/20190422/matlab/tvmc_init

n = size(a,1);
m = size(b,1);
l = size(c,1);

[~,~,rho] = nqs_density_mat(@nqs_density,a,b,c,w,u);
t = 0;

curr_time = 0;
rt = 0;
for i=1:num_saves
    tic();
    [ti,rhoi,yi] = nqs_open_tvmc(a, b, c, w, u, dt, save_time);
    rt = rt + toc();
    rho = cat(3, rho, rhoi(:,:,2:end));
    t = cat(1,t,ti(2:end) + t(end));
    curr_time = curr_time + save_time;
    save(save_filepath);
    disp([num2str(curr_time), '/', num2str(tfinal), ' in ', num2str(rt), ' seconds']);
    [a, b, c, w, u] = unpack_open_tvmc(yi(end,:)',n,m,l);
end
