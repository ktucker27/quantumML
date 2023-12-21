% Run script for NQS ground state eigensolver
% To run from the command line, use:
% /path_to_matlab/matlab -nodisplay -nosplash -nodesktop -r "run('/path_to_script/nqs_tvmc_run.m'), exit"

% Add path to NQS package
addpath('~/git/quantumML/matlab/nqs/');

disp('Starting nqs_tvmc_run...');

% Output file
save_filepath = '/data/rey/tuckerkj/20190506/matlab/nqs_tvmc_exc_run_all_n20_a2';

dt = 0.001;

save_time = .004;
num_saves = 500;
tfinal = num_saves*save_time;
time_steps_per_save = save_time/dt;

% Load init file
%load /data/rey/tuckerkj/20190506/matlab/tvmc_exc_init
%n = size(a,1);
%m = size(b,1);

% Initialize to small random a, b, w
alpha = 2;
n = 20;
m = alpha*n;
a = (rand(n,1) + 1i*rand(n,1))*1e-10;
b = (rand(m,1) + 1i*rand(m,1))*1e-10;
w = (rand(m,n) + 1i*rand(m,n))*1e-10;
ys = [a;b;reshape(w,[m*n,1])];

if n < 16
    [~,psi] = nqs_wave_vec(a, b, w);
else
    psi = [];
end

evs = 0;
t = 0;

curr_time = 0;
rt = 0;
for i=1:num_saves
    tic();
    [a,b,w] = unpack_tvmc(ys(:,end), n, m);
    [ti,psin,ysi,evsi] = nqs_tvmc(a, b, w, dt, save_time);
    rt = rt + toc();
    
    if n < 16
        psi = cat(2, psi, psin(:,2:end));
    end
    
    if i == 1
        evs = evsi;
    else
        evs = cat(2, evs, evsi(:,2:end));
    end
    
    ys = cat(2, ys, ysi(:,2:end));
    t = cat(1,t,ti(2:end) + t(end));
    
    curr_time = curr_time + save_time;
    save(save_filepath);
    disp([num2str(curr_time), '/', num2str(tfinal), ' in ', num2str(rt), ' seconds']);
end
