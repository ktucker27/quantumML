function [ldist, rdist] = test_open_sampler(rho, a, b, c, w, u, num_samps, num_steps)

%rho = psi*psi';
density_op = @(a, theta_l, theta_r, theta_m, szl, szr)(kron_density(rho, a, theta_l, theta_r, theta_m, szl, szr));
%density_op = @(a, theta_l, theta_r, theta_m, szl, szr)(nqs_density(a, theta_l, theta_r, theta_m, szl, szr));

ldist = {};
rdist = {};

for i=1:num_samps
    [szl, szr, ~] = nqs_open_sample(density_op, a, b, c, w, u, num_steps);
    ldist = proc_sample(ldist, szl);
    rdist = proc_sample(rdist, szr);
end

ldist = norm_dist(ldist, num_samps);
rdist = norm_dist(rdist, num_samps);

print_dist(ldist, 'Left');
print_dist(rdist, 'Right');
end

function dist = proc_sample(dist, sz)
found = 0;
idx = 1;
for i=1:size(dist,1)
    if norm(dist{i,1}.samp - sz) == 0
        dist{i,1}.count = dist{i,1}.count + 1;
        found = 1;
        break;
    end
    idx = idx + 1;
end

if found == 0
    dist{idx,1}.samp = sz;
    dist{idx,1}.count = 1;
end
end

function dist = norm_dist(dist, num_samps)
for i=1:size(dist,1)
    dist{i,1}.prob = dist{i,1}.count/num_samps;
end
end

function print_dist(dist, name)
disp([name, ':']);
for i=1:size(dist,1)
    samp = dist{i,1}.samp;
    disp(['[', num2str(samp(1)), ',', num2str(samp(2)), ']: ', num2str(dist{i,1}.count), ', ', num2str(dist{i,1}.prob)]);
end
disp(' ');
end