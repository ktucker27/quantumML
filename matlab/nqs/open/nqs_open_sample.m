function [szl, szr, qsz] = nqs_open_sample(density, a, b, c, w, u, num_steps)
% nqs_sample_open Generate a vector of spins sampled from a distribution
%                 governed by the NQS density determined by the set of
%                 parameters (a, b, c, w, u)
%
% INPUT:     
%            a - Nx1 vector of biases for the visible variables 
%            b - Mx1 vector of biases for the hidden variables
%            c - Lx1 vector of biases for the mixing layer
%            w - MxN matrix of weights between hidden and visible variables
%            u - LxN matrix of weights between mixing and visible variables
%    num_steps - Number of steps to take in the Markov process
%
% OUTPUT:   
%           szl - Vector of left spins sampled according to the density
%                 operator given by the parameters (a, b, c, W, U)
%           szr - Vector of right spins sampled according to the density
%                 operator given by the parameters (a, b, c, W, U)
%           qsz - Vector of spins sampled according to the diagonal of the 
%                 density operator given by the parameters (a, b, c, W, U)

n = size(w,2);

% Initialize to spin-up
szl = ones(n,1);
szr = ones(n,1);
qsz = ones(n,1);

% Initialize theta
theta_l = b + w*szl;
theta_r = conj(b) + conj(w)*szr;
theta_m = c + conj(c) + u*szl + conj(u)*szr;
qtheta = b + w*qsz;
qtheta_m = c + conj(c) + u*qsz + conj(u)*qsz;

% Walk to a state with nonzero probability. This randomly moves around a
% column of the desnity operator TODO - Explore more of the operator
while density(a, theta_l, theta_r, theta_m, szl, szr) == 0
    flip_idx = ceil(n*rand());
    szl(flip_idx) = -1*szl(flip_idx);
    
    theta_l = theta_l - 2*w(:,flip_idx)*szl(flip_idx);
    theta_m = theta_m - 2*u(:,flip_idx)*szl(flip_idx);
end

% Walk to a state with nonzero probability for the diaganol distribution
while density(a, qtheta, conj(qtheta), qtheta_m, qsz, qsz) == 0
    flip_idx = ceil(n*rand());
    qsz(flip_idx) = -1*qsz(flip_idx);
    
    qtheta = qtheta - 2*w(:,flip_idx)*qsz(flip_idx);
    qtheta_m = qtheta_m - 2*u(:,flip_idx)*qsz(flip_idx) - 2*conj(u(:,flip_idx))*qsz(flip_idx);
end

% Perform a random walk on the spins according to Metropolis-Hastings,
% where the simple transitions are taken to be a single spin flip with
% equal probability
for step_idx=1:num_steps
    % Randomly select a spin to flip
    if rand() < 0.5
        % Left
        [new_szl, new_theta_l, new_theta_m] = flip_one(szl, theta_l, theta_m, w, u);
        new_szr = szr;
        new_theta_r = theta_r;
    else
        % Right
        [new_szr, new_theta_r, new_theta_m] = flip_one(szr, theta_r, theta_m, conj(w), conj(u));
        new_szl = szl;
        new_theta_l = theta_l;
    end
    
    % Randomly select a q spin to flip
    [new_qsz, new_qtheta, new_qtheta_m] = flip_both(qsz, qtheta, qtheta_m, w, u);
    
    % Compute acceptance probability
    ap = min([1, abs(density(a, new_theta_l, new_theta_r, new_theta_m, new_szl, new_szr)/ ...
                     density(a, theta_l, theta_r, theta_m, szl, szr))^2]);
                 
    qap = min([1, abs(density(a, new_qtheta, conj(new_qtheta), new_qtheta_m, new_qsz, new_qsz)/ ...
                      density(a, qtheta, conj(qtheta), qtheta_m, qsz, qsz))^2]);
    
    % Accept state according to the acceptance probability
    if rand() < ap
        szl = new_szl;
        szr = new_szr;
        theta_l = new_theta_l;
        theta_r = new_theta_r;
        theta_m = new_theta_m;
    end
    
    if rand() < qap
        qsz = new_qsz;
        qtheta = new_qtheta;
        qtheta_m = new_qtheta_m;
    end
end
end

function [new_sz, new_theta, new_theta_m] = flip_one(sz, theta, theta_m, w, u)
n = size(sz, 1);
new_sz = sz;
flip_idx = ceil(n*rand());
new_sz(flip_idx) = -1*sz(flip_idx);

new_theta = theta - 2*w(:,flip_idx)*sz(flip_idx);
new_theta_m = theta_m - 2*u(:,flip_idx)*sz(flip_idx);
end

function [new_sz, new_theta, new_theta_m] = flip_both(sz, theta, theta_m, w, u)
n = size(sz, 1);
new_sz = sz;
flip_idx = ceil(n*rand());
new_sz(flip_idx) = -1*sz(flip_idx);

new_theta = theta - 2*w(:,flip_idx)*sz(flip_idx);
new_theta_m = theta_m - 2*u(:,flip_idx)*sz(flip_idx) - 2*conj(u(:,flip_idx))*sz(flip_idx);
end