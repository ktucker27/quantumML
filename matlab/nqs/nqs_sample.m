function sz = nqs_sample(wave, a, b, w, num_steps)
% nqs_sample Generate a vector of spins sampled from a distribution
%            governed by the NQS wave function determined by the set of
%            parameters (a, b, w)
%
% INPUT:     
%            a - Nx1 vector of biases for the visible variables 
%            b - Mx1 vector of biases for the hidden variables
%            w - MxN matrix of weights between hidden and visible variables
%    num_steps - Number of steps to take in the Markov process
%
% OUTPUT:   
%           sz - Vector of spins sampled according to the wave function
%                given by the parameters (a, b, W)

n = size(w,2);

% Initialize to spin-up
sz = ones(n,1);

% Initialize theta
theta = b + w*sz;

% Walk to a state with nonzero probability
while wave(a, theta, sz) == 0
    flip_idx = ceil(n*rand());
    sz(flip_idx) = -1*sz(flip_idx);
    
    theta = theta - 2*w(:,flip_idx)*sz(flip_idx);
end

% Perform a random walk on the spins according to Metropolis-Hastings,
% where the simple transitions are taken to be a single spin flip with
% equal probability
for step_idx=1:num_steps
    % Randomly select a spin to flip
    new_sz = sz;
    flip_idx = ceil(n*rand());
    new_sz(flip_idx) = -1*sz(flip_idx);
    
    new_theta = theta - 2*w(:,flip_idx)*sz(flip_idx);
    
    % Compute acceptance probability
    ap = min([1, abs(wave(a, new_theta, new_sz)/wave(a, theta, sz))^2]);
    
    % Accept state according to the acceptance probability
    if rand() < ap
        sz = new_sz;
        theta = new_theta;
    end
end