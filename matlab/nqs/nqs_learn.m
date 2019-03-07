function [a, b, w, eloc_evs] = nqs_learn(a, b, w, eloc, wave, eta, num_iterations, num_samps, num_steps)
% nqs_learn Learn the parameters (a, b, w) of a neural quantum state (NQS)
%           for a given local energy operator eloc

n = size(a,1);
m = size(b,1);

% Regularization parameters
l0 = 100;
lb = 0.9;
lmin = 1e-4;

eloc_evs = zeros(num_iterations,1);
for i=1:num_iterations
    %if mod(i,10) == 0
    %    disp(['i = ', num2str(i)]);
    %end
    
    % Get a matrix of spin samples
    sz_samps = zeros(n, num_samps);
    theta_mat = zeros(m, num_samps);
    dw_mat = zeros(m*n, num_samps);
    eloc_vec = zeros(num_samps, 1);
    for j=1:num_samps
        sz = nqs_sample(wave, a, b, w, num_steps);
        theta = b + w*sz;
        sz_samps(:,j) = sz;
        theta_mat(:,j) = theta;
        dw_mat(:,j) = reshape(tanh(theta)*sz', [m*n,1]);
        eloc_vec(j,1) = feval(eloc, wave, a, b, w, sz);
    end
    
    % Construct derivative structures
    omat = [sz_samps;tanh(theta_mat);dw_mat];    
    ovec = sum(omat,2)/num_samps;
    eloc_ev = sum(eloc_vec)/num_samps;
    eloc_evs(i,1) = eloc_ev;
    
    % Build the stochastic reconfiguration matrix and force vector
    nk = size(omat, 1);
    Skk = zeros(nk, nk);
    Fk = zeros(nk, 1);
    for j=1:num_samps
        Skk = Skk + conj(omat(:,j))*omat(:,j).';
        Fk = Fk + eloc_vec(j,1)*conj(omat(:,j));
    end
    Skk = Skk/num_samps - conj(ovec)*ovec.';
    Fk = Fk/num_samps - eloc_ev*conj(ovec);
    
    % Apply regularization to S matrix
    Skk = Skk + diag(lambda(l0, lb, i, lmin)*diag(Skk));
    
    % Compute update
    %update = -eta*(Skk\Fk);
    update = -eta*minresqlp(Skk,Fk);
    
    if isnan(norm(update))
        disp(['Received NaN at i = ', num2str(i)]);
        break;
    end
    
    % Update parameters according to gradient descent
    a = a + update(1:n,1);
    b = b + update(n+1:n+m,1);
    w = w + reshape(update(n+m+1:end,1),[m,n]);
end
end

function l = lambda(l0, lb, p, lmin)
l = max([l0*lb^p, lmin]);
end