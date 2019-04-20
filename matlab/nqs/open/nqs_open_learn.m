function [a, b, c, w, u, norms] = nqs_open_learn(a, b, c, w, u, lloc, density, eta, num_iterations, num_samps, num_steps, L)
% nqs_open_learn Learn the parameters (a, b, c, w, u) of a neural quantum 
%                state (NQS) for a given local energy Liouvillian lloc

n = size(a,1);
m = size(b,1);
l = size(c,1);

% Regularization parameters
l0 = 100;
lb = 0.9;
lmin = 1e-4;

norms = zeros(num_iterations,1);
for i=1:num_iterations
    if mod(i,10) == 0
        disp(['i = ', num2str(i)]);
    end
    
    if size(L,1) > 0
        [~,~,~,rho_vec] = nqs_density_mat(a, b, c, w, u);
        norms(i,1) = norm(L*rho_vec);
    end
    
    [S,F] = nqs_open_sf(a, b, c, w, u, lloc, density, num_samps, num_steps);
    
    % Apply regularization to S matrix
    S = S + diag(lambda(l0, lb, i, lmin)*diag(S));
    
    % Compute update
    %update = -eta*(S\F);
    update = -eta*minresqlp(S,F);
    
    if isnan(norm(update))
        disp(['Received NaN at i = ', num2str(i)]);
        break;
    end
    
    % Update parameters according to gradient descent
    a = a + update(1:n,1) + 1i*update(n+1:2*n,1);
    sidx = 2*n+1;
    b = b + update(sidx:sidx+m-1,1) + 1i*update(sidx+m:sidx+2*m-1,1);
    sidx = sidx + 2*m;
    c = c + update(sidx:sidx+l-1,1);
    sidx = sidx + l;
    w = w + reshape(update(sidx:sidx+m*n-1,1),[m,n]) ...
          + 1i*reshape(update(sidx+m*n:sidx+2*m*n-1,1),[m,n]);
    sidx = sidx + 2*m*n;
    u = u + reshape(update(sidx:sidx+l*n-1,1),[l,n]) ...
          + 1i*reshape(update(sidx+l*n:end,1),[l,n]);
end
end

function l = lambda(l0, lb, p, lmin)
l = max([l0*lb^p, lmin]);
end