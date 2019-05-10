function [S,F,lloc_ev] = nqs_open_sf(a, b, c, w, u, lloc, density, num_samps, num_steps)
%nqs_open_sf Returns the stochastic reconfiguration and force vector for
%            the density operator determined by the set of parameters
%            (a, b, c, w, u)

S = [];
F = [];

n = size(a,1);
m = size(b,1);
l = size(c,1);

% Get a matrix of spin samples
szl_samps = zeros(n, num_samps);
szr_samps = zeros(n, num_samps);
xib_l_mat = zeros(m, num_samps);
xib_r_mat = zeros(m, num_samps);
xic_mat = zeros(l, num_samps);
dw_mat = zeros(m*n, num_samps);
du_mat = zeros(l*n, num_samps);
lloc_vec = zeros(num_samps, 1);
for j=1:num_samps
    [szl, szr, ~] = nqs_open_sample(density, a, b, c, w, u, num_steps);
    xib_l = tanh(b + w*szl);
    xib_r = conj(tanh(b + w*szr)); % Conjugate now to save time later
    xic = tanh(c + conj(c) + u*szl + conj(u)*szr);
    szl_samps(:,j) = szl;
    szr_samps(:,j) = szr;
    xib_l_mat(:,j) = xib_l;
    xib_r_mat(:,j) = xib_r;
    xic_mat(:,j) = xic;
    dw_mat(:,j) = reshape(xib_l*szl' + xib_r*szr', [m*n,1]) + ...
               1i*reshape(xib_l*szl' - xib_r*szr', [m*n,1]);
    du_mat(:,j) = reshape(xic*(szl + szr)', [l*n,1]) + ...
               1i*reshape(xic*(szl - szr)', [l*n,1]);
    lloc_vec(j,1) = feval(lloc, density, a, b, c, w, u, szl, szr);
end

% Construct derivative structures
num_derivs = 2*n + 2*m + l + 2*m*n + 2*l*n;
omat = zeros(num_derivs, num_samps);
sidx = 1;
% a
omat(sidx:sidx+n-1,:) = szl_samps + szr_samps;   % Re
sidx = sidx + n;
omat(sidx:sidx+n-1,:) = (szl_samps - szr_samps); % Im
sidx = sidx + n;
% b
omat(sidx:sidx+m-1,:) = xib_l_mat + xib_r_mat;   % Re
sidx = sidx + m;
omat(sidx:sidx+m-1,:) = (xib_l_mat - xib_r_mat); % Im
sidx = sidx + m;
% c
omat(sidx:sidx+l-1,:) = 2*xic_mat;
sidx = sidx + l;
% W
omat(sidx:sidx+m*n-1,:) = real(dw_mat); % Re
sidx = sidx + m*n;
omat(sidx:sidx+m*n-1,:) = imag(dw_mat); % Im
sidx = sidx + m*n;
% U
omat(sidx:sidx+l*n-1,:) = real(du_mat); % Re
sidx = sidx + l*n;
omat(sidx:sidx+l*n-1,:) = imag(du_mat); % Im
sidx = sidx + l*n;

if sidx - 1 ~= num_derivs
    disp(['ERROR - Incorrect number of derivative values set: ', ...
        num2str(sidx - 1), '. Expected: ', num2str(num_derivs)]);
    return;
end

%ovec = sum(omat,2)/num_samps;
%lloc_ev = sum(lloc_vec)/num_samps;

% Build the stochastic reconfiguration matrix and force vector
nk = size(omat, 1);
S = zeros(nk, nk);
F = zeros(nk, 1);
for j=1:num_samps
    S = S + conj(omat(:,j))*omat(:,j).';
    F = F + lloc_vec(j,1)*conj(omat(:,j));
end
S = real(S)/num_samps;
F = real(F)/num_samps;
lloc_ev = sum(abs(lloc_vec))/num_samps;

end

