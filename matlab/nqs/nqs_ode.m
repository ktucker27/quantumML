function dydt = nqs_ode(t, y, n, m, eloc, num_samps, num_steps)

global tmax

if t == 0
    tmax = 0;
end

if t > tmax
    disp(['Time: ', num2str(t)]);
    tmax = tmax + .001;
end

a = y(1:n);
b = y(n+1:n+m);
w = reshape(y(n+m+1:end), [m,n]);

% Get a matrix of spin samples
sz_samps = zeros(n, num_samps);
theta_mat = zeros(m, num_samps);
dw_mat = zeros(m*n, num_samps);
eloc_vec = zeros(num_samps, 1);
for j=1:num_samps
    sz = nqs_sample(@nqs_wave, a, b, w, num_steps);
    theta = b + w*sz;
    sz_samps(:,j) = sz;
    theta_mat(:,j) = theta;
    dw_mat(:,j) = reshape(tanh(theta)*sz', [m*n,1]);
    eloc_vec(j,1) = feval(eloc, @nqs_wave, a, b, w, sz);
end

% Construct derivative structures
omat = [sz_samps;tanh(theta_mat);dw_mat];
ovec = sum(omat,2)/num_samps;
eloc_ev = sum(eloc_vec)/num_samps;

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

dydt = -1i*minresqlp(Skk,Fk);