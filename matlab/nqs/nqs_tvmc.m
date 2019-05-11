function [t,psi,y,evs] = nqs_tvmc(a0, b0, w0, dt, tf)

num_samps = 1000;
% h = 1;
% eloc = @(wave, a, b, w, sz)(tfi_eloc(wave, a, b, w, sz, h));
n = size(a0,1);
m = size(b0,1);
num_steps = 4*n;

%eloc = @(wave, a, b, w, sz)(afh_eloc(wave, a, b, w, sz, n, 1));
d = 3;
eloc = @(wave, a, b, w, sz)(exc_eloc(wave, a, b, w, sz, d));

odes = @(t,y)(nqs_ode(t,y,n,m,eloc,num_samps,num_steps));

y0 = [a0;b0;reshape(w0,[m*n,1])];

if dt > 0
    tvec = (0:dt:tf)';
else
    tvec = [0;tf];
end

options = odeset('RelTol',1e-3,'AbsTol',1e-3);
[t,y] = ode45(odes,tvec,y0,options);

if n < 16
    psi = zeros(2^n,size(t,1));
else
    psi = [];
end
evs = zeros(1,size(t,1));
y = y.';
for i=1:size(t,1)
    a = y(1:n,i);
    b = y(n+1:n+m,i);
    w = reshape(y(n+m+1:end,i),[m,n]);
    if n < 16
        [~,psi(:,i)] = nqs_wave_vec(a, b, w);
    end
    evs(1,i) = nqs_ev(@sx_oloc, @nqs_wave, a, b, w, 5*num_samps, num_steps);
end