function [t,psi] = nqs_tvmc(a0, b0, w0, dt, tf)

num_samps = 1000;
h = 2;
eloc = @(wave, a, b, w, sz)(tfi_eloc(wave, a, b, w, sz, h));
n = size(a0,1);
m = size(b0,1);
num_steps = 2*n;

odes = @(t,y)(nqs_ode(t,y,n,m,eloc,num_samps,num_steps));

y0 = [a0;b0;reshape(w0,[m*n,1])];

if dt > 0
    tvec = (0:dt:tf)';
else
    tvec = [0;tf];
end

options = odeset('RelTol',1e-4,'AbsTol',1e-5);
[t,y] = ode45(odes,tvec,y0,options);

size(t)
size(y)

psi = zeros(2^n,size(t,1));
for i=1:size(t,1)
    a = y(i,1:n)';
    b = y(i,n+1:n+m)';
    w = reshape(y(i,n+m+1:end)',[m,n]);
    [~,psi(:,i)] = nqs_wave_vec(a, b, w);
end