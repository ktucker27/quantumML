function l = thermal_loss_slq(H, e0, beta, numsteps, nv)

%rho = expm(-beta*H);
%z = trace(rho);

fun = @(x)(exp(-beta*x));
z = slq(H, fun, numsteps, nv);

%l = (trace(rho*H)/z - e0)^2;

fun2 = @(x)(x.*exp(-beta*x));
z2 = slq(H, fun2, numsteps, nv);

l = (z2/z - e0)^2;