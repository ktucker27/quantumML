function l = thermal_loss(H, e0, beta)

rho = expm(-beta*H);
z = trace(rho);
l = (trace(rho*H)/z - e0)^2;