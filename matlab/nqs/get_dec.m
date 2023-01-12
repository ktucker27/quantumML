function d = get_dec(b)

n = max(size(b));
b = (-1/2)*(b - 1);
d = sum(2.^((n-1):-1:0).*b);