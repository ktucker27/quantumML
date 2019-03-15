function b = get_bi(d, num_dig)

b = [];
while d > 0
    b = [b;mod(d,2)];
    d = floor(d/2);
end

b = flip(b);
if nargin > 1 && num_dig > size(b,1)
    b = [zeros(num_dig - size(b,1),1);b];
end