function O = local_op_to_prod(Ol, part, n, m)

idn = sparse(eye(n));
id = sparse(1);
for ii=1:part-1
    id = kron(id, idn);
end

O = kron(id, Ol);

id = sparse(1);
for ii=part+1:m
    id = kron(id, idn);
end

O = kron(O, id);
