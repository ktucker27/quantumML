function mps_out = apply_mpo(mpo, mps)

n = mps.num_sites();

ms = {};
for ii=1:n
    T = mpo.tensors{ii}.contract(mps.tensors{ii}, [4,3]);
    T2 = T.group({[1,4],[2,5],3});
    
    cat(2, ms, {T2});
end

mps_out = MPS(ms);