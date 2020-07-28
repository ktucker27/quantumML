function mps_out = apply_mpo(mpo, mps)

obc = mps.obc;
if mpo.obc ~= obc
    error('MPO and MPS must have same type of boundary conditions');
end

n = mps.num_sites();

ms = {};
for ii=1:n
    if ii == n && obc
        T = mpo.tensors{ii}.contract(mps.tensors{ii}, [3,2]);
        T2 = T.group({[1,3],2});
    else
        T = mpo.tensors{ii}.contract(mps.tensors{ii}, [4,3]);
        T2 = T.group({[1,4],[2,5],3});
    end
    
    cat(2, ms, {T2});
end

mps_out = MPS(ms);